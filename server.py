#!/usr/bin/env python3
"""
openwebui-kb-mcp — an MCP tool server that routes big-document work through
Open WebUI's RAG pipeline without the overhead of Knowledge Bases.

Workflow it gives the LLM:
  1. fetch_url          -> Fetches URL content or downloads/indexes large files.
                           (Returns text, a file_id, or a collection_name)
  2. semantic_search    -> Queries the indexed files or collections in Open WebUI.

Config (environment variables, or .env file next to this script):
  OPENWEBUI_URL           base URL, e.g. http://localhost:3000
  OPENWEBUI_API_KEY       sk-... API key (Settings > Account)
  OPENWEBUI_DEFAULT_MODEL model id used by query_documents (optional)
  KB_PROCESS_TIMEOUT      seconds to wait for embedding (default 600)
  KB_MAX_FILE_BYTES       download size cap (default 2 GiB)
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import tempfile
import time
import uuid
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Minimal .env loader (KEY=VALUE lines) so the MCP server can be
    configured without shell exports. Real environment wins."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

OPENWEBUI_URL = os.environ.get("OPENWEBUI_URL", "http://localhost:3000").rstrip("/")
OPENWEBUI_API_KEY = os.environ.get("OPENWEBUI_API_KEY", "")
DEFAULT_MODEL = os.environ.get("OPENWEBUI_DEFAULT_MODEL", "")
PROCESS_TIMEOUT = int(os.environ.get("KB_PROCESS_TIMEOUT", "600"))
POLL_INTERVAL = float(os.environ.get("KB_POLL_INTERVAL", "2"))
MAX_FILE_BYTES = int(os.environ.get("KB_MAX_FILE_BYTES", str(2 * 1024 * 1024 * 1024)))

if not OPENWEBUI_API_KEY:
    raise SystemExit(
        "OPENWEBUI_API_KEY is not set. Create a key in Open WebUI under "
        "Settings > Account (prefix sk-) and set it as an environment variable "
        "or in a .env file next to server.py."
    )


def base_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {OPENWEBUI_API_KEY}"}


def _api_error(resp: httpx.Response) -> Exception:
    try:
        detail = resp.json().get("detail") if resp.headers.get("content-type", "").startswith("application/json") else None
    except Exception:
        detail = None
    return Exception(
        f"Open WebUI API error {resp.status_code} for {resp.request.method} "
        f"{resp.url}: {detail or resp.text[:500]}"
    )


class Client:
    """Thin synchronous httpx wrapper around the Open WebUI API."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url = path if path.startswith("http") else f"{OPENWEBUI_URL}{path}"
        headers = {**base_headers(), **kwargs.pop("headers", {})}
        with httpx.Client(follow_redirects=True, timeout=kwargs.pop("timeout", self.timeout)) as http:
            resp = http.request(method, url, headers=headers, **kwargs)
            if resp.status_code >= 400:
                raise _api_error(resp)
            return resp

    def get(self, path: str, **kwargs: Any) -> Any:
        return self.request("GET", path, **kwargs).json()

    def post(self, path: str, **kwargs: Any) -> Any:
        return self.request("POST", path, **kwargs).json()


client = Client()


def wait_for_file_processing(file_id: str, timeout: int | None = None) -> dict[str, Any]:
    """Poll until RAG processing (extraction + embedding) is done.

    Prefers GET /api/v1/files/{id}/process/status (current Open WebUI).
    Falls back to GET /api/v1/files/{id} and checking for `data.content`
    on older instances that lack the status endpoint.
    """
    timeout = timeout or PROCESS_TIMEOUT
    deadline = time.monotonic() + timeout
    status_path = f"/api/v1/files/{file_id}/process/status"
    legacy = False
    while True:
        if not legacy:
            try:
                data = client.get(status_path)
            except Exception as exc:
                if "404" in str(exc):
                    legacy = True
                else:
                    raise
        if legacy:
            data = client.get(f"/api/v1/files/{file_id}")
            if (data.get("data") or {}).get("content"):
                return {"status": "completed", "file_id": file_id}
        else:
            status = data.get("status")
            if status == "completed":
                return data
            if status == "failed":
                raise Exception(f"RAG processing failed for file {file_id}: {data.get('error')}")
        if time.monotonic() >= deadline:
            raise Exception(
                f"Timed out after {timeout}s waiting for file {file_id} to finish processing."
            )
        time.sleep(POLL_INTERVAL)


def _content_to_text(content: Any) -> str:
    """Handle both str content and OpenAI-parts content arrays."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in (None, "text"):
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# FastMCP server + tools
# ---------------------------------------------------------------------------

mcp = FastMCP("openwebui-direct-files")


@mcp.tool()
def query_fetched_file(query: str, file_ids: list[str] = [], collection_names: list[str] = [], top_k: int = 10) -> str:
    """Perform a pure semantic search on specific documents within the Open WebUI vector database.
    
    This is NOT a global web search or global database search. You MUST provide at least one
    `file_id` or `collection_name` (obtained via the `fetch_url` tool) to scope your search.
    If you want to search the internet, use the `search_web` tool instead.
    
    Returns the raw text chunks matching the query. Bypasses the internal LLM completely.
    """
    if not query:
        return json.dumps({"ok": False, "error": "query is required"}, ensure_ascii=False)
    if not file_ids and not collection_names:
        return json.dumps({
            "ok": False, 
            "error": "You must provide at least one file_id or collection_name to search. This tool does not support global search. If you are trying to search the internet, please use the `search_web` tool instead."
        }, ensure_ascii=False)

    all_chunks = []
    
    try:
        # Search individual files
        for file_id in file_ids:
            if not file_id.startswith("file-") and not file_id.startswith("web-search-"):
                collection_name = f"file-{file_id}"
            else:
                collection_name = file_id
            payload = {"query": query, "collection_name": collection_name, "k": top_k, "r": -1.0}
            resp = client.post("/api/v1/retrieval/query/doc", json=payload, timeout=60.0)
            if resp is None:
                continue
                
            docs = resp.get("documents", []) or resp.get("chunks", []) or resp.get("data", [])
            metas = resp.get("metadatas", [])
            dists = resp.get("distances", [])
            
            if docs and isinstance(docs[0], list): docs = docs[0]
            if metas and isinstance(metas[0], list): metas = metas[0]
            if dists and isinstance(dists[0], list): dists = dists[0]
            
            for idx, doc in enumerate(docs):
                if not doc or str(doc).strip() == "[]": continue
                meta = metas[idx] if idx < len(metas) else {}
                dist = dists[idx] if idx < len(dists) else "N/A"
                all_chunks.append({"text": doc, "meta": meta, "dist": dist})
            
        # Search collections
        if collection_names:
            payload = {"query": query, "collection_names": collection_names, "k": top_k, "r": -1.0}
            resp = client.post("/api/v1/retrieval/query/collection", json=payload, timeout=60.0)
            if resp is not None:
                docs = resp.get("documents", []) or resp.get("chunks", []) or resp.get("data", [])
                metas = resp.get("metadatas", [])
                dists = resp.get("distances", [])
                
                if docs and isinstance(docs[0], list): docs = docs[0]
                if metas and isinstance(metas[0], list): metas = metas[0]
                if dists and isinstance(dists[0], list): dists = dists[0]
                
                for idx, doc in enumerate(docs):
                    if not doc or str(doc).strip() == "[]": continue
                    meta = metas[idx] if idx < len(metas) else {}
                    dist = dists[idx] if idx < len(dists) else "N/A"
                    all_chunks.append({"text": doc, "meta": meta, "dist": dist})
            
        # Format chunks safely to prevent token explosion
        results = []
        for i, chunk in enumerate(all_chunks[:top_k]):  # Limit to requested chunks
            if isinstance(chunk, dict) and "text" in chunk:
                # New format with metadata
                text = chunk["text"]
                if isinstance(text, dict):
                    text = text.get("document", text.get("content", str(text)))
                else:
                    text = str(text)
                meta = chunk["meta"]
                # Prefer source_url/url for web fetches, fallback to file source
                source = meta.get("source_url", meta.get("url", meta.get("source", "Unknown")))
                loc = meta.get("loc", "")
                if loc: source += f" ({loc})"
                dist = chunk.get("dist", "N/A")
                if isinstance(dist, float): dist = round(dist, 4)
                results.append(f"--- Chunk {i+1} (Source: {source} | Score: {dist}) ---\n{text[:2000]}")
            else:
                # Fallback for old format
                if isinstance(chunk, dict):
                    text = chunk.get("document", chunk.get("content", str(chunk)))
                else:
                    text = str(chunk)
                results.append(f"--- Chunk {i+1} ---\n{text[:2000]}") # Cap each chunk
            
        if not results:
            return json.dumps({
                "ok": True, 
                "answer": "No relevant text chunks found in the database.",
                "debug_raw_response": resp if (file_ids or collection_names) else "No files/collections queried"
            }, ensure_ascii=False)
            
        answer_text = "\n\n".join(results)
        return json.dumps({
            "ok": True,
            "answer": answer_text
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"Semantic search failed: {str(e)}"})



@mcp.tool()
def search_web(query: str) -> str:
    """Search the web for information using Open WebUI's configured search engines.
    Returns a list of search results or a collection name.
    """
    if not query:
        raise Exception("query is required")
        
    try:
        resp = client.post("/api/v1/retrieval/process/web/search", json={"queries": [query]}, timeout=120.0)
        
        if resp is None:
            return json.dumps({"ok": False, "error": "Open WebUI returned None. Web search may be disabled or failed."}, ensure_ascii=False)
            
        return json.dumps({
            "ok": True,
            "query": query,
            "results": resp
        }, ensure_ascii=False)
        
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg and "No results found" in error_msg:
            return json.dumps({"ok": False, "error": "Search engine returned 0 results. Please try a different query."}, ensure_ascii=False)
        return json.dumps({"ok": False, "error": f"Web search failed: {e}"}, ensure_ascii=False)

@mcp.tool()
def fetch_url(url: str) -> str:
    """Fetch the text content of a URL (web page or PDF).
    If the document is too large (over 8,000 tokens), it will automatically index the document
    and return a collection_name or file_id which you can pass to `query_fetched_file`.
    """
    if not url:
        raise Exception("url is required")

    try:
        # Quick check to prevent Open WebUI from swallowing 404s into generic 400s
        with httpx.Client(follow_redirects=True, timeout=5.0) as http:
            head_resp = http.head(url)
            
            # Automatically update the URL if we were redirected (fixes Open WebUI 301 scraping bugs)
            if str(head_resp.url) != url:
                url = str(head_resp.url)
                
            if head_resp.status_code == 404:
                return json.dumps({"ok": False, "error": f"The URL {url} does not exist (404 Not Found). You likely guessed a broken link. Please use `search_web` to find the correct URL."}, ensure_ascii=False)
    except Exception:
        pass

    try:
        # First, attempt to fetch the text without processing
        resp = client.post(
            "/api/v1/retrieval/process/url",
            params={"process": "false"},
            json={"url": url},
            timeout=120.0
        )
        
        # If it's a web page or youtube video and returned content, try to return it directly
        if resp.get("type") in ("web", "youtube") and resp.get("content"):
            text = resp.get("content")
            
            try:
                parsed = json.loads(text)
                text = json.dumps(parsed, indent=2)
            except Exception:
                pass
                
            try:
                import tiktoken
                encoding = tiktoken.get_encoding("cl100k_base")
                token_count = len(encoding.encode(text, disallowed_special=()))
            except ImportError:
                token_count = len(text) // 4
                
            if token_count <= 8000:
                return json.dumps({
                    "ok": True,
                    "url": url,
                    "token_count": token_count,
                    "content": text[:30000]
                }, ensure_ascii=False)

        # Otherwise (too big, or it's a file), let Open WebUI fully process it
        index_resp = client.post(
            "/api/v1/retrieval/process/url",
            params={"process": "true"},
            json={"url": url},
            timeout=300.0
        )
        
        item_type = index_resp.get("type")
        
        if item_type in ("web", "youtube"):
            collection_name = index_resp.get("collection_name")
            return json.dumps({
                "ok": True,
                "url": url,
                "indexed": True,
                "collection_name": collection_name,
                "hint": f"Document was automatically indexed.\nYou MUST now call the `query_fetched_file` tool and pass exactly `collection_names=[\"{collection_name}\"]` to search its contents."
            }, ensure_ascii=False)
        else:
            file_data = index_resp.get("file", {})
            file_id = file_data.get("id")
            if file_id:
                wait_for_file_processing(file_id)
                return json.dumps({
                    "ok": True,
                    "url": url,
                    "indexed": True,
                    "file_id": file_id,
                    "hint": f"Document was automatically indexed.\nYou MUST now call the `query_fetched_file` tool and pass exactly `file_ids=[\"{file_id}\"]` to search its contents."
                }, ensure_ascii=False)
            else:
                return json.dumps({"ok": False, "error": f"Failed to get file_id from response: {index_resp}"}, ensure_ascii=False)
                
    except Exception as e:
        error_msg = str(e)
        if "400" in error_msg and "Error processing URL" in error_msg:
             return json.dumps({"ok": False, "error": f"Open WebUI failed to process the URL. Since we know it's not a 404, the site is likely blocking access (e.g. 403 Forbidden, bot protection) or the file format is unsupported. Please use `search_web` to find alternative sources."}, ensure_ascii=False)
        return json.dumps({"ok": False, "error": f"Fetch failed: {e}"}, ensure_ascii=False)


@mcp.tool()
def grep_fetched_file(file_id: str, query: str, is_regex: bool = False, ignore_case: bool = True, context_lines: int = 2) -> str:
    """Exact-text search (grep) on a file stored in Open WebUI.
    
    Use this for precise exact-match searches that semantic search struggles with.
    
    Args:
        file_id: The ID of the file in Open WebUI (e.g., from download_file or search).
        query: The string or regex to search for.
        is_regex: If True, treats query as a regular expression.
        ignore_case: If True, makes the search case-insensitive.
        context_lines: Number of lines to show before and after each match.
    """
    if not file_id or not query:
        return json.dumps({"ok": False, "error": "file_id and query are required"})
        
    try:
        # Fetch the extracted text content from Open WebUI
        # /data/content returns the extracted text, while /content returns the raw binary file
        content = client.get(f"/api/v1/files/{file_id}/data/content")
        
        if not content:
            return json.dumps({"ok": False, "error": f"File {file_id} not found or has no extracted text content."})
            
        if isinstance(content, dict) and "content" in content:
            text = content["content"]
        elif isinstance(content, dict) and "data" in content and isinstance(content["data"], dict):
            text = content["data"].get("content", "")
        else:
            text = str(content)
            
        # Try to prettify JSON to break up minified single-line responses
        try:
            parsed = json.loads(text)
            text = json.dumps(parsed, indent=2)
        except Exception:
            pass
            
        lines = text.split("\n")
        flags = re.IGNORECASE if ignore_case else 0
        
        if not is_regex:
            query = re.escape(query)
            
        pattern = re.compile(query, flags)
        
        results = []
        matches_count = 0
        
        # Simple sliding window for context
        for i, line in enumerate(lines):
            if pattern.search(line):
                matches_count += 1
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                
                match_block = []
                for j in range(start, end):
                    prefix = "> " if j == i else "  "
                    line_content = lines[j]
                    
                    # Truncate extremely long lines (e.g. from minified JSON)
                    if len(line_content) > 300:
                        if j == i:
                            # Try to show context around the actual match
                            m = pattern.search(line_content)
                            if m:
                                m_start = max(0, m.start() - 150)
                                m_end = min(len(line_content), m.end() + 150)
                                line_content = ("..." if m_start > 0 else "") + line_content[m_start:m_end] + ("..." if m_end < len(line_content) else "")
                            else:
                                line_content = line_content[:300] + "..."
                        else:
                            line_content = line_content[:300] + "..."
                            
                    match_block.append(f"{j+1:04d} {prefix} {line_content}")
                    
                results.append("\n".join(match_block))
                
        if not results:
            return json.dumps({
                "ok": True,
                "matches": 0,
                "results": f"No matches found for '{query}' in file {file_id}. Note: grep is line-based. Your regex must match within a single line."
            })
            
        # Limit to first 20 matches to avoid blowing up context window
        cap_msg = ""
        if len(results) > 20:
            cap_msg = f"\n...and {len(results) - 20} more matches omitted."
            results = results[:20]
            
        output = f"Found {matches_count} matches in file {file_id}:\n\n" + "\n---\n".join(results) + cap_msg
        
        if len(output) > 20000:
            output = output[:20000] + "\n\n...[OUTPUT TRUNCATED DUE TO EXTREME LENGTH]..."
            
        return json.dumps({
            "ok": True,
            "matches": matches_count,
            "results": output
        })
        
    except Exception as e:
        return json.dumps({"ok": False, "error": f"Failed to grep file: {e}"})


# ---------------------------------------------------------------------------
# Entrypoint — stdio (default) or streamable-http (Open WebUI native MCP)
# ---------------------------------------------------------------------------

def main() -> None:
    transport = os.environ.get("MCP_TRANSPORT", "stdio").lower()
    if transport in ("streamable-http", "http", "streamable_http"):
        host = os.environ.get("MCP_HTTP_HOST", "0.0.0.0")
        port = int(os.environ.get("MCP_HTTP_PORT", "8766"))
        mcp.settings.host = host
        mcp.settings.port = port
        try:
            mcp.run(transport="streamable-http")
        except (ImportError, ValueError, TypeError) as exc:
            raise SystemExit(
                f"streamable-http transport unavailable ({exc}). Install the latest "
                f"mcp package, or run with MCP_TRANSPORT=stdio and front the server "
                f"with the mcpo proxy for Open WebUI."
            )
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()