#!/usr/bin/env python3
"""
openwebui-kb-mcp — an MCP tool server that routes big-document work through
Open WebUI's RAG pipeline without the overhead of Knowledge Bases.

Workflow it gives the LLM:
  1. download_file      -> download URL to disk (streamed, size-capped),
                           POST /api/v1/files/ (process=true)
                           poll GET /api/v1/files/{id}/process/status
                           (Returns a file_id)
  2. process_web_url    -> POST /api/v1/retrieval/process/web (no download)
                           (Returns a collection_name)
  3. query_documents    -> POST /api/chat/completions with
                           files=[{type:"file", id:<file id>}, {type:"collection", id:<collection name>}]

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


def extract_filename(url: str) -> str:
    name = os.path.basename(urlparse(url).path)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name) or "download"
    return name


def download_to_disk(url: str) -> tuple[str, str, int, str]:
    """Stream a URL to a temp file. Returns (path, filename, size_bytes, content_type)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise Exception(f"Only http/https URLs are supported, got: {url}")
    filename = extract_filename(url)
    fd, tmp_path = tempfile.mkstemp(prefix="owui-kb-", suffix=os.path.splitext(filename)[1] or ".bin")
    os.close(fd)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    size = 0
    content_type = ""
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=600.0)) as http:
        with http.stream("GET", url, headers=headers) as resp:
            if resp.status_code >= 400:
                raise Exception(f"Download failed with HTTP {resp.status_code} for {url}")
            content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
            for chunk in resp.iter_bytes(1024 * 256):
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    raise Exception(
                        f"File exceeds KB_MAX_FILE_BYTES ({MAX_FILE_BYTES}); aborting download of {url}"
                    )
                with open(tmp_path, "ab") as fh:
                    fh.write(chunk)
    if size == 0:
        raise Exception(f"Download of {url} produced an empty file.")
    return tmp_path, filename, size, content_type


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
def download_file(url: str, wait: bool = True) -> str:
    """Download a file (PDF, doc, etc.) and upload it directly to Open WebUI.

    Use this instead of fetching big documents into the context window.
    The file is processed and you will receive a file_id, which you can
    pass directly to semantic_search.
    """
    if not url:
        raise Exception("url is required")

    tmp_path, filename, size, ctype = download_to_disk(url)
    try:
        metadata = {
            "process": True,
            "source": "openwebui-mcp",
            "source_url": url,
        }
        mime = ctype or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        with open(tmp_path, "rb") as fh:
            payload = client.post(
                "/api/v1/files/",
                files={"file": (filename, fh, mime), "metadata": (None, json.dumps(metadata), "application/json")},
                params={"process": "true", "process_in_background": "false"},
                timeout=600.0,
            )
        file_id = payload.get("id")
        if not file_id:
            raise Exception(f"File upload returned no id: {payload}")

        processing = {"file_id": file_id, "status": "pending"}
        if wait:
            processing = wait_for_file_processing(file_id)

        return json.dumps(
            {
                "ok": True,
                "filename": filename,
                "size_bytes": size,
                "file_id": file_id,
                "processing": processing,
                "hint": "Ready: call semantic_search with file_ids=[\"" + file_id + "\"] to search it.",
            },
            ensure_ascii=False,
            default=str,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@mcp.tool()
def process_web_url(url: str) -> str:
    """Ingest a web page directly into Open WebUI.

    Best for HTML documentation. Returns a collection_name which you
    can pass to semantic_search.
    """
    if not url:
        raise Exception("url is required")

    result = client.post(
        "/api/v1/retrieval/process/web",
        params={"process": "true", "overwrite": "true"},
        json={"url": url},
        timeout=300.0,
    )
    
    collection_name = result.get("collection_name")
    
    return json.dumps(
        {
            "ok": True,
            "url": url,
            "collection_name": collection_name,
            "upstream_result": {k: v for k, v in (result or {}).items() if k in ("count", "documents", "success", "error")},
            "hint": "Ready: call semantic_search with collection_names=[\"" + collection_name + "\"] to search it.",
        },
        ensure_ascii=False,
        default=str,
    )


@mcp.tool()
def semantic_search(query: str, file_ids: list[str] = [], collection_names: list[str] = [], top_k: int = 10) -> str:
    """Perform a pure semantic search on the Open WebUI vector database.
    
    Returns the raw text chunks matching the query. Bypasses the internal LLM completely.
    Since Hybrid Search (BM25 + Vector) is enabled, you can use natural language OR keyword-based searches!
    """
    if not query:
        raise Exception("query is required")
    if not file_ids and not collection_names:
        raise Exception("You must provide at least one file_id or collection_name.")

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
                results.append(f"--- Chunk {i+1} (Source: {source} | Distance: {dist}) ---\n{text[:2000]}")
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
    """Search the web for information using the configured SearxNG engine.
    Returns a list of search results with titles, snippets, and URLs.
    """
    if not query:
        raise Exception("query is required")
        
    # 1. Fetch config from Open WebUI to get SearxNG URL
    config = client.get("/api/v1/retrieval/config")
    web_config = config.get("web", {})
    engine = web_config.get("WEB_SEARCH_ENGINE")
    
    if engine != "searxng":
        return json.dumps({"ok": False, "error": f"Open WebUI is configured to use '{engine}', but this tool currently only supports searxng."})
        
    searxng_url = web_config.get("SEARXNG_QUERY_URL")
    if not searxng_url:
        return json.dumps({"ok": False, "error": "SEARXNG_QUERY_URL is not configured in Open WebUI."})
        
    # Translate Docker-internal hostname and port to the host's exposed port
    searxng_url = searxng_url.replace("http://searxng:8080", "http://127.0.0.1:8888")
        
    # 2. Query SearxNG
    headers = {
        'User-Agent': 'Open WebUI (https://github.com/open-webui/open-webui) RAG Bot',
        'Accept': 'text/html',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    params = {
        'q': query,
        'format': 'json',
        'pageno': 1,
        'safesearch': '1',
        'language': 'all',
        'theme': 'simple',
        'image_proxy': 0,
    }
    
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as http:
            resp = http.get(searxng_url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return json.dumps({"ok": False, "error": f"Failed to reach SearxNG at {searxng_url}: {e}"})
        
    results = data.get("results", [])
    if not results:
        return json.dumps({
            "ok": False,
            "error": f"SearxNG returned 0 results. The engine may be down, rate-limited, or misconfigured. Raw response: {data}"
        }, ensure_ascii=False)
    
    return json.dumps(
        {
            "ok": True,
            "query": query,
            "results": [
                {
                    "title": item.get("title"),
                    "link": item.get("url"),
                    "snippet": item.get("content")
                } for item in results[:5]
            ]
        },
        ensure_ascii=False,
    )

def format_json_for_rag(data):
    """Flattens JSON into a Markdown-friendly format for better semantic chunking."""
    if isinstance(data, list):
        return "\n\n".join([format_json_for_rag(item) for item in data])
    if isinstance(data, dict):
        lines = []
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                sub_str = json.dumps(v, ensure_ascii=False)
                if len(sub_str) < 150:
                    lines.append(f"{k}: {sub_str}")
                else:
                    lines.append(f"### {k}")
                    lines.append(format_json_for_rag(v))
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)
    return str(data)

@mcp.tool()
def fetch_url(url: str) -> str:
    """Fetch the text content of a URL (web page or PDF).
    If the document is too large (over 8,000 tokens), it will automatically index the document
    and return a collection_name or file_id which you can pass to `semantic_search`.
    """
    if not url:
        raise Exception("url is required")

    # 1. First, attempt to let Open WebUI's native web loader fetch it (bypasses bot protection for sites like Reuters)
    try:
        owui_resp = client.post(
            "/api/v1/retrieval/process/url",
            params={"process": "false"},
            json={"url": url},
            timeout=120.0
        )
        if owui_resp.get("type") in ("web", "youtube") and owui_resp.get("content"):
            text = owui_resp.get("content")
            
            try:
                import tiktoken
                encoding = tiktoken.get_encoding("cl100k_base")
                token_count = len(encoding.encode(text, disallowed_special=()))
            except ImportError:
                token_count = len(text) // 4
                
            if token_count > 8000:
                # Detect JSON and force fallback so it can be formatted as Markdown
                if text.strip().startswith("{") and text.strip().endswith("}") or (text.strip().startswith("[") and text.strip().endswith("]")):
                    try:
                        json.loads(text) # Just to verify it's valid
                        raise Exception("JSON detected. Falling back to local upload to flatten it.")
                    except Exception as e:
                        if "JSON detected" in str(e): raise e
                try:
                    index_resp = client.post(
                        "/api/v1/retrieval/process/url",
                        params={"process": "true"},
                        json={"url": url},
                        timeout=120.0
                    )
                    collection_name = index_resp.get("collection_name")
                    return json.dumps({
                        "ok": True,
                        "url": url,
                        "token_count": token_count,
                        "indexed": True,
                        "collection_name": collection_name,
                        "hint": f"Document was too large ({token_count} tokens). It was automatically indexed.\nSource URL: {url}\nCall `semantic_search` with collection_names=[\"{collection_name}\"] to search it."
                    }, ensure_ascii=False)
                except Exception as index_e:
                    return json.dumps({
                        "ok": True,
                        "error": f"Document is too large to read directly ({token_count} tokens) and automatic indexing failed: {index_e}"
                    }, ensure_ascii=False)
                
            return json.dumps({
                "ok": True,
                "url": url,
                "token_count": token_count,
                "content": text[:30000]
            }, ensure_ascii=False)
    except Exception as e:
        # If it fails or it's a file, we fall back to local download
        pass

    # 2. Fallback for PDFs or if Open WebUI native fetch failed
    try:
        tmp_path, filename, size, ctype = download_to_disk(url)
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            hint = "The URL does not exist (404 Not Found). You likely guessed a broken link. Please use `search_web` to find the correct URL, or try fetching the parent directory."
        elif "401" in error_msg or "403" in error_msg or "503" in error_msg:
            hint = "The website is aggressively blocking standard HTTP bots (like Cloudflare). Please use `search_web` instead to read alternative sources."
        else:
            hint = "Failed to download the file. Try using `search_web` to read alternative sources."
            
        return json.dumps({
            "ok": True,
            "error": error_msg,
            "hint": hint
        }, ensure_ascii=False)
    try:
        mime = ctype or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        text = ""

        if mime == "application/pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(tmp_path)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            except Exception as e:
                return json.dumps({"ok": True, "error": f"Failed to extract PDF text: {e}"})
        elif mime in ["text/html", "application/xhtml+xml"]:
            try:
                from bs4 import BeautifulSoup
                with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                    soup = BeautifulSoup(f.read(), "html.parser")
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text = soup.get_text(separator="\n", strip=True)
            except Exception as e:
                return json.dumps({"ok": True, "error": f"Failed to parse HTML: {e}"})
        else:
            try:
                with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception:
                text = ""

        char_count = len(text)
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            token_count = len(encoding.encode(text, disallowed_special=()))
        except ImportError:
            token_count = char_count // 4

        # If it's a large JSON, format it to Markdown and overwrite the file before indexing
        if token_count > 8000 and (text.strip().startswith("{") and text.strip().endswith("}") or (text.strip().startswith("[") and text.strip().endswith("]"))):
            try:
                data = json.loads(text)
                text = format_json_for_rag(data)
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(text)
                if not filename.endswith(".md"):
                    filename = filename + ".md"
                mime = "text/markdown"
                # Recalculate token count for the new formatted text
                try:
                    token_count = len(encoding.encode(text, disallowed_special=()))
                except:
                    token_count = len(text) // 4
            except Exception:
                pass

        char_count = len(text)
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            token_count = len(encoding.encode(text, disallowed_special=()))
        except ImportError:
            token_count = char_count // 4

        if token_count > 8000:
            try:
                metadata = {"process": True, "source": "openwebui-mcp", "source_url": url}
                with open(tmp_path, "rb") as fh:
                    payload = client.post(
                        "/api/v1/files/",
                        files={"file": (filename, fh, mime), "metadata": (None, json.dumps(metadata), "application/json")},
                        params={"process": "true", "process_in_background": "false"},
                        timeout=600.0,
                    )
                file_id = payload.get("id")
                if file_id:
                    wait_for_file_processing(file_id)
                    return json.dumps({
                        "ok": True,
                        "url": url,
                        "token_count": token_count,
                        "indexed": True,
                        "file_id": file_id,
                        "hint": f"Document was too large ({token_count} tokens). It was automatically indexed.\nSource URL: {url}\nCall `semantic_search` with file_ids=[\"{file_id}\"] to search it."
                    }, ensure_ascii=False)
            except Exception as index_e:
                return json.dumps({
                    "ok": True,
                    "error": f"Document is too large to read directly ({token_count} tokens) and automatic indexing failed: {index_e}"
                }, ensure_ascii=False)

        return json.dumps(
            {
                "ok": True,
                "url": url,
                "filename": filename,
                "token_count": token_count,
                "content": text[:30000] # extra safety cap
            },
            ensure_ascii=False,
            default=str,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


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