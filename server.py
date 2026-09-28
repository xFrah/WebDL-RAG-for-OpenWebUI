#!/usr/bin/env python3
"""
openwebui-kb-mcp — an MCP tool server that routes big-document work through
Open WebUI's RAG pipeline instead of the LLM context window.

Workflow it gives the LLM:
  1. create_knowledge_base   -> POST /api/v1/knowledge/create
  2. download_and_index      -> download URL to disk (streamed, size-capped),
                                POST /api/v1/files/  (process=true, metadata.knowledge_id)
                                poll GET  /api/v1/files/{id}/process/status
                                POST /api/v1/knowledge/{id}/file/add
  3. process_web_url         -> POST /api/v1/retrieval/process/web  (no download)
  4. query_knowledge_base    -> POST /api/chat/completions with
                                files=[{type:"collection", id:<kb id>}]

Config (environment variables, or .env file next to this script):
  OPENWEBUI_URL           base URL, e.g. http://localhost:3000
  OPENWEBUI_API_KEY       sk-... API key (Settings > Account)
  OPENWEBUI_DEFAULT_MODEL model id used by query_knowledge_base (optional)
  KB_PROCESS_TIMEOUT      seconds to wait for embedding (default 600)
  KB_MAX_FILE_BYTES       download size cap (default 2 GiB)

Run (stdio — for MCP clients such as Claude Desktop, or Open WebUI's mcpo proxy):
  uv run server.py

Run (Streamable HTTP — the transport Open WebUI connects to natively, v0.6.31+):
  MCP_TRANSPORT=streamable-http MCP_HTTP_PORT=8766 uv run server.py
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
from mcp.server.mcpserver import MCPServer

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


def download_to_disk(url: str) -> tuple[str, str, int]:
    """Stream a URL to a temp file. Returns (path, filename, size_bytes)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise Exception(f"Only http/https URLs are supported, got: {url}")
    filename = extract_filename(url)
    fd, tmp_path = tempfile.mkstemp(prefix="owui-kb-", suffix=os.path.splitext(filename)[1] or ".bin")
    os.close(fd)
    headers = {"User-Agent": "openwebui-kb-mcp/1.0 (+https://github.com/open-webui/open-webui)"}
    size = 0
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=600.0)) as http:
        with http.stream("GET", url, headers=headers) as resp:
            if resp.status_code >= 400:
                raise Exception(f"Download failed with HTTP {resp.status_code} for {url}")
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
    return tmp_path, filename, size


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
                f"Timed out after {timeout}s waiting for file {file_id} to finish processing. "
                f"You can check later with list_knowledge_files or the Open WebUI UI."
            )
        time.sleep(POLL_INTERVAL)


def add_file_to_knowledge(knowledge_id: str, file_id: str) -> dict[str, Any]:
    """POST /api/v1/knowledge/{id}/file/add — runs process_file against the
    KB collection, which is what actually triggers the RAG embedding into the
    knowledge base collection."""
    return client.post(
        f"/api/v1/knowledge/{knowledge_id}/file/add",
        json={"file_id": file_id},
        timeout=max(300.0, PROCESS_TIMEOUT),
    )


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

mcp = MCPServer("openwebui-kb")


@mcp.tool()
def create_knowledge_base(name: str, description: str = "") -> str:
    """Create a new Open WebUI knowledge base (vector collection).

    Call this when you need a place to store and later RAG-query documents
    (PDFs, large text files, ...). Returns a JSON object with the KB's id,
    which you then pass to download_and_index / process_web_url /
    query_knowledge_base. If a KB with the same name already exists, the
    response tells you so — use search_knowledge_bases to find its id.
    """
    try:
        kb = client.post(
            "/api/v1/knowledge/create",
            json={"name": name, "description": description, "access_grants": []},
        )
        return json.dumps(
            {
                "ok": True,
                "knowledge_base": {
                    "id": kb.get("id"),
                    "name": kb.get("name"),
                    "description": kb.get("description"),
                },
                "hint": "Use this id with download_and_index, process_web_url, or query_knowledge_base.",
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        if "exists" in str(exc).lower() or "already" in str(exc).lower():
            return json.dumps(
                {"ok": False, "error": "A knowledge base with this name already exists.",
                 "hint": "Call search_knowledge_bases(query=<name>) to get its id instead of creating a duplicate."},
                ensure_ascii=False,
            )
        raise


@mcp.tool()
def search_knowledge_bases(query: str = "", page: int = 1) -> str:
    """Search your Open WebUI knowledge bases by name/description.

    Use this to find the id of an existing knowledge base before querying or
    adding documents to it, so you don't create duplicates.
    """
    data = client.get("/api/v1/knowledge/search", params={"query": query or None, "page": page})
    items = [
        {"id": kb.get("id"), "name": kb.get("name"), "description": kb.get("description", "")}
        for kb in (data.get("items") or [])
    ]
    return json.dumps(
        {"ok": True, "total": data.get("total", len(items)), "knowledge_bases": items},
        ensure_ascii=False,
        default=str,
    )


@mcp.tool()
def download_and_index(url: str, knowledge_id: Optional[str] = None,
                       knowledge_name: Optional[str] = None,
                       wait: bool = True) -> str:
    """Download a file from a URL and run it through Open WebUI's RAG pipeline.

    Use this instead of fetching big or binary documents (PDFs, DOCX, large
    reports, code archives) directly — those would overflow the context
    window. The file is streamed to disk (never into the conversation),
    uploaded to Open WebUI with process=true, the server waits until
    embedding completes, and the file is then added to the target knowledge
    base, which triggers the RAG pipeline for the KB collection.

    Provide either knowledge_id (an existing KB) or knowledge_name (creates a
    new KB). Set wait=false to skip blocking on embedding (you can still
    query later once processing finishes).
    """
    if not url:
        raise Exception("url is required")
    kb_id = knowledge_id
    kb_name_used = None
    if not kb_id:
        kb_name_used = knowledge_name or extract_filename(url)
        kb = client.post(
            "/api/v1/knowledge/create",
            json={"name": kb_name_used, "description": f"Auto-created by openwebui-kb-mcp from {url}",
                  "access_grants": []},
        )
        kb_id = kb.get("id")
        if not kb_id:
            raise Exception(f"Knowledge base creation returned no id: {kb}")

    tmp_path, filename, size = download_to_disk(url)
    try:
        metadata = {
            "knowledge_id": kb_id,
            "process": True,
            "source": "openwebui-kb-mcp",
            "source_url": url,
        }
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        with open(tmp_path, "rb") as fh:
            payload = client.post(
                "/api/v1/files/",
                files={"file": (filename, fh, mime), "metadata": (None, json.dumps(metadata), "application/json")},
                params={"process": "true", "process_in_background": "true"},
                timeout=600.0,
            )
        file_id = payload.get("id")
        if not file_id:
            raise Exception(f"File upload returned no id: {payload}")

        processing = {"file_id": file_id, "status": "pending"}
        if wait:
            processing = wait_for_file_processing(file_id)

        added = False
        error: Optional[str] = None
        if wait and processing.get("status") == "completed":
            try:
                add_file_to_knowledge(kb_id, file_id)
                added = True
            except Exception as exc:
                error = str(exc)

        return json.dumps(
            {
                "ok": True,
                "filename": filename,
                "size_bytes": size,
                "file_id": file_id,
                "knowledge_id": kb_id,
                "knowledge_name": kb_name_used,
                "processing": processing,
                "added_to_knowledge_base": added,
                "error": error,
                "hint": (
                    "Ready: call query_knowledge_base with knowledge_id to ask questions about this document."
                    if added
                    else "File is still processing in the background. Re-call this tool's follow-up by calling "
                    "list_knowledge_files, or query once a minute or two."
                ),
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
def process_web_url(url: str, knowledge_id: Optional[str] = None,
                    knowledge_name: Optional[str] = None,
                    overwrite: bool = False) -> str:
    """Ingest a web page directly into an Open WebUI knowledge base.

    Open WebUI fetches and parses the URL server-side (POST /api/v1/retrieval/
    process/web) — nothing is downloaded into the conversation. Best for HTML
    documentation pages. For PDFs and other binary files use
    download_and_index instead. Provide knowledge_id, or knowledge_name to
    create a new KB. overwrite=true replaces the collection's existing
    vectors with just this URL's content.
    """
    if not url:
        raise Exception("url is required")
    kb_id = knowledge_id
    kb_name_used = None
    if not kb_id:
        kb_name_used = knowledge_name or f"web:{urlparse(url).netloc}/{urlparse(url).path}"
        kb = client.post(
            "/api/v1/knowledge/create",
            json={"name": kb_name_used, "description": f"Auto-created by openwebui-kb-mcp from {url}",
                  "access_grants": []},
        )
        kb_id = kb.get("id")
        if not kb_id:
            raise Exception(f"Knowledge base creation returned no id: {kb}")

    collection_name = f"web:{uuid.uuid4().hex[:8]}:{kb_id}"
    result = client.post(
        "/api/v1/retrieval/process/web",
        params={"process": "true", "overwrite": "true" if overwrite else "false"},
        json={"url": url, "collection_name": collection_name},
        timeout=300.0,
    )
    return json.dumps(
        {
            "ok": True,
            "url": url,
            "knowledge_id": kb_id,
            "knowledge_name": kb_name_used,
            "collection_name": collection_name,
            "upstream_result": {k: v for k, v in (result or {}).items() if k in ("count", "documents", "success", "error")},
            "hint": (
                "If upstream_result has no chunk count, the page may still be processing or may be a binary file. "
                "Check with list_knowledge_files; for PDFs re-run with download_and_index."
            ),
        },
        ensure_ascii=False,
        default=str,
    )


@mcp.tool()
def list_knowledge_files(knowledge_id: str, page: int = 1) -> str:
    """List files currently in a knowledge base, with processing status.

    Useful to verify a file finished being added after download_and_index /
    process_web_url, and to see file ids.
    """
    data = client.get(f"/api/v1/knowledge/{knowledge_id}/files", params={"page": page})
    items = []
    for f in data.get("items") or []:
        items.append(
            {
                "id": f.get("id"),
                "filename": f.get("filename"),
                "size": f.get("size"),
                "content_length": len((f.get("data") or {}).get("content") or "") if isinstance(f.get("data"), dict) else None,
            }
        )
    return json.dumps(
        {"ok": True, "knowledge_id": knowledge_id, "total": data.get("total", len(items)), "files": items},
        ensure_ascii=False,
        default=str,
    )


@mcp.tool()
def query_knowledge_base(question: str, knowledge_id: str, model: Optional[str] = None) -> str:
    """Ask a question, answered by RAG over an Open WebUI knowledge base.

    Runs POST /api/chat/completions with files=[{type:"collection",
    id:knowledge_id}], so the model grounds its answer in the KB's embedded
    chunks — the whole point of using this MCP instead of fetching big files
    into the context window. Pass model to override OPENWEBUI_DEFAULT_MODEL.
    Returns the assistant's answer text.
    """
    if not question:
        raise Exception("question is required")
    if not knowledge_id:
        raise Exception("knowledge_id is required (see search_knowledge_bases / create_knowledge_base)")
    model = model or DEFAULT_MODEL
    if not model:
        raise Exception("No model specified and OPENWEBUI_DEFAULT_MODEL is not set. Pass model=... or set the env var.")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "files": [{"type": "collection", "id": knowledge_id}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    chunks: list[str] = []
    usage: Optional[dict[str, Any]] = None
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=1200.0)) as http:
        with http.stream("POST", f"{OPENWEBUI_URL}/api/chat/completions", headers=base_headers(), json=payload) as resp:
            if resp.status_code >= 400:
                raise _api_error(resp)
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "usage" in evt and evt["usage"]:
                    usage = evt["usage"]
                choices = evt.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    text = _content_to_text(delta.get("content"))
                    if text:
                        chunks.append(text)
    answer = "".join(chunks).strip()
    return json.dumps(
        {"ok": bool(answer), "answer": answer, "usage": usage, "knowledge_id": knowledge_id, "model": model},
        ensure_ascii=False,
        default=str,
    )


@mcp.tool()
def delete_knowledge_base(knowledge_id: str) -> str:
    """Delete a knowledge base and its vector collection (irreversible)."""
    data = client.request("DELETE", f"/api/v1/knowledge/{knowledge_id}/delete").json()
    return json.dumps({"ok": bool(data), "knowledge_id": knowledge_id}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Entrypoint — stdio (default) or streamable-http (Open WebUI native MCP)
# ---------------------------------------------------------------------------

def main() -> None:
    transport = os.environ.get("MCP_TRANSPORT", "stdio").lower()
    if transport in ("streamable-http", "http", "streamable_http", "sse"):
        host = os.environ.get("MCP_HTTP_HOST", "0.0.0.0")
        port = int(os.environ.get("MCP_HTTP_PORT", "8766"))
        try:
            # In MCP SDK 2.x, host and port are passed to run() instead of mcp.settings
            mcp.run(transport="sse", host=host, port=port)
        except (ImportError, ValueError, TypeError) as exc:
            raise SystemExit(
                f"HTTP/SSE transport unavailable ({exc}). Install the latest "
                f"mcp package, or run with MCP_TRANSPORT=stdio and front the server "
                f"with the mcpo proxy for Open WebUI."
            )
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()