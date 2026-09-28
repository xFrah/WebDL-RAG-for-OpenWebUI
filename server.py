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
    pass directly to query_documents.
    """
    if not url:
        raise Exception("url is required")

    tmp_path, filename, size = download_to_disk(url)
    try:
        metadata = {
            "process": True,
            "source": "openwebui-mcp",
            "source_url": url,
        }
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
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
                "hint": "Ready: call query_documents with file_ids=[\"" + file_id + "\"] to ask questions.",
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
    can pass to query_documents.
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
            "hint": "Ready: call query_documents with collection_names=[\"" + collection_name + "\"] to ask questions.",
        },
        ensure_ascii=False,
        default=str,
    )


@mcp.tool()
def query_documents(question: str, file_ids: list[str] = [], collection_names: list[str] = [], model: Optional[str] = None) -> str:
    """Ask a question, answered by RAG over the provided files or web collections.

    Runs POST /api/chat/completions so the model grounds its answer in
    the provided files. You must provide at least one file_id or collection_name.
    """
    if not question:
        raise Exception("question is required")
    if not file_ids and not collection_names:
        raise Exception("You must provide at least one file_id or collection_name.")
        
    model = model or DEFAULT_MODEL
    if not model:
        raise Exception("No model specified and OPENWEBUI_DEFAULT_MODEL is not set. Pass model=... or set the env var.")
        
    files = []
    for fid in file_ids:
        files.append({"type": "file", "id": fid})
    for cid in collection_names:
        files.append({"type": "collection", "id": cid})

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "files": files,
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
        {"ok": bool(answer), "answer": answer, "usage": usage, "model": model},
        ensure_ascii=False,
        default=str,
    )


@mcp.tool()
def estimate_url_size(url: str) -> str:
    """CRITICAL RULE: You MUST call this tool FIRST before calling `fetch_url` or reading any webpage/document directly!
    
    This tool downloads a URL and estimates its size in characters and tokens.
    Use this to decide if a document is small enough to be read directly using 
    your standard `fetch_url` tool (under ~8,000 tokens), or if it's too large 
    and MUST be processed by Open WebUI via `download_file` or `process_web_url`.
    """
    if not url:
        raise Exception("url is required")

    tmp_path, filename, size = download_to_disk(url)
    try:
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
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
                return json.dumps({"ok": False, "error": f"Failed to extract PDF text: {e}"})
        elif mime in ["text/html", "application/xhtml+xml"]:
            try:
                from bs4 import BeautifulSoup
                with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                    soup = BeautifulSoup(f.read(), "html.parser")
                    text = soup.get_text(separator="\n", strip=True)
            except Exception as e:
                return json.dumps({"ok": False, "error": f"Failed to parse HTML: {e}"})
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

        return json.dumps(
            {
                "ok": True,
                "url": url,
                "filename": filename,
                "file_size_bytes": size,
                "character_count": char_count,
                "estimated_tokens": token_count,
                "hint": "If estimated_tokens is under ~8,000, you can safely read it directly. If it is very large, use download_file to add it to RAG instead.",
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