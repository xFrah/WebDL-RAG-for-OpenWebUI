# openwebui-kb-mcp

An MCP tool server that integrates AI agents with [Open WebUI](https://openwebui.com)'s powerful RAG and Web Search pipelines.

By routing heavy tasks through Open WebUI, agents can index large PDFs, search the web, and semantically query knowledge **without polluting their own context window**.

### Tools Provided

| Tool | What it does | Open WebUI API |
| --- | --- | --- |
| `fetch_url` | Fetches the text content of a URL (web page or PDF) directly. If too large, it automatically indexes the document and returns a `file_id`/`collection_name`. | `POST /api/v1/retrieval/process/url` |
| `query_fetched_file` | Performs vector-based RAG search over specific `file_ids` using Open WebUI's retrieval engine. | `POST /api/v1/retrieval/query/doc` |
| `grep_fetched_file` | Performs an exact-text regular expression search against the parsed text of an uploaded file. Perfect for finding precise code snippets. | `GET /api/v1/files/{id}/data/content` |
| `search_web` | Searches the internet using Open WebUI's configured search backend (e.g. SearxNG, OpenSerp). | `POST /api/v1/retrieval/process/web/search` |

## Setup

### 1. Configure

```bash
cd openwebui-kb-mcp
cp .env.example .env
# edit .env: set OPENWEBUI_URL, OPENWEBUI_API_KEY, OPENWEBUI_DEFAULT_MODEL