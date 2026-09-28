# openwebui-kb-mcp

An MCP tool server that keeps big documents **out of the LLM context window** by
routing them through [Open WebUI](https://openwebui.com)'s RAG pipeline.

When the model sees a PDF / large document URL, it does **not** fetch the body
into the conversation. Instead it calls these tools:

| Tool | What it does | Open WebUI endpoint |
| --- | --- | --- |
| `create_knowledge_base` | Create a KB (vector collection) | `POST /api/v1/knowledge/create` |
| `search_knowledge_bases` | Find an existing KB's id | `GET /api/v1/knowledge/search` |
| `download_and_index` | Download URL → upload with `process=true` → **wait for embedding** → add to KB (triggers RAG) | `POST /api/v1/files/`, `GET /api/v1/files/{id}/process/status`, `POST /api/v1/knowledge/{id}/file/add` |
| `process_web_url` | Server-side fetch+parse of an HTML page into a KB (no download at all) | `POST /api/v1/retrieval/process/web` |
| `list_knowledge_files` | Check what's in a KB and whether processing finished | `GET /api/v1/knowledge/{id}/files` |
| `query_knowledge_base` | Ask a question, grounded by RAG over the KB | `POST /api/chat/completions` with `files=[{type:"collection",id:…}]` |
| `delete_knowledge_base` | Delete a KB + collection | `DELETE /api/v1/knowledge/{id}/delete` |

The important sequencing (and the reason naive fetch-then-add fails with
`400: content provided is empty`) is handled for you: **upload → poll until
`status == "completed"` → then `file/add`**.

## Setup

### 1. Configure

```bash
cd openwebui-kb-mcp
cp .env.example .env
# edit .env: set OPENWEBUI_URL, OPENWEBUI_API_KEY, OPENWEBUI_DEFAULT_MODEL