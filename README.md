# RAG Chatbot

Ask questions in plain language and get short answers from a library of large PDFs. Each answer cites the PDF name and page, and the chatbot says so when the documents don't cover a question. It runs locally on open-source models.

You need Python 3.11, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) and a GPU with at least 4 GB of memory.

## Commands

Install the dependencies:

```bash
uv sync
```

Download the language model:

```bash
ollama pull qwen2.5:3b
```

Download the 10 demo PDFs (or put your own in `data/pdfs`):

```bash
uv run ragbot download-corpus
```

Index the PDFs (the first run can take an hour or more):

```bash
uv run ragbot ingest
```

Start the chatbot, then open http://127.0.0.1:8000:

```bash
uv run ragbot serve
```

Or start it with a free public link through a Cloudflare tunnel. This needs [cloudflared](https://github.com/cloudflare/cloudflared/releases) on your PATH or at `data/tools/cloudflared.exe`, and the link works only while this computer is running the command:

```bash
uv run ragbot share
```

The full guide is in [docs/RAG_Chatbot_Documentation.docx](docs/RAG_Chatbot_Documentation.docx).
