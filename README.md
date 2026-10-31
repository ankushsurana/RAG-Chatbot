# RAG Chatbot

Ask questions in plain language and get short answers from a library of large PDFs. Each answer cites the PDF name and page, and the chatbot says so when the documents don't cover a question. It runs on your own computer using open-source models.

## What you need

- Windows 10 or 11 (these steps use PowerShell)
- A GPU with at least 4 GB of memory, for answers within 2 to 5 seconds
- About 10 GB of free disk
- An internet connection for the setup

## Setup

Run each command in PowerShell, one at a time.

### 1. Install Git, uv and Ollama

Skip any you already have. Close and reopen PowerShell after installing them, so the commands are found.

```powershell
winget install Git.Git
```

```powershell
winget install astral-sh.uv
```

```powershell
winget install Ollama.Ollama
```

### 2. Download the project

```powershell
git clone https://github.com/ankushsurana/RAG-Chatbot.git
```

```powershell
cd RAG-Chatbot
```

### 3. Install the dependencies

uv also downloads Python 3.11 or 3.12 if you don't have it.

```powershell
uv sync
```

### 4. Download the language model

This is about 2 GB.

```powershell
ollama pull qwen2.5:3b
```

### 5. Add the PDFs

Download the 10 demo PDFs (about 170 MB):

```powershell
uv run ragbot download-corpus
```

Or use your own: create the `data\pdfs` folder and copy your PDFs into it.

### 6. Build the search index

The first run also downloads the embedding, reranking and OCR models. The demo PDFs took about an hour here, mostly reading scanned pages, and the time depends on your computer. If it stops, run it again and it continues from the cached work.

```powershell
uv run ragbot ingest
```

### 7. Check that it worked

Every PDF should show `ready`:

```powershell
uv run ragbot documents
```

### 8. Start the chatbot

```powershell
uv run ragbot serve
```

When the terminal shows `Uvicorn running`, open **http://127.0.0.1:8000** in your browser. Keep the terminal open while you use it, and press **Ctrl+C** to stop.

Next time, you only need `cd RAG-Chatbot` and `uv run ragbot serve`.

## Share a public link (optional)

This gives anyone a free link to your chatbot through a Cloudflare tunnel. No account is needed.

Download the tunnel tool, one time only:

```powershell
New-Item -ItemType Directory -Force data\tools
```

```powershell
curl.exe -L -o data\tools\cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
```

Stop `serve` if it's running, then start the chatbot with a link:

```powershell
uv run ragbot share
```

The terminal prints a link ending in `trycloudflare.com`, which can take up to a minute to start working.

- The link works only while your computer is on and this command is running.
- The link changes each time you restart the command.
- There is no login, so don't share a link to private PDFs.

## Add more PDFs

New PDFs aren't indexed automatically.

1. Stop the chatbot with **Ctrl+C**.
2. Copy the PDFs into `data\pdfs`.
3. Run `uv run ragbot ingest`. PDFs already indexed are skipped.
4. Start the chatbot again and reload the page.

## If something goes wrong

| Problem | Fix |
|---|---|
| `uv`, `git` or `ollama` isn't recognized | Close and reopen PowerShell. If it still fails, install it again |
| The page shows no answers | Run `uv run ragbot serve` and open http://127.0.0.1:8000, not the HTML file |
| "The language model is unavailable" | Open Ollama from the Start menu, then run `ollama pull qwen2.5:3b` |
| `failed to remove file ... ragbot.exe` | Another `ragbot` is running. Close it and try again |
| Answers take over 5 seconds | Run `ollama ps`, which should show `100% GPU`. Close other programs using the GPU |

## More information

The full guide, covering the architecture, settings, API and test results, is in [docs/RAG_Chatbot_Documentation.docx](docs/RAG_Chatbot_Documentation.docx).
