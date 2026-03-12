# Local Document Chat — Vector RAG System

A fully local, privacy-first **Retrieval-Augmented Generation (RAG)** chatbot that lets you have natural conversations with your documents. Powered by [Ollama](https://ollama.com/) for LLM inference and embeddings, [ChromaDB](https://www.trychroma.com/) for vector storage, and a rich terminal UI.

> **No data leaves your machine.** Everything runs locally — models, embeddings, and vector search.

---

## 🎬 Demo

<video src="assets/video.mp4" width="100%" controls autoplay muted loop></video>

---

## ✨ Features

| Feature | Description |
|---|---|
| **Multi-format ingestion** | PDF, TXT, Markdown, HTML, DOCX, CSV |
| **Live URL scraping** | `/add-url` fetches & indexes any webpage mid-session |
| **Streaming responses** | Token-by-token output with Ctrl+C interrupt |
| **Text-to-speech** | macOS `say` integration with toggle (`/voice`) |
| **Chat persistence** | Save/load conversations as `.md` + `.json` |
| **PDF export** | Export any conversation to a formatted PDF |
| **Auto-save** | Silently saves every N turns so you never lose work |
| **Source transparency** | `/why` shows the exact chunks used for each answer |
| **Multi-model support** | Switch between models mid-session (`/change-model`) |
| **Hot-swap documents** | Add or change docs without restarting (`/add-doc`, `/change-docs`) |

---

## 📁 Project Structure

```
rag-system/
├── rag/                    # Core Python package
│   ├── __init__.py         # Package marker
│   ├── config.py           # All configuration constants
│   ├── console.py          # Rich console + custom theme
│   ├── speech.py           # TTS queue (multithreaded daemon)
│   ├── loaders.py          # File readers (PDF, DOCX, CSV…)
│   ├── chunking.py         # Text chunker with overlap
│   ├── vectordb.py         # ChromaDB + Ollama embeddings
│   ├── scraper.py          # URL fetcher + indexer
│   ├── export.py           # PDF conversation export
│   ├── ui.py               # Model/doc selectors + help menu
│   ├── query.py            # Core RAG query runner
│   └── chat.py             # Interactive chat loop + commands
├── main.py                 # Entry point
├── rag.sh                  # One-click launcher (bash)
├── docs/                   # Drop your documents here
├── .gitignore
└── README.md
```

---

## 🛠️ Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** installed and running
- **macOS** (for TTS via `say` — the rest works cross-platform)

### Pull the required models

```bash
# Embedding model (required)
ollama pull nomic-embed-text

# At least one chat model
ollama pull sadiq-bd/llama3.2-1b-uncensored      # fast
ollama pull IHA089/drana-infinity-0.5b:0.5b       # very fast
ollama pull sadiq-bd/llama3.2-3b-uncensored       # smarter
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/rag-system.git
cd rag-system
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your documents

Drop any `.pdf`, `.txt`, `.md`, `.html`, `.docx`, or `.csv` files into the `docs/` folder.

### 5. Run

```bash
# Option A — direct
python3 main.py

# Option B — shell script
bash rag.sh
```

---

## 💬 Usage

On launch you'll be prompted to:
1. **Choose a model** (1–3)
2. **Select documents** to load (by number, or `all`)

Then just type your questions. The AI will use your documents as context.

### Available Commands

| Command | What it does |
|---|---|
| `/help` | Show the full command reference |
| `/status` | Dashboard — model, docs, chunks, voice, uptime |
| `/history` | Print the full conversation |
| `/search-history <kw>` | Search & highlight turns by keyword |
| `/summarize` | Ask the model to summarize the conversation |
| `/clear` | Wipe history (keep model & docs) |
| `/retry` | Regenerate the last AI response |
| `/why` | Show the source chunks behind the last answer |
| `/save-chat [name]` | Save conversation to `chats/` (.md + .json) |
| `/load-chat` | Resume a previously saved chat |
| `/export-pdf [name]` | Export conversation as formatted PDF |
| `/list-docs` | Show loaded docs and chunk counts |
| `/add-doc` | Add a new doc from `docs/` mid-session |
| `/add-url <url>` | Scrape & index a webpage on the fly |
| `/change-docs` | Swap all docs and rebuild the vector DB |
| `/change-model` | Switch to a different LLM |
| `/voice` | Toggle text-to-speech on/off |
| `exit` | Quit |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
