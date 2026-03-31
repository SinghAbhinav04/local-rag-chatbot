# 🌌 RAG Agent: The Multimodal AI Workspace

A powerful, hybrid local-and-cloud Retrieval-Augmented Generation (RAG) assistant featuring a stunning, minimalist web interface, deep CLI integration, an expandable skills system, and native multimodal support (PDFs & Images) powered by advanced LLMs like Ollama, Google Gemini, Anthropic, and OpenAI.

Inspired by robust autonomous agents, RAG Agent functions as your intelligent co-pilot—capable of reading multi-page PDFs, searching the web, executing secure shell commands, scheduling tasks, maintaining persistent memory, and seamlessly chatting with your local document corpus.

---

## ✨ Core Highlights

- **Hybrid LLM Support**: Switch instantly between local models running on Ollama and premium cloud APIs (Gemini, Claude, GPT-4o, DeepSeek, Groq, etc.).
- **Multimodal Intelligence**: Send 1000-page PDFs, images, and massive documents to Gemini directly via the chat interface for deep structural analysis.
- **Premium Monochromatic Web UI**: A sleek, distraction-free, zero-emoji black-and-white dashboard.
- **Expandable Skills System**: Ships with 7 powerful bundled tools (Terminal/Tmux, Browser Automation, Hardware Interaction, GitHub CI/CD, and more).
- **Persistent AI Memory**: Long-term associative memory and daily chronological notes ensure the agent never forgets context across sessions.
- **Background Automation**: Write, edit, append files, run cron jobs, and perform complex web scraping automatically.

---

## 💻 CLI Usage (`rag` command)

The system ships with a high-performance CLI wrapper for rapid interactions from your terminal.

```bash
# Start the Web UI & Agent Server (Port 18800)
rag serve

# Interactive Chat (CLI Mode)
rag chat

# Add a document or directory to the active RAG Context
rag add ./docs/architecture.pdf
rag add https://example.com/article

# Search the knowledge base directly
rag search "How is authentication handled?"

# Remove a document from the RAG Context
rag remove architecture.pdf

# Clear out the current context
rag clear

# Spawn a background autonomous subagent
rag subagent "Research quantum computing and save notes to quantum.md"
```

---

## 🌐 Web Dashboard Capabilities

Launch the dashboard via `rag serve` and navigate to `http://localhost:18800`. The minimalist workspace provides robust control over the entire agent state:

### 1. 💬 Chat & Agent Interface
- **Chat Interface**: Stream text natively from your chosen LLM. Attach heavy files (PDFs, Source Code, Images) directly via the paperclip icon.
- **RAG Integration**: Any query automatically embeds and retrieves the most relevant semantic chunks from your indexed documents.
- **Agentic Actions**: The LLM will autonomously decide when to invoke built-in capabilities (like fetching the weather, reading a file, or running a shell script) directly inside the conversation stream.

### 2. 🧠 Knowledge Base (Documents)
- View a real-time list of all parsed documents and their chunk counts.
- Add URLs directly to the index from the UI.
- See total corpus stats for your active RAG environment.

### 3. ⚙️ Cloud Models
Configure external API keys to plug in the world's most capable foundation models:
- **Major Providers**: OpenAI, Google Gemini (Native API), Anthropic, Mistral.
- **Open-Source Hosted**: Groq, Together AI, Fireworks AI, DeepSeek, OpenRouter.
- Changes update in real-time, instantly porting your agent's brain to a new architecture.

### 4. 🧰 Tools & Integrations
A dedicated UI for one-click knowledge enhancement. *(Inputs sync continuously to the system backend to save your preferences)*.
- **Wikipedia**: Ingest comprehensive wiki articles into your context.
- **Weather**: Feed the agent live topological meteorological data.
- **News Headlines**: Grab top breaking news articles via NewsAPI.
- **Deep Web Scrape**: Pluck text data out of any arbitrary URL.

### 5. 📂 Memory & Cron
- **Persistent Memory**: Manage what the AI remembers long-term or read its daily journal entries.
- **Cron Jobs**: See and manage scheduled threads. Ask the LLM to *"remind me to check the server logs in 10 minutes"*, and manage that job visually here.

### 6. 🌠 Skills Registry
Out-of-the-box instructions telling the AI exactly how to use complex CLI toolchains:
- `github`: Manage issues and inspect Actions.
- `agent-browser`: Control a headless browser.
- `summarize`: Hook into complex summarization shell pipelines.
- `tmux`: Remotely control and detach terminal sessions.
- `skill-creator`: A meta-skill teaching the agent how to code new skills!

---

## 🛠 Built-in AI Tools (Auto-invoked by LLMs)

The exact capabilities the LLM can use on your behalf during a conversation:

1. **`web_search`**: DuckDuckGo search to ground the AI in reality.
2. **`exec`**: Safely execute shell commands (with a heavily guarded sandbox blocking `rm`, `sudo`, `docker`, etc.).
3. **`read_file`** / **`write_file`** / **`edit_file`** / **`append_file`**: Full filesystem CRUD.
4. **`web_fetch`**: Read clean text out of arbitrary URLs.
5. **`memory_write`**: Write notes to its permanent internal database.
6. **`cron`**: Schedule threaded reminders and loops.
7. **`find_skills`**: Search for new `.md` skills inside your workspace.

---

## 🏗 System Architecture

- **Backend**: Python 3.10+ Native `http.server` (No bloated frameworks).
- **Frontend**: Vanilla HTML5/CSS3/JS utilizing ultra-lightweight CSS layout grids and responsive modular coding.
- **RAG Core**: TF-IDF Document Vectorization built to run natively without heavy vector DB dependencies paired with precise Cosine Similarity metrics.
- **Persistence**: Flat-file JSON architectures (`cloud_config.json`, memory collections) guarantee zero-database setups and clean transportability.
