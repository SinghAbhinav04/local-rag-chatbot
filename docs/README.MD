# 🧠 minimal.ai Backend

Backend API for **minimal.ai** — an AI-powered intelligent task execution system with multi-provider LLM orchestration, OTP email verification, and voice biometric authentication.

## ✨ Features

- **Secure Authentication**
  - Email/Password with **6-digit OTP email verification**
  - OAuth (Google, GitHub)
  - Voice biometric verification (Android)
  - **Login security alerts** — IP, device, and location emailed on every login

- **Multi-Provider LLM Orchestration**
  - AI-powered task routing across 3 models:
    - ⚡ **GPT-OSS-120B** (Groq) → Quick Q&A, simple tasks
    - 🧠 **Gemini 2.5 Flash** → Complex reasoning, deep analysis
    - 👁️ **Gemini 3 Flash Preview** → Image understanding
  - Automatic complexity classification
  - Cost and latency tracking per request

- **Session-Based Task Management**
  - Create sessions, submit requests, track execution steps
  - Full observability: strategy, model, tokens, latency per request

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Framework** | FastAPI (Python 3.11+) |
| **Database** | MongoDB Atlas (Motor async driver) |
| **Auth** | JWT + OTP (SHA-256 hashed) + OAuth2 |
| **LLM** | Google Gemini + Groq (OpenAI-compatible) |
| **Email** | Gmail SMTP (TLS) |
| **Voice** | Resemblyzer (voice embeddings) |
| **Deployment** | Docker / Railway |

## 📁 Project Structure

```
backend/
├── app/
│   ├── main.py                  # FastAPI app + lifespan
│   ├── config.py                # Pydantic settings
│   ├── api/
│   │   ├── auth.py              # JWT + password utilities
│   │   ├── routes_auth.py       # Auth: signup, login, OTP, OAuth
│   │   ├── routes_tasks.py      # Sessions + task execution
│   │   └── routes_voice.py      # Voice biometric endpoints
│   ├── db/
│   │   ├── database.py          # MongoDB connection (Motor)
│   │   ├── models.py            # Pydantic request/response models
│   │   └── queries.py           # MongoDB operations
│   ├── orchestrator/
│   │   ├── orchestrator.py      # Multi-provider LLM engine
│   │   ├── gemini_provider.py   # Gemini 2.5 Flash + 3 Flash
│   │   ├── groq_provider.py     # GPT-OSS-120B via Groq
│   │   └── strategies.py        # AI-powered model routing
│   ├── services/
│   │   └── email_service.py     # OTP emails + login alerts
│   └── voice/
│       └── voice_processor.py   # Voice embedding + matching
├── requirements.txt
├── Dockerfile
└── .env
```

## 🚀 Setup

### Prerequisites
- Python 3.9+
- MongoDB Atlas account (free tier works)
- Google API Key (Gemini)
- Groq API Key
- Gmail account with App Password

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Run the server
uvicorn app.main:app --reload
```

Server starts at `http://localhost:8000`

### API Docs
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 📡 API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/signup` | Register → sends OTP email |
| POST | `/auth/login` | Login → sends OTP email |
| POST | `/auth/verify-otp` | Verify OTP → returns JWT |
| POST | `/auth/resend-otp` | Resend OTP code |
| POST | `/auth/refresh` | Refresh access token |
| POST | `/auth/logout` | Revoke refresh token |
| GET | `/auth/me` | Get current user |
| GET | `/auth/oauth/google` | Google OAuth flow |
| GET | `/auth/oauth/github` | GitHub OAuth flow |

### Voice Verification
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/voice/enroll` | Enroll voice sample |
| POST | `/voice/verify` | Verify voice → login |
| GET | `/voice/samples` | List enrolled samples |
| DELETE | `/voice/delete` | Delete all samples |

### Task Execution
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/sessions` | Create new session |
| GET | `/sessions` | List user sessions |
| GET | `/sessions/{id}` | Get session details |
| POST | `/sessions/{id}/requests` | Submit task request |
| GET | `/requests/{id}` | Get request details |
| GET | `/requests/{id}/steps` | Get execution steps |
| GET | `/requests/{id}/response` | Get final response |

## 🤖 LLM Routing

The orchestrator uses AI meta-decision to route each request to the optimal model:

| Model | Provider | Use Case | Speed |
|-------|----------|----------|-------|
| `gpt-oss-120b` | Groq | Quick facts, definitions, simple Q&A | ⚡ Fastest |
| `gemini-2.5-flash` | Google | Analysis, planning, complex reasoning | 🧠 Balanced |
| `gemini-3-flash-preview` | Google | Image understanding, vision tasks | 👁️ Multimodal |

## 🔐 Auth Flow (OTP)

```
1. POST /auth/signup  →  { requires_otp: true }
2. User receives 6-digit code via email
3. POST /auth/verify-otp  →  { access_token, refresh_token }
4. User receives login security alert email (IP, device, location)
```

## ⚙️ Environment Variables

| Variable | Description |
|----------|-------------|
| `MONGODB_URI` | MongoDB Atlas connection string |
| `MONGODB_DB_NAME` | Database name (default: `minimal_ai`) |
| `JWT_SECRET_KEY` | Secret for signing JWT tokens |
| `GOOGLE_API_KEY` | Google Gemini API key |
| `GROQ_API_KEY` | Groq API key |
| `SMTP_HOST` | SMTP server (default: `smtp.gmail.com`) |
| `SMTP_PORT` | SMTP port (default: `587`) |
| `SMTP_EMAIL` | Sender email address |
| `SMTP_PASSWORD` | Gmail App Password |
| `OTP_EXPIRE_MINUTES` | OTP validity (default: `5`) |
| `CORS_ORIGINS` | Allowed origins (comma-separated) |

## 🐳 Docker

```bash
docker build -t minimal-ai-backend .
docker run -p 8000:8000 --env-file .env minimal-ai-backend
```

## 📄 License

MIT
