# minimal.ai Backend Deployment & Integration Guide

## 🔌 Backend-Frontend Integration

### Architecture Overview

```
Android App (Kotlin)
       ↓ HTTPS/HTTP
FastAPI Backend (Python)
       ↓
PostgreSQL Database
       ↓
Google Gemini AI
```

---

## 💻 Local Development Setup

### 1. Start Backend on Mac

```bash
# Navigate to backend
cd /Users/abhinavsingh/Documents/DesktopData/minimal-ai/backend

# Activate virtual environment
source env/bin/activate

# Start server (accessible on network)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**What this does:**
- `--host 0.0.0.0` → Makes server accessible from other devices on same network
- `--port 8000` → Runs on port 8000
- `--reload` → Auto-reload on code changes

### 2. Find Your Mac's Local IP

```bash
# Get local IP address
ifconfig | grep "inet " | grep -v 127.0.0.1

# Example output:
# inet 192.168.1.45 netmask 0xffffff00 broadcast 192.168.1.255
#      ^^^^^^^^^^^^
#      This is your IP!
```

### 3. Configure Android App

**For Android Emulator:**
```kotlin
// app/src/main/java/com/minimal/ai/data/api/ApiConfig.kt
object ApiConfig {
    const val BASE_URL = "http://10.0.2.2:8000/"  
    // 10.0.2.2 is special address that points to host machine
}
```

**For Real Android Device:**
```kotlin
object ApiConfig {
    const val BASE_URL = "http://192.168.1.45:8000/"  
    // Use your Mac's IP from step 2
    // IMPORTANT: Phone must be on same WiFi as Mac!
}
```

### 4. Test Connection

**From Android App:**
```kotlin
// In your ViewModel or Repository
viewModelScope.launch {
    try {
        val response = apiService.healthCheck()
        Log.d("API", "Backend is reachable: $response")
    } catch (e: Exception) {
        Log.e("API", "Cannot reach backend: ${e.message}")
    }
}
```

**Or use browser on phone:**
- Open Chrome on phone
- Go to `http://192.168.1.45:8000/docs`
- You should see FastAPI Swagger docs

---

## ☁️ Cloud Deployment (Production)

### Option 1: Railway.app ⭐ **RECOMMENDED**

**Why Railway:**
- One-click deployment
- Auto HTTPS
- Free $5 credit monthly
- Easy to use
- GitHub integration

**Setup (5 minutes):**

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Navigate to backend
cd /Users/abhinavsingh/Documents/DesktopData/minimal-ai/backend

# 4. Initialize project
railway init
# Name: minimal-ai-backend

# 5. Deploy!
railway up

# 6. Add domain
railway domain
# Will give you: https://minimal-ai-backend-production.up.railway.app
```

**Environment Variables (Railway Dashboard):**
```
DATABASE_URL=<your-postgres-url>
GOOGLE_API_KEY=<your-gemini-key>
JWT_SECRET_KEY=<random-secret>
VOICE_MATCH_THRESHOLD=0.85
```

**Cost:** ~$5-10/month (free $5 credit = essentially free for low traffic)

---

### Option 2: Render.com

**Why Render:**
- Free tier available
- GitHub auto-deploy
- Easy setup

**Setup:**

1. **Push to GitHub:**
```bash
cd /Users/abhinavsingh/Documents/DesktopData/minimal-ai
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/minimal-ai.git
git push -u origin main
```

2. **Create `render.yaml`:**
```yaml
services:
  - type: web
    name: minimal-ai-backend
    env: python
    region: oregon
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: DATABASE_URL
        sync: false
      - key: GOOGLE_API_KEY
        sync: false
```

3. **Deploy on Render:**
- Go to render.com → New → Web Service
- Connect GitHub repo
- Select `backend` directory
- Deploy!

**URL:** `https://minimal-ai-backend.onrender.com`

**Cost:** Free tier (spins down after inactivity), or $7/month for always-on

---

### Option 3: Google Cloud Run (Serverless)

**Why Cloud Run:**
- Pay only for requests
- Auto-scaling
- Free tier: 2 million requests/month

**Setup:**

1. **Install Google Cloud SDK:**
```bash
brew install --cask google-cloud-sdk
gcloud init
```

2. **Create `Dockerfile` (already exists in backend/):**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

3. **Deploy:**
```bash
cd backend
gcloud run deploy minimal-ai --source . --region us-central1 --allow-unauthenticated
```

**URL:** `https://minimal-ai-<hash>-uc.a.run.app`

**Cost:** ~$1-5/month for typical usage (free tier covers most small apps)

---

### Option 4: DigitalOcean App Platform

1. Connect GitHub
2. Select repo & backend directory
3. Auto-detects Python
4. Deploy

**Cost:** $12/month (includes 1GB RAM, auto-scaling)

---

## 🔄 Switching Between Local & Cloud

### Use Build Variants

```kotlin
// app/build.gradle.kts
android {
    buildTypes {
        debug {
            buildConfigField("String", "API_URL", "\"http://10.0.2.2:8000/\"")
        }
        release {
            buildConfigField("String", "API_URL", "\"https://your-app.railway.app/\"")
        }
    }
}

// ApiConfig.kt
object ApiConfig {
    val BASE_URL = BuildConfig.API_URL
}
```

**Development:**
- Use debug build → Connects to local Mac
- Fast iteration, no deploy needed

**Testing/Production:**
- Use release build → Connects to cloud
- Shareable with testers

---

## 📦 Database Setup

### Local Development

**Option A: Use Docker (Recommended)**

```bash
cd /Users/abhinavsingh/Documents/DesktopData/minimal-ai/db

# Start PostgreSQL
docker-compose up -d

# Check status
docker ps

# View logs
docker logs minimal-ai-db
```

**Connection string:** `postgresql://postgres:minimalai@localhost:5432/minimal_ai`

**Option B: Install PostgreSQL on Mac**

```bash
brew install postgresql@14
brew services start postgresql@14
createdb minimal_ai
psql minimal_ai < init.sql
```

### Cloud Database

**Recommended: Neon.tech** (Free PostgreSQL)
1. Go to neon.tech
2. Create free account
3. Create database
4. Copy connection string
5. Add to Railway/Render environment variables

**Or use Railway's built-in Postgres:**
```bash
railway add
# Select PostgreSQL
# Connection string auto-added to env vars
```

---

## 🧪 Testing the Integration

### 1. Health Check

```bash
# Test backend is running
curl http://localhost:8000/health

# Or from phone
curl http://192.168.1.45:8000/health
```

### 2. Full Flow Test

**A. Signup:**
```bash
curl -X POST http://localhost:8000/auth/signup \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "testpass123",
    "full_name": "Test User"
  }'
```

**B. Login:**
```bash
curl -X POST http://localhost:8000/auth/login \
  -d "email=test@example.com&password=testpass123"
```

**C. Voice Enrollment:**
```bash
TOKEN="<your-jwt-token>"
curl -X POST http://localhost:8000/voice/enroll \
  -H "Authorization: Bearer $TOKEN" \
  -F "audio=@test_audio.wav"
```

### 3. Android Integration Test

```kotlin
// Test in Android app
class AuthRepositoryTest {
    private val api = ApiServiceFactory.create()
    
    @Test
    fun testSignup() = runBlocking {
        val response = api.signup(
            email = "test@example.com",
            password = "testpass123",
            fullName = "Test User"
        )
        assert(response.accessToken.isNotEmpty())
    }
}
```

---

## 🚨 Troubleshooting

### "Connection Refused" Error

**Problem:** Android app can't reach backend

**Solutions:**
1. **Check firewall:** Mac firewall may block incoming connections
   - System Settings → Network → Firewall → Allow Python
2. **Check WiFi:** Phone and Mac on same network?
3. **Check IP:** Use correct IP address from `ifconfig`
4. **Check backend:** Is uvicorn running? Check terminal

### "SSL Certificate Error" (Cloud)

**Problem:** HTTPS certificate issues

**Solution:** Railway/Render auto-handle HTTPS. If issues:
```kotlin
// TEMPORARY FIX (don't use in production!)
val okHttpClient = OkHttpClient.Builder()
    .hostnameVerifier { _, _ -> true }
    .build()
```

### "404 Not Found"

**Problem:** Endpoint not found

**Solution:**
1. Check API docs: `http://localhost:8000/docs`
2. Verify endpoint path matches exactly
3. Check HTTP method (POST vs GET)

---

## 📊 Monitoring & Logs

### Local Development

```bash
# Backend logs
# Just watch your terminal where uvicorn is running

# Database logs
docker logs -f minimal-ai-db
```

### Production (Railway)

```bash
# View logs
railway logs

# Or in Railway dashboard
# Project → Deployments → View Logs
```

---

## 🎯 Recommended Setup for You

**For Development (Now):**
```
✅ Run backend locally on Mac (uvicorn)
✅ Use local PostgreSQL (Docker)
✅ Android emulator with 10.0.2.2:8000
✅ Free, fast iteration
```

**For Testing on Real Phone:**
```
✅ Find Mac's local IP (192.168.1.X)
✅ Update Android app BASE_URL
✅ Both on same WiFi
✅ Test with real device
```

**For Production (Later):**
```
✅ Deploy to Railway.app ($5 free credit)
✅ Use Neon.tech for PostgreSQL (free)
✅ Update Android app to use Railway URL
✅ Share with friends/testers
```

---

## 💰 Cost Estimates

| Setup | Monthly Cost |
|-------|--------------|
| **Local only** | $0 (free) |
| **Railway + Neon** | $0-5 (free tiers) |
| **Render + Neon** | $0-7 (free/paid tier) |
| **Cloud Run + Neon** | $1-5 (pay per use) |
| **DigitalOcean** | $12 (fixed) |

**Recommendation:** Start local → Test on Railway free tier → Upgrade if needed

---

## ✅ Quick Start Checklist

- [ ] Backend running locally (`uvicorn` command)
- [ ] Database running (Docker or local)
- [ ] Found Mac's IP address
- [ ] Updated Android `ApiConfig.kt`
- [ ] Tested `/health` endpoint
- [ ] Created test account via API
- [ ] Android app successfully calls backend
- [ ] (Optional) Deployed to Railway for testing

**You're ready to build!** 🚀
