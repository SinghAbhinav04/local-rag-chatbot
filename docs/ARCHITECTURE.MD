# Architecture & Infrastructure Decisions

This document explains **how and why** minimal.ai is structured the way it is.

The goal:
- Clarity
- Debuggability
- Scalability
- Learning real-world system design

---

## 🧠 Core Architectural Idea

minimal.ai is a **task-oriented AI system**, not a chat application.

This means:
- Sessions represent ongoing tasks
- Requests represent user intents
- Steps represent execution phases
- Responses represent outcomes

---

## 🗃️ Database-First Design

We intentionally designed the database **before writing backend code**.

### Why?
- Data models are hardest to change later
- Forces clarity of system behavior
- Prevents ORM-driven bad schemas
- Mirrors real production practices

### Process
1. Design schema in raw SQL
2. Test manually via psql
3. Validate constraints & relationships
4. Introduce backend logic later

---

## 🐘 PostgreSQL (Dockerized)

PostgreSQL runs locally using Docker.

### Reasons
- No local installation issues
- Reproducible setup
- Production-like behavior
- Easy resets for development

The database is treated as a **long-lived stateful component**.

---

## 🧩 Docker Compose Separation

The database is intentionally isolated from the backend.

### Why?
- Database is stateful
- Backend is stateless
- Different lifecycles
- Prevents accidental data loss
- Mirrors real-world infrastructure

---

## 📁 Project Structure

minimal-ai/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── db/
│   │   ├── orchestrator/
│   └── Dockerfile
├── db/
│   ├── init.sql
│   ├── docker-compose.yml
├── docker-compose.yml
└── .env

---

## 🧠 Database Schema Philosophy

The schema supports:
- Users & authentication
- Sessions (chat / task)
- Requests (user intent)
- Responses (final output)
- Request steps (planning & execution trace)

This enables:
- Task tracking
- Debuggable AI behavior
- Cost & latency monitoring
- Safe iteration on orchestration logic

---

## 🚫 What We Are NOT Doing (Yet)

This is intentional.

- No vector database
- No Redis
- No autoscaling
- No billing
- No UI coupling
- No premature Kubernetes

---

## ☸️ Why Docker Before Kubernetes

Docker teaches:
- Containers
- Volumes
- Networking
- Process isolation

Kubernetes is introduced **only after** fundamentals are solid.

---

## 🔮 Long-Term Direction

- Backend deployed to Kubernetes
- PostgreSQL becomes managed service
- Orchestrator supports advanced strategies
- Web & Android clients added
- Task automation expanded

---

## 🧠 Guiding Principle

> **Build boring infrastructure first.  
Add intelligence on top.**
