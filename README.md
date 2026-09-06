# KaziIQ — Multi-Agent Career Intelligence

A production multi-agent AI system for the East African job market.
Built with **JarvisCore** + **Gemini 3.6 Flash** | Deployed on **Render.com**

## Architecture

Three agents run sequentially — each builds on the previous agent's output:

| Agent | Role | Output |
|-------|------|--------|
| **Scout** | Job Description Analysis | Structured requirements, skills, responsibilities |
| **Analyst** | CV Fit Evaluation | Fit score, matched skills, gaps, strategic advice |
| **Coach** | Application Materials | Tailored cover letter + 5 interview questions |

## Stack

- **Agent Framework:** JarvisCore (`jarviscore-framework`)
- **LLM:** Gemini 3.6 Flash via JarvisCore's Gemini adapter
- **API:** FastAPI + Uvicorn
- **Deployment:** Render.com (free tier)

## Local Setup

```bash
git clone https://github.com/bluprint023/kaziiq
cd kaziiq
pip install -r requirements.txt
cp .env.example .env        # add your GEMINI_API_KEY
uvicorn main:app --reload
```

Open http://localhost:8000

## Environment Variables

| Variable | Value |
|----------|-------|
| `GEMINI_API_KEY` | Your Google AI Studio key |
| `GEMINI_MODEL` | `gemini-3.6-flash` |
