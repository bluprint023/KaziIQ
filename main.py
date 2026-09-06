"""
KaziIQ — Multi-Agent Career Intelligence System
JarvisCore + Gemini API | Deployed on Render.com
"""

import asyncio
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from jarviscore import Mesh, AutoAgent

# ─────────────────────────────────────────────
#  AGENT DEFINITIONS
# ─────────────────────────────────────────────

class ScoutAgent(AutoAgent):
    name = "Scout"
    role = "scout"
    capabilities = ["job_analysis", "text_extraction", "requirements_parsing"]
    description = "Extracts and structures requirements from job descriptions."
    system_prompt = """
    You are a career intelligence analyst specialising in the East African job market.
    When given a job description, extract and return a structured summary with:

    1. ROLE TITLE: The exact job title
    2. SENIORITY: Entry / Mid / Senior / Lead
    3. REQUIRED SKILLS: Technical and soft skills explicitly required (bullet list)
    4. NICE TO HAVE: Skills mentioned as preferred but not mandatory (bullet list)
    5. KEY RESPONSIBILITIES: Top 5 core duties in plain language (bullet list)
    6. RED FLAGS: Any unusual requirements or concerns worth noting

    Be precise. Do not invent information not present in the job description.
    Format your response with clear headers for each section.
    """


class AnalystAgent(AutoAgent):
    name = "Analyst"
    role = "analyst"
    capabilities = ["cv_evaluation", "gap_analysis", "fit_scoring"]
    description = "Evaluates CV fit against job requirements and identifies skill gaps."
    system_prompt = """
    You are a brutal but fair career analyst. You receive structured job requirements
    and a candidate's CV. Your job is to:

    1. FIT SCORE: Give a percentage fit score (0-100%) with a one-line justification
    2. MATCHED SKILLS: List skills from the CV that directly match requirements
    3. SKILL GAPS: List required skills missing or weak in the CV — be honest and specific
    4. STRENGTHS TO LEAD WITH: 2-3 things the candidate should emphasise for THIS role
    5. STRATEGIC ADVICE: 2-3 concrete, actionable things the candidate should do

    Do not sugarcoat. A 45% fit score is better served honestly than a false 80%.
    Format your response with clear headers for each section.
    """


class CoachAgent(AutoAgent):
    name = "Coach"
    role = "coach"
    capabilities = ["cover_letter_writing", "interview_preparation", "career_coaching"]
    description = "Generates cover letters and interview preparation materials."
    system_prompt = """
    You are a career coach who writes direct, non-generic application materials
    for the East African professional context.

    You receive a job analysis and CV assessment. From this, produce:

    COVER LETTER:
    - Max 350 words
    - Opening: hook with a specific, relevant accomplishment — never start with
      "I am writing to apply for..."
    - Body: address 2 specific requirements from the job using evidence from the CV
    - Closing: confident, specific ask — no filler phrases
    - Tone: professional but direct, not stiff

    INTERVIEW PREPARATION:
    List 5 questions the interviewer is most likely to ask for THIS specific role,
    with a one-paragraph coaching note on how to approach each answer.

    Format with clear section headers.
    """


# ─────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────

async def run_pipeline(job_description: str, cv_text: str) -> dict:
    mesh = Mesh()
    mesh.add(ScoutAgent)
    mesh.add(AnalystAgent)
    mesh.add(CoachAgent)
    await mesh.start()

    scout_output = await mesh.run_task(
        agent="scout",
        task=f"Extract and structure the requirements from this job description:\n\n{job_description}"
    )

    analyst_output = await mesh.run_task(
        agent="analyst",
        task=(
            f"Here are the structured job requirements:\n\n{scout_output}\n\n"
            f"Here is the candidate's CV:\n\n{cv_text}\n\n"
            f"Evaluate the fit and identify gaps."
        )
    )

    coach_output = await mesh.run_task(
        agent="coach",
        task=(
            f"Job Analysis:\n{scout_output}\n\n"
            f"CV Assessment:\n{analyst_output}\n\n"
            f"Generate a tailored cover letter and interview preparation materials."
        )
    )

    await mesh.stop()

    return {
        "scout": str(scout_output),
        "analyst": str(analyst_output),
        "coach": str(coach_output),
    }


# ─────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="KaziIQ",
    description="Multi-Agent Career Intelligence for East Africa",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    job_description: str
    cv_text: str


class AnalysisResponse(BaseModel):
    scout: str
    analyst: str
    coach: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "KaziIQ"}


@app.post("/analyse", response_model=AnalysisResponse)
async def analyse(request: AnalysisRequest):
    results = await run_pipeline(request.job_description, request.cv_text)
    return AnalysisResponse(**results)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


# ─────────────────────────────────────────────
#  FRONTEND (single-file HTML)
# ─────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>KaziIQ — Multi-Agent Career Intelligence</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #0a0f1e;
      --surface:   #111827;
      --border:    #1f2d45;
      --accent:    #3b82f6;
      --accent-dim:#1d4ed8;
      --green:     #10b981;
      --amber:     #f59e0b;
      --red:       #ef4444;
      --text:      #e2e8f0;
      --muted:     #64748b;
      --mono:      'JetBrains Mono', monospace;
    }

    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      line-height: 1.6;
    }

    /* ── HEADER ── */
    header {
      border-bottom: 1px solid var(--border);
      padding: 1.25rem 2rem;
      display: flex;
      align-items: center;
      gap: 1rem;
    }
    .logo-mark {
      width: 36px; height: 36px;
      background: var(--accent);
      border-radius: 8px;
      display: grid; place-items: center;
      font-weight: 700; font-size: 0.85rem; color: white;
    }
    .logo-text { font-weight: 700; font-size: 1.15rem; }
    .logo-sub  { font-size: 0.78rem; color: var(--muted); margin-left: auto; }

    /* ── PIPELINE BAR ── */
    .pipeline {
      display: flex;
      align-items: center;
      gap: 0;
      padding: 1rem 2rem;
      border-bottom: 1px solid var(--border);
      background: var(--surface);
      overflow-x: auto;
    }
    .pipe-step {
      display: flex; align-items: center; gap: 0.5rem;
      font-size: 0.78rem; font-weight: 600;
      color: var(--muted);
      white-space: nowrap;
    }
    .pipe-step.active { color: var(--accent); }
    .pipe-step.done   { color: var(--green); }
    .pipe-dot {
      width: 28px; height: 28px; border-radius: 50%;
      border: 2px solid currentColor;
      display: grid; place-items: center;
      font-size: 0.7rem; font-weight: 700;
    }
    .pipe-arrow {
      width: 2.5rem; height: 1px;
      background: var(--border);
      margin: 0 0.25rem;
    }

    /* ── LAYOUT ── */
    main {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0;
      height: calc(100vh - 113px);
    }
    .panel {
      padding: 1.5rem 2rem;
      overflow-y: auto;
      border-right: 1px solid var(--border);
    }
    .panel:last-child { border-right: none; }
    .panel-label {
      font-size: 0.7rem; font-weight: 600;
      color: var(--muted); letter-spacing: 0.08em;
      text-transform: uppercase; margin-bottom: 1rem;
    }

    /* ── INPUTS ── */
    textarea {
      width: 100%;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      font-family: var(--mono);
      font-size: 0.78rem;
      line-height: 1.7;
      padding: 0.875rem 1rem;
      resize: vertical;
      outline: none;
      transition: border-color 0.15s;
    }
    textarea:focus { border-color: var(--accent); }
    label {
      display: block;
      font-size: 0.8rem; font-weight: 600;
      color: var(--text); margin-bottom: 0.4rem;
    }
    .field { margin-bottom: 1.25rem; }

    /* ── BUTTON ── */
    button#run-btn {
      width: 100%;
      padding: 0.75rem 1.5rem;
      background: var(--accent);
      color: white; font-weight: 700; font-size: 0.95rem;
      border: none; border-radius: 8px; cursor: pointer;
      transition: background 0.15s, transform 0.1s;
      display: flex; align-items: center; justify-content: center; gap: 0.5rem;
    }
    button#run-btn:hover:not(:disabled) { background: var(--accent-dim); }
    button#run-btn:active:not(:disabled) { transform: scale(0.98); }
    button#run-btn:disabled { opacity: 0.5; cursor: not-allowed; }

    /* ── OUTPUT TABS ── */
    .tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 1.25rem; }
    .tab {
      padding: 0.6rem 1.1rem;
      font-size: 0.8rem; font-weight: 600;
      color: var(--muted); cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: color 0.15s, border-color 0.15s;
      white-space: nowrap;
    }
    .tab.active { color: var(--accent); border-color: var(--accent); }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }

    /* ── OUTPUT CONTENT ── */
    .output-box {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1.25rem;
      font-size: 0.82rem;
      line-height: 1.75;
      white-space: pre-wrap;
      word-break: break-word;
      min-height: 200px;
      font-family: var(--mono);
    }
    .output-box.empty { color: var(--muted); font-style: italic; }

    /* ── STATUS BADGES ── */
    .badge {
      display: inline-flex; align-items: center; gap: 0.35rem;
      font-size: 0.7rem; font-weight: 600; padding: 0.2rem 0.6rem;
      border-radius: 999px; margin-bottom: 0.75rem;
    }
    .badge.waiting { background: #1f2d45; color: var(--muted); }
    .badge.running { background: #1e3a5f; color: var(--accent); }
    .badge.done    { background: #064e3b; color: var(--green); }
    .badge.error   { background: #450a0a; color: var(--red); }
    .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
    .dot.pulse { animation: pulse 1s infinite; }

    /* ── SPINNER ── */
    .spinner {
      width: 14px; height: 14px;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: white;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      display: none;
    }

    @keyframes spin  { to { transform: rotate(360deg); } }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

    /* ── RESPONSIVE ── */
    @media (max-width: 768px) {
      main { grid-template-columns: 1fr; height: auto; }
      .panel { border-right: none; border-bottom: 1px solid var(--border); }
    }
  </style>
</head>
<body>

<header>
  <div class="logo-mark">KQ</div>
  <span class="logo-text">KaziIQ</span>
  <span class="logo-sub">Multi-Agent Career Intelligence · East Africa</span>
</header>

<div class="pipeline">
  <div class="pipe-step" id="step-scout">
    <div class="pipe-dot">1</div> Scout — Extract Requirements
  </div>
  <div class="pipe-arrow"></div>
  <div class="pipe-step" id="step-analyst">
    <div class="pipe-dot">2</div> Analyst — Score CV Fit
  </div>
  <div class="pipe-arrow"></div>
  <div class="pipe-step" id="step-coach">
    <div class="pipe-dot">3</div> Coach — Generate Materials
  </div>
</div>

<main>
  <!-- INPUT PANEL -->
  <div class="panel">
    <div class="panel-label">Inputs</div>

    <div class="field">
      <label for="jd">Job Description</label>
      <textarea id="jd" rows="13" placeholder="Paste the full job description here...">Data Scientist - Nairobi, Kenya

We are looking for a Data Scientist to join our analytics team at a leading Kenyan fintech company.

Requirements:
- Bachelor's degree in Computer Science, Statistics, or related field
- 2+ years experience in data science or machine learning roles
- Proficiency in Python (pandas, scikit-learn, numpy)
- Experience with SQL databases
- Familiarity with cloud platforms (GCP, AWS, or Azure)
- Strong data visualization skills (Tableau, Power BI, or similar)
- Experience building and deploying ML models in production

Nice to have:
- Experience with NLP or time series forecasting
- Knowledge of financial services or mobile money
- Familiarity with dbt or Airflow

Responsibilities:
- Build predictive models for credit scoring and fraud detection
- Design and run A/B experiments to improve product features
- Collaborate with engineering to deploy models to production
- Create dashboards and reports for business stakeholders
- Work with large transaction datasets to surface insights</textarea>
    </div>

    <div class="field">
      <label for="cv">Your CV / Resume</label>
      <textarea id="cv" rows="13" placeholder="Paste your full CV text here...">RAIA KAWAIDA
Nairobi, Kenya | github.com/vitumob026

EDUCATION
BSc Computer Science — UoN (2024)
Moringa Data Science Programme (2026)
Cisco CCNA (2022)

EXPERIENCE
Full stack developer — ChapaKazi Inc (Sep 2025 – )
- Designed, developed and maintained websites.
- Trained users of these websites

PROJECTS
Weka kakitu — EPL Match Outcome Predictor
- ML pipeline using scikit-learn, XGBoost, FastAPI
- Automated API data ingestion from football statistics APIs

Kenya Fertilizer Subsidy Access Gap Analysis
- ETL pipeline across all 47 counties
- Two allocation models with policy brief output

JKUAT Multilingual Text Classifier
- 95.5% accuracy on 14,778 samples
- English/Kiswahili/Sheng classification

SKILLS
Python (pandas, scikit-learn, XGBoost, FastAPI), SQL, Power BI,
Machine Learning, NLP, Network Monitoring
Teamwork, Leadership, Communication</textarea>
    </div>

    <button id="run-btn" onclick="runAnalysis()">
      <div class="spinner" id="spinner"></div>
      <span id="btn-text">Run Analysis →</span>
    </button>
  </div>

  <!-- OUTPUT PANEL -->
  <div class="panel">
    <div class="panel-label">Agent Outputs</div>

    <div class="tabs">
      <div class="tab active" onclick="switchTab('scout')">🔍 Scout</div>
      <div class="tab" onclick="switchTab('analyst')">📊 Analyst</div>
      <div class="tab" onclick="switchTab('coach')">✍️ Coach</div>
    </div>

    <!-- Scout Tab -->
    <div class="tab-panel active" id="tab-scout">
      <div class="badge waiting" id="badge-scout">
        <div class="dot"></div> Waiting
      </div>
      <div class="output-box empty" id="out-scout">
Agent 1 output will appear here.

The Scout reads the job description and extracts structured requirements — role title, seniority, required skills, nice-to-haves, responsibilities, and red flags.
      </div>
    </div>

    <!-- Analyst Tab -->
    <div class="tab-panel" id="tab-analyst">
      <div class="badge waiting" id="badge-analyst">
        <div class="dot"></div> Waiting
      </div>
      <div class="output-box empty" id="out-analyst">
Agent 2 output will appear here.

The Analyst compares your CV against the Scout's extracted requirements and produces a fit score, matched skills, skill gaps, and strategic advice.
      </div>
    </div>

    <!-- Coach Tab -->
    <div class="tab-panel" id="tab-coach">
      <div class="badge waiting" id="badge-coach">
        <div class="dot"></div> Waiting
      </div>
      <div class="output-box empty" id="out-coach">
Agent 3 output will appear here.

The Coach takes the Analyst's assessment and generates a tailored cover letter (≤350 words) plus 5 predicted interview questions with coaching notes.
      </div>
    </div>
  </div>
</main>

<script>
  function switchTab(name) {
    document.querySelectorAll('.tab').forEach((t, i) => {
      const names = ['scout', 'analyst', 'coach'];
      t.classList.toggle('active', names[i] === name);
    });
    document.querySelectorAll('.tab-panel').forEach(p => {
      p.classList.toggle('active', p.id === 'tab-' + name);
    });
  }

  function setBadge(agent, state) {
    const badge = document.getElementById('badge-' + agent);
    const labels = { waiting: 'Waiting', running: 'Running…', done: 'Done', error: 'Error' };
    badge.className = 'badge ' + state;
    badge.innerHTML = `<div class="dot ${state === 'running' ? 'pulse' : ''}"></div> ${labels[state]}`;
  }

  function setOutput(agent, text, isEmpty = false) {
    const el = document.getElementById('out-' + agent);
    el.textContent = text;
    el.classList.toggle('empty', isEmpty);
  }

  async function runAnalysis() {
    const jd = document.getElementById('jd').value.trim();
    const cv = document.getElementById('cv').value.trim();

    if (!jd || !cv) {
      alert('Please fill in both the job description and your CV.');
      return;
    }

    // Lock UI
    const btn = document.getElementById('run-btn');
    btn.disabled = true;
    document.getElementById('spinner').style.display = 'block';
    document.getElementById('btn-text').textContent = 'Running pipeline…';

    // Reset all agents
    ['scout', 'analyst', 'coach'].forEach(a => {
      setBadge(a, 'waiting');
      setOutput(a, 'Waiting for pipeline to reach this agent…', true);
    });

    // Show scout running first
    setBadge('scout', 'running');
    switchTab('scout');
    setOutput('scout', 'Scout is reading the job description…', true);

    try {
      const res = await fetch('/analyse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_description: jd, cv_text: cv })
      });

      if (!res.ok) {
        const err = await res.text();
        throw new Error(err);
      }

      const data = await res.json();

      // Scout done
      setBadge('scout', 'done');
      setOutput('scout', data.scout);

      // Analyst
      setBadge('analyst', 'done');
      setOutput('analyst', data.analyst);

      // Coach
      setBadge('coach', 'done');
      setOutput('coach', data.coach);

      // Auto-switch to analyst (most interesting first result)
      switchTab('analyst');

    } catch (err) {
      ['scout', 'analyst', 'coach'].forEach(a => setBadge(a, 'error'));
      setOutput('scout', 'Error: ' + err.message);
      console.error(err);
    } finally {
      btn.disabled = false;
      document.getElementById('spinner').style.display = 'none';
      document.getElementById('btn-text').textContent = 'Run Analysis →';
    }
  }
</script>
</body>
</html>
"""
