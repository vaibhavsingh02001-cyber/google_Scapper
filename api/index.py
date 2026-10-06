"""
api/index.py - FastAPI Serverless Endpoint for Discovery Engine AI Assistant on Vercel.
Serves both static frontend assets (index.html, styles.css, app.js, data) and AI streaming endpoints (/api/chat, /api/health).
"""
import json, os, sys
from pathlib import Path
from typing import List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
    from pydantic import BaseModel
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]", "pydantic"])
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
    from pydantic import BaseModel

try:
    from groq import Groq
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "groq"])
    from groq import Groq

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_PRIMARY  = "llama-3.3-70b-versatile"
LLM_FALLBACK = "llama-3.1-8b-instant"

app = FastAPI(title="Discovery Engine & AI Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def find_file(rel_path: str) -> Optional[Path]:
    filename = Path(rel_path).name
    candidates = [
        Path(__file__).resolve().parent / rel_path,
        Path(__file__).resolve().parent / filename,
        Path(__file__).resolve().parent / "data" / filename,
        Path(__file__).resolve().parent.parent / rel_path,
        Path(__file__).resolve().parent.parent / filename,
        Path.cwd() / rel_path,
        Path.cwd() / filename,
        Path(rel_path),
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    # Recursive fallback
    for base in [Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent, Path.cwd()]:
        try:
            for root, _, files in os.walk(base):
                if filename in files:
                    return Path(root) / filename
        except Exception:
            pass
    return None

_BUNDLE: dict = {}
_DATA_CTX: str = ""

def load_bundle():
    global _BUNDLE, _DATA_CTX
    bundle_path = find_file("data/bundle.json") or find_file("bundle.json")
    if not bundle_path:
        _DATA_CTX = "No data bundle available. Answer general questions."
        return

    try:
        with open(bundle_path, encoding="utf-8") as f:
            _BUNDLE = json.load(f)
        items  = _BUNDLE.get("items", [])
        opps   = _BUNDLE.get("opportunity_areas", [])
        report = _BUNDLE.get("synthesis_report", "")
        signal_lines = []
        for it in items[:60]:
            src   = it.get("source", "")
            txt   = (it.get("text") or "")[:280]
            fps   = ", ".join(it.get("failure_points") or [])
            areas = ", ".join(it.get("assigned_areas") or [])
            signal_lines.append(f'[{src}] "{txt}" | Failure: {fps} | Area: {areas}')
        opp_lines = []
        for o in opps:
            name = o.get("Opportunity Area") or o.get("name", "")
            desc = (o.get("Description") or o.get("description", ""))[:180]
            vol  = o.get("Evidence Volume") or o.get("evidence_count", "?")
            sev  = o.get("Severity Score") or o.get("avg_frustration", "?")
            opp_lines.append(f"- {name} (Vol:{vol}, Sev:{sev}/3) -- {desc}")
        _DATA_CTX = (
            f"=== DISCOVERY ENGINE DATA ===\n"
            f"Dataset: {len(items)} verified signals (Play Store, Reddit, App Store, YouTube, Help Forums, Web)\n\n"
            f"OPPORTUNITY AREAS:\n" + "\n".join(opp_lines[:7]) + "\n\n"
            f"VERBATIM SIGNALS (sample {min(60, len(items))}):\n" + "\n".join(signal_lines) + "\n\n"
            f"EXECUTIVE SYNTHESIS:\n{report[:1400] if report else 'N/A'}\n"
            f"=== END DATA ==="
        )
    except Exception as e:
        _DATA_CTX = f"Error reading bundle: {e}"

load_bundle()

def system_prompt():
    return (
        "You are the **Google Photos Discovery Assistant** -- an expert AI analyst embedded in a research intelligence dashboard.\n\n"
        "You have deep knowledge of a curated dataset of real user feedback about photo retrieval and memory recall problems in Google Photos.\n\n"
        "Rules:\n"
        "- For dataset questions: cite specific signals, percentages, user quotes from the data context.\n"
        "- For general questions: answer helpfully like a knowledgeable colleague.\n"
        "- Always use markdown formatting: headers, bullet lists, tables, bold text where useful.\n"
        "- Be warm, precise, and conversational. Keep responses focused and insightful.\n\n"
        + _DATA_CTX
    )

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "mixtral-8x7b-32768"
]

async def handle_chat_stream(req: ChatRequest):
    api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    if not api_key:
        raise HTTPException(503, "GROQ_API_KEY not set in environment")
    client = Groq(api_key=api_key)
    msgs = [{"role": "system", "content": system_prompt()}]
    msgs += [{"role": m.role, "content": m.content} for m in req.messages]

    def gen():
        last_err = ""
        for model in GROQ_MODELS:
            try:
                stream = client.chat.completions.create(
                    model=model, messages=msgs, temperature=0.7,
                    max_tokens=1024, stream=True
                )
                for chunk in stream:
                    d = chunk.choices[0].delta
                    if d and d.content:
                        yield f"data: {json.dumps({'t': d.content})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return
            except Exception as e:
                last_err = str(e)
                continue
        yield f"data: {json.dumps({'error': f'Groq AI Error: {last_err}'})}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

def handle_health():
    api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    return {"ok": True, "key": bool(api_key), "items": len(_BUNDLE.get("items", []))}

# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.post("/chat")
@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    return await handle_chat_stream(req)

@app.get("/health")
@app.get("/api/health")
async def health_endpoint():
    return handle_health()

@app.get("/api")
async def api_status():
    return {"status": "Discovery Engine API is running"}

# ── Static File Endpoints (Guarantees UI renders on any Vercel routing) ───────
@app.get("/")
@app.get("/index.html")
async def serve_index():
    f = find_file("index.html")
    if f:
        return FileResponse(f, media_type="text/html")
    return HTMLResponse("<h1>Google Photos Discovery Engine</h1><p>index.html not found</p>")

@app.get("/styles.css")
async def serve_css():
    f = find_file("styles.css")
    if f:
        return FileResponse(f, media_type="text/css")
    raise HTTPException(404, "styles.css not found")

@app.get("/app.js")
async def serve_js():
    f = find_file("app.js")
    if f:
        return FileResponse(f, media_type="application/javascript")
    raise HTTPException(404, "app.js not found")

@app.get("/data/{filename}")
async def serve_data_file(filename: str):
    f = find_file(f"data/{filename}") or find_file(filename)
    if f:
        media_type = "application/json" if filename.endswith(".json") else "text/plain"
        return FileResponse(f, media_type=media_type)
    raise HTTPException(404, f"data/{filename} not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
