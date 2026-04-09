from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime
import json

app = FastAPI()
sessions = []
total_patterns = 0

@app.post("/analyze")
async def analyze(request: Request):
    global total_patterns
    data = await request.json()
    data["timestamp"] = datetime.utcnow().isoformat()
    sessions.append(data)
    total_patterns += 1

    miss_streak = data.get("miss_streak", 0)
    confidence  = data.get("confidence", 100)
    suggestion  = {}

    if miss_streak >= 3:
        suggestion["bf_phase"] = "Phase 2 (Aggressive)"
    if miss_streak >= 6:
        suggestion["bf_phase"] = "Phase 3 (Custom)"
        suggestion["override_baim"] = True
    if confidence < 50:
        suggestion["override_baim"] = True

    return JSONResponse(suggestion)

@app.get("/stats")
async def stats():
    return {
        "users_online":      1,
        "patterns_saved":    total_patterns,
        "resolver_records":  len([s for s in sessions if s.get("resolver_mode")]),
        "ai_iterations":     total_patterns * 3,
        "avg_confidence":    round(sum(s.get("confidence",70) for s in sessions) / max(len(sessions),1)),
        "last_sync":         sessions[-1]["timestamp"] if sessions else "Never",
        "your_contribution": "Active" if sessions else "Inactive"
    }

@app.get("/")
async def root():
    return {"status": "hematic backend online"}
