from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json, os

app = FastAPI()
sessions = []

@app.post("/analyze")
async def analyze(request: Request):
    data = await request.json()
    sessions.append(data)

    miss_streak = data.get("miss_streak", 0)
    confidence  = data.get("confidence", 100)
    mode        = data.get("resolver_mode", "Adaptive")

    suggestion = {}

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
    return {"total_sessions": len(sessions), "data": sessions[-10:]}

@app.get("/")
async def root():
    return {"status": "hematic backend online"}
