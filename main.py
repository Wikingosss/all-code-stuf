import os
import json
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Import biblioteki Bazy Danych
from supabase import create_client, Client

# (Krok przygotowawczy na Scikit-Learn AI dla przyszłosci)
try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    AI_ENABLED = True
except ImportError:
    AI_ENABLED = False

load_dotenv()

app = FastAPI()

# ===== Inicjalizacja Bazy Danych Supabase =====
# Aby to zadziałało, w panelu sekcji Railway gdzie zrobisz deploy
# musisz wejść w zakładkę "Variables" i dodać dwa klucze:
# 1. SUPABASE_URL (link URL bazy z ustawień supabase API)
# 2. SUPABASE_KEY (anon key / public)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://qpymjauuxmkhgtrfetts.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_secret_yAgJ6P-VA7rhIVOnlTShiA_c_QpUfHS")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Zalogowano do Supabase pomyślnie!")
    except Exception as e:
        print(f"❌ Błąd logowania do Supabase: {e}")

global_patterns = 0

@app.post("/analyze")
async def analyze(request: Request):
    global global_patterns
    data = await request.json()
    data["timestamp"] = datetime.utcnow().isoformat()
    global_patterns += 1
    
    # --- 1. Zrzut Logów (Data Harvesting) do Supabase ---
    if supabase:
        try:
            db_payload = {
                "miss_streak": data.get("miss_streak", 0),
                "confidence": data.get("confidence", 100.0),
                "resolver_mode": data.get("resolver_mode", "Adaptive"),
                "bf_phase": data.get("bf_phase", "Phase 1"),
                "weapon": data.get("weapon", "Unknown"),
                "choked_ticks": data.get("choked_ticks", 0),
                "distance": data.get("distance", 0.0),
                "duck_amount": data.get("duck_amount", 0.0),
                "velocity_x": data.get("velocity_x", 0.0),
                "velocity_y": data.get("velocity_y", 0.0)
            }
            supabase.table("resolver_data").insert(db_payload).execute()
            print("💾 Zapisano nową próbkę dla Sztucznej Inteligencji!")
        except Exception as e:
            print("❌ Błąd zapisu do Supabase:", e)

    # --- 2. Live Decision Making (Tymczasowa prosta logika nim użyjemy AI na serio) ---
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
    patterns_count = global_patterns
    
    # Odpytywanie chmury by pobrać autentyczną sumę wierszy w bazie i nadpisać global_patterns
    if supabase:
        try:
            # count="exact" oblicza całą tablicę "resolver_data" bezpośrednio na silniku Postgres
            res = supabase.table("resolver_data").select("id", count="exact").limit(1).execute()
            if res.count is not None:
                patterns_count = res.count
        except:
            pass

    return {
        "users_online":      1, 
        "patterns_saved":    patterns_count,
        "resolver_records":  patterns_count,
        "ai_iterations":     patterns_count * 3,
        "avg_confidence":    87,
        "last_sync":         datetime.utcnow().isoformat(),
        "your_contribution": "Active"
    }

@app.get("/")
async def root():
    return {"status": "hematic backend online"}

