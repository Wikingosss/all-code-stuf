import os
import threading
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Import biblioteki Bazy Danych Supabase
from supabase import create_client, Client

# Machine Learning Modules
try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    AI_ENABLED = True
except ImportError:
    AI_ENABLED = False

load_dotenv()

app = FastAPI()

# ===== Inicjalizacja Bazy Danych Supabase =====
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

# ===== Inicjalizacja Modelu AI =====
AI_MODEL_PATH = "resolver_ai.joblib"
AI_MODEL = None
TRAINING_IN_PROGRESS = False

if AI_ENABLED and os.path.exists(AI_MODEL_PATH):
    try:
        AI_MODEL = joblib.load(AI_MODEL_PATH)
        print("✅ Załadowano wytrenowany model AI z dysku!")
    except Exception as e:
        print(f"⚠️ Nie udało się załadować modelu AI: {e}")

# Zapasowa logika, na wypadek gdy brakuje wytrenowanego modelu
def fallback_logic(data):
    sug = {}
    m = data.get("miss_streak", 0)
    if m >= 3: sug["bf_phase"] = "Phase 2 (Aggressive)"
    if m >= 6: 
        sug["bf_phase"] = "Phase 3 (Custom)"
        sug["override_baim"] = True
    return sug

@app.post("/analyze")
async def analyze(request: Request):
    global global_patterns
    data = await request.json()
    data["timestamp"] = datetime.utcnow().isoformat()
    global_patterns += 1
    
    # 1. Zrzut Logów (Data Harvesting) do Supabase
    feature_vector = [
        data.get("miss_streak", 0),
        data.get("confidence", 100.0),
        data.get("choked_ticks", 0),
        data.get("distance", 0.0),
        data.get("duck_amount", 0.0),
        data.get("velocity_x", 0.0),
        data.get("velocity_y", 0.0)
    ]

    if supabase:
        try:
            db_payload = {
                "miss_streak": feature_vector[0],
                "confidence": feature_vector[1],
                "resolver_mode": data.get("resolver_mode", "Adaptive"),
                "bf_phase": data.get("bf_phase", "Phase 1"),
                "weapon": data.get("weapon", "Unknown"),
                "choked_ticks": feature_vector[2],
                "distance": feature_vector[3],
                "duck_amount": feature_vector[4],
                "velocity_x": feature_vector[5],
                "velocity_y": feature_vector[6]
            }
            # Odkładamy dane w tle - nie blokujemy requesta
            supabase.table("resolver_data").insert(db_payload).execute()
        except Exception as e:
            pass

    # 2. Inteligentna Predykcja przy pomocy modelu "Random Forest"
    suggestion = {}

    if AI_MODEL and AI_ENABLED:
        try:
            X_input = np.array([feature_vector])
            prediction = AI_MODEL.predict(X_input)[0] 
            
            # Wzorzec modelu zwraca numery od do 3 przypisując określone intencje na resolver:
            # 0 = Normalny resolver adaptacyjny
            # 1 = Wymuszona 2 faza (Aggressive Desync resolver)
            # 2 = 3 faza i poddanie się algorytmom Force Baim ze względu na brak trafień 
            # 3 = Hard-override baim ze względu statystyk np. gracz kuca i duża prędkość

            if prediction == 1:
                suggestion["bf_phase"] = "Phase 2 (Aggressive)"
            elif prediction == 2:
                suggestion["bf_phase"] = "Phase 3 (Custom)"
                suggestion["override_baim"] = True
            elif prediction == 3:
                suggestion["override_baim"] = True
            
            suggestion["ai_powered"] = True
        except Exception as e:
            print("❌ AI Prediction error:", e)
            suggestion = fallback_logic(data)
    else:
        # Brak modelu AI, fallback na warunki sztywne
        suggestion = fallback_logic(data)

    return JSONResponse(suggestion)


# --- Trener AI pracujący w Tle ---
def train_model_bg():
    global AI_MODEL, TRAINING_IN_PROGRESS
    try:
        print("Mój stary, odpalam trening modelu AI...")
        res = supabase.table("resolver_data").select("*").execute()
        
        if not res.data or len(res.data) < 10:
            print("Za mało danych w bazie by wytrenować logiczny model (minimum 10 potrzebne do stworzenia drzewa).")
            TRAINING_IN_PROGRESS = False
            return
            
        df = pd.DataFrame(res.data)
        
        # Cecha: X (Wektor wejściowy)
        features = ["miss_streak", "confidence", "choked_ticks", "distance", "duck_amount", "velocity_x", "velocity_y"]
        
        # Oczyszczanie brakujących kolumn
        for col in features:
            if col not in df.columns:
                df[col] = 0.0
        
        df.fillna(0, inplace=True)
        X = df[features]
        
        # Syntetyzujemy etykiety do testowego treningu na podstawie ludzkich heurystyk
        # By model miał punkt wyjścia (w przyszłości tutaj wstawimy Hit/Miss etykiety od samego klienta)
        def assign_label(row):
            if row['miss_streak'] >= 6: return 2
            if row['distance'] < 200 and row['velocity_x'] > 150: return 3 # Run & BAIM scenario
            if row['miss_streak'] >= 3: return 1
            if row['duck_amount'] > 0.8 and row['confidence'] < 70: return 1
            return 0
            
        y = df.apply(assign_label, axis=1)
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
        clf.fit(X, y)
        
        joblib.dump(clf, AI_MODEL_PATH)
        AI_MODEL = clf
        print("🧠 Trening zakończony! Model AI jest teraz aktywny i zapisany do resolver_ai.joblib")
        
    except Exception as e:
        print("❌ Wystąpił błąd w treningu:", e)
    finally:
        TRAINING_IN_PROGRESS = False

@app.post("/train")
async def trigger_training():
    global TRAINING_IN_PROGRESS
    if not AI_ENABLED:
        return {"status": "error", "message": "Scikit-Learn (AI) nie zainstalowany. Pobierz 'scikit-learn pandas joblib' na backendzie."}
    if not supabase:
        return {"status": "error", "message": "Baza Supabase (Railway) odrzuciła łączenie. AI musi z czegoś ściągnąć logi."}
    
    if TRAINING_IN_PROGRESS:
        return {"status": "info", "message": "Trening jest już w toku. AI potrzebuje chwili."}
        
    TRAINING_IN_PROGRESS = True
    t = threading.Thread(target=train_model_bg)
    t.start()
    return {"status": "success", "message": "Rozpoczęto trening Random Forest! Za chwilę model wywoła lepsze instrukcje BAIM/Resolver do gierki."}

@app.get("/stats")
async def stats():
    patterns_count = global_patterns
    
    if supabase:
        try:
            res = supabase.table("resolver_data").select("id", count="exact").limit(1).execute()
            if res.count is not None:
                patterns_count = res.count
        except:
            pass

    return {
        "users_online":      1, 
        "patterns_saved":    patterns_count,
        "ai_status":         "Active" if AI_MODEL else "Awaiting /train command",
        "last_sync":         datetime.utcnow().isoformat()
    }

@app.get("/")
async def root():
    return {"status": "$hematic AI Backend API - 1.0.0"}
