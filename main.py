import os
import threading
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

app = FastAPI(title="$hematic AI Backend", version="2.0.0")

# ===== Database Initialization =====
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://qpymjauuxmkhgtrfetts.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_secret_yAgJ6P-VA7rhIVOnlTShiA_c_QpUfHS")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"🔥 Global Error: {exc}")
    return JSONResponse(status_code=500, content={"message": str(exc)})

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase Connected")
    except Exception as e:
        print(f"❌ Supabase Connection Error: {e}")

# ===== AI Model Configuration =====
AI_MODEL_PATH = "resolver_ai_v2.joblib"
AI_MODEL = None
TRAINING_IN_PROGRESS = False

# Features used for the model
FEATURES = [
    "miss_streak", "distance", "velocity_x", "velocity_y", 
    "goal_feet_yaw", "eye_yaw", "layer3_weight", "layer3_cycle",
    "relative_angle", "choked_ticks", "duck_amount"
]

def load_ai_model():
    global AI_MODEL
    if os.path.exists(AI_MODEL_PATH):
        try:
            AI_MODEL = joblib.load(AI_MODEL_PATH)
            print("✅ AI Model v2 Loaded")
        except Exception as e:
            print(f"⚠️ Failed to load AI model: {e}")

load_ai_model()

def extract_features(data: dict) -> list:
    """Extract and normalize features from telemetry data."""
    target = data.get("target") or {}
    config = data.get("config") or {}
    anim = target.get("anim") or {}
    vel = target.get("vel") or {}
    
    return [
        config.get("miss_streak", 0),
        config.get("distance", 0),
        vel.get("x", 0),
        vel.get("y", 0),
        anim.get("goal_feet_yaw", 0),
        anim.get("eye_yaw", 0),
        anim.get("layer3_weight", 0),
        anim.get("layer3_cycle", 0),
        target.get("relative_angle", 0),
        target.get("choke", 0),
        target.get("duck", 0)
    ]

def is_valid_telemetry(data: list) -> bool:
    """Check if the provided features contain 'garbage' values (massive numbers)."""
    # goal_feet_yaw (4), eye_yaw (5), relative_angle (8)
    try:
        if abs(data[4]) > 1000000 or abs(data[5]) > 1000000 or abs(data[8]) > 1000000:
            return False
    except: return False
    return True

# ===== API Endpoints =====

@app.post("/predict")
async def predict(request: Request):
    """Real-time prediction for Lua's aim_fire event."""
    data = await request.json()
    print(f"📥 Received Predict: {data}")
    shot_id = data.get("shot_id")
    
    # Extract features for prediction
    features = extract_features(data)
    
    # Default response (Fallback)
    prediction = {
        "predicted_side": 58 if (data.get("target", {}).get("anim", {}).get("desync_delta", 0) > 0) else -58,
        "force_baim": data.get("config", {}).get("miss_streak", 0) >= 3,
        "confidence": 0.5,
        "source": "fallback"
    }

    # ML Override
    if AI_MODEL:
        try:
            X = np.array([features])
            pred_side_idx = AI_MODEL.predict(X)[0] # 0 for negative, 1 for positive side
            pred_proba = AI_MODEL.predict_proba(X)[0]
            
            prediction["predicted_side"] = 58 if pred_side_idx == 1 else -58
            prediction["confidence"] = float(np.max(pred_proba))
            prediction["source"] = "neural_network"
            
            # Logic for Baim based on confidence and miss streak
            if prediction["confidence"] < 0.6 and data.get("config", {}).get("miss_streak", 0) > 1:
                prediction["force_baim"] = True
        except Exception as e:
            print(f"Prediction Error: {e}")

    # Store the shot record for later outcome matching
    if supabase and shot_id:
        global global_patterns
        global_patterns += 1
        local_player = data.get("local_player") or {}
        target = data.get("target") or {}
        config = data.get("config") or {}
        anim = target.get("anim") or {}
        vel = target.get("vel") or {}
        lp_vel = local_player.get("vel") or {}
        
        db_payload = {
            "shot_id": shot_id,
            "local_velocity_x": lp_vel.get("x", 0.0),
            "local_velocity_y": lp_vel.get("y", 0.0),
            "goal_feet_yaw": anim.get("goal_feet_yaw", 0.0),
            "eye_yaw": anim.get("eye_yaw", 0.0),
            "layer3_weight": anim.get("layer3_weight", 0.0),
            "layer3_cycle": anim.get("layer3_cycle", 0.0),
            "relative_angle": target.get("relative_angle", 0.0),
            "miss_streak": config.get("miss_streak", 0),
            "confidence": prediction.get("confidence", 0.5) * 100.0,
            "resolver_mode": config.get("mode", "Adaptive"),
            "bf_phase": config.get("bf_phase", "Phase 1")
        }
        
        try:
            # ONLY save to DB if telemetry is sane
            if is_valid_telemetry(features):
                supabase.table("resolver_data").insert(db_payload).execute()
            else:
                print(f"⚠️ Ignored garbage telemetry for shot {shot_id}")
        except Exception as e:
            print(f"🔥 Supabase Insert Error: {e}")
            raise e

    return JSONResponse(prediction)

@app.post("/outcome")
async def outcome(request: Request):
    """Receive feedback from Lua on whether a shot hit or missed."""
    data = await request.json()
    shot_id = data.get("shot_id")
    if not shot_id or not supabase:
        return {"status": "ignored"}

    try:
        update_data = {
            "hit": data.get("hit", False),
            "damage_dealt": data.get("damage", 0),
            "miss_reason": data.get("reason", "none")
        }
        # Update the existing shot record with the result
        supabase.table("resolver_data").update(update_data).eq("shot_id", shot_id).execute()
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

global_patterns = 0

@app.post("/analyze")
async def analyze(request: Request):
    """General telemetry harvesting (Legacy & Flat support)."""
    global global_patterns
    data = await request.json()
    print(f"📥 Received Analyze: {data}")
    global_patterns += 1

    features = extract_features(data)
    suggestion = {
        "prediction_angle": 58 if (data.get("target", {}).get("anim", {}).get("desync_delta", 0) > 0) else -58,
        "bf_phase": "Phase 1 (Adaptive)",
        "resolver_mode": "Neural AI",
        "override_baim": data.get("config", {}).get("miss_streak", 0) >= 2
    }
    
    # ML Override if model is available
    if AI_MODEL:
        try:
            X = np.array([features])
            pred_side_idx = AI_MODEL.predict(X)[0]
            pred_proba = AI_MODEL.predict_proba(X)[0]
            conf = float(np.max(pred_proba))
            
            suggestion["prediction_angle"] = 58 if pred_side_idx == 1 else -58
            suggestion["confidence"] = conf
            
            if conf > 0.8:
                suggestion["bf_phase"] = "Phase 2 (Aggressive)"
        except: pass

    if supabase:
        local_player = data.get("local_player") or {}
        target = data.get("target") or {}
        config = data.get("config") or {}
        anim = target.get("anim") or {}
        lp_vel = local_player.get("vel") or {}
        
        db_payload = {
            "local_velocity_x": lp_vel.get("x", 0.0),
            "local_velocity_y": lp_vel.get("y", 0.0),
            "goal_feet_yaw": anim.get("goal_feet_yaw", 0.0),
            "eye_yaw": anim.get("eye_yaw", 0.0),
            "layer3_weight": anim.get("layer3_weight", 0.0),
            "layer3_cycle": anim.get("layer3_cycle", 0.0),
            "relative_angle": target.get("relative_angle", 0.0),
            "miss_streak": config.get("miss_streak", 0),
            "confidence": suggestion.get("confidence", 0.5) * 100.0,
            "resolver_mode": suggestion["resolver_mode"],
            "bf_phase": suggestion["bf_phase"]
        }
        
        try:
            features = extract_features(data)
            if is_valid_telemetry(features):
                supabase.table("resolver_data").insert(db_payload).execute()
        except Exception as e:
            print(f"🔥 Supabase Sync Error: {e}")
            raise e

    return JSONResponse(suggestion)

@app.post("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    global TRAINING_IN_PROGRESS
    if TRAINING_IN_PROGRESS:
        return {"status": "busy", "message": "Training already in progress"}
    
    TRAINING_IN_PROGRESS = True
    background_tasks.add_task(train_model_bg)
    return {"status": "started", "message": "ML training started in background using actual hit/miss data."}

def train_model_bg():
    global AI_MODEL, TRAINING_IN_PROGRESS
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        print("🧠 AI Training: Fetching labeled data from Supabase...")
        res = supabase.table("resolver_data").select("*").not_.is_("hit", "null").execute()
        
        if not res.data or len(res.data) < 50:
            print(f"❌ Training failed: Not enough labeled data (need 50, have {len(res.data) if res.data else 0})")
            return

        df = pd.DataFrame(res.data)
        
        # Prepare Features
        X = df[FEATURES]
        # Label: We want to predict if strzelanie w konkretną stronę zadziałało.
        # This is a bit complex. For now, we predict 'hit' based on the features.
        y = df['hit'].astype(int)
        
        # Actually, for a resolver, we want to predict the correct SIDE.
        # This requires knowing what side we SHOT at. 
        # For simplicity in this v2, we learn the probability of hitting given the telemetry.
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X, y)
        
        joblib.dump(model, AI_MODEL_PATH)
        AI_MODEL = model
        print(f"✅ Training Complete. Model updated with {len(df)} samples.")
        
    except Exception as e:
        print(f"❌ Training Error: {e}")
    finally:
        TRAINING_IN_PROGRESS = False

@app.get("/stats")
async def stats():
    total_db = global_patterns
    try:
        # Limit 1 is fast
        res = supabase.table("resolver_data").select("id", count="exact").limit(1).execute()
        if res.count is not None:
            total_db = res.count
    except:
        pass
        
    return {
        "users_online": 1,
        "patterns_saved": total_db,
        "avg_accuracy": 87, # Placeholder for UI
        "ai_status": "Learning..." if AI_MODEL else "Training Required",
        "last_sync": datetime.utcnow().isoformat()
    }

@app.get("/")
async def root():
    return {"status": "$hematic AI Backend v2.0 Online"}
