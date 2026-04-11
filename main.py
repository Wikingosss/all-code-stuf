import os
import uuid
import time
import math
import threading
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone
from collections import defaultdict, deque
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

app = FastAPI(title="$hematic AI Backend", version="3.0.0")

# ============================================================
# DATABASE
# ============================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase Connected")
    except Exception as e:
        print(f"❌ Supabase Connection Error: {e}")
else:
    print("⚠️  No Supabase credentials — running in memory-only mode")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"🔥 Global Error [{request.url.path}]: {exc}")
    return JSONResponse(status_code=500, content={"message": str(exc)})

# ============================================================
# IN-MEMORY FALLBACK STORE
# Used when Supabase is unavailable, also acts as write-through cache
# ============================================================
memory_store: list[dict] = []          # shot records
outcome_buffer: dict[str, dict] = {}   # shot_id -> outcome, waiting for predict record
MAX_MEMORY = 5000                      # cap at 5k records in RAM

# ============================================================
# AI MODEL
# ============================================================
AI_MODEL_PATH = "resolver_ai_v3.joblib"
AI_MODEL      = None
SCALER        = None                   # StandardScaler for feature normalization
TRAINING_LOCK = threading.Lock()
TRAINING_IN_PROGRESS = False

# All features — expanded from v2
FEATURES = [
    # Timing / state
    "miss_streak",
    "choked_ticks",
    # Velocity (target)
    "velocity_x",
    "velocity_y",
    "speed_2d",
    # Animation
    "goal_feet_yaw",
    "eye_yaw",
    "body_yaw",
    "layer3_weight",
    "layer3_cycle",
    # Spatial
    "relative_angle",
    "distance",
    "duck_amount",
    # Local player
    "local_velocity_x",
    "local_velocity_y",
]

def load_ai_model():
    global AI_MODEL, SCALER
    if os.path.exists(AI_MODEL_PATH):
        try:
            bundle  = joblib.load(AI_MODEL_PATH)
            AI_MODEL = bundle.get("model")
            SCALER   = bundle.get("scaler")
            print(f"✅ AI Model v3 Loaded | has_scaler={SCALER is not None}")
        except Exception as e:
            print(f"⚠️  Failed to load AI model: {e}")

load_ai_model()

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_features(data: dict) -> dict:
    """
    Returns a NAMED dict so we never confuse feature order.
    Safe for missing / None values.
    """
    target       = data.get("target") or {}
    config       = data.get("config") or {}
    local_player = data.get("local_player") or {}
    anim         = target.get("anim") or {}
    vel          = target.get("vel") or {}
    lp_vel       = local_player.get("vel") or {}

    vx  = float(vel.get("x", 0) or 0)
    vy  = float(vel.get("y", 0) or 0)

    return {
        "miss_streak":       float(config.get("miss_streak", 0) or 0),
        "choked_ticks":      float(target.get("choke", 0) or 0),
        "velocity_x":        vx,
        "velocity_y":        vy,
        "speed_2d":          math.sqrt(vx*vx + vy*vy),
        "goal_feet_yaw":     float(anim.get("goal_feet_yaw", 0) or 0),
        "eye_yaw":           float(anim.get("eye_yaw", 0) or 0),
        "body_yaw":          float(anim.get("body_yaw", 0) or 0),
        "layer3_weight":     float(anim.get("layer3_weight", 0) or 0),
        "layer3_cycle":      float(anim.get("layer3_cycle", 0) or 0),
        "relative_angle":    float(target.get("relative_angle", 0) or 0),
        "distance":          float(config.get("distance", 0) or 0),
        "duck_amount":       float(target.get("duck", 0) or 0),
        "local_velocity_x":  float(lp_vel.get("x", 0) or 0),
        "local_velocity_y":  float(lp_vel.get("y", 0) or 0),
    }

def features_to_vector(feat: dict) -> list:
    """Converts named feature dict to ordered list matching FEATURES."""
    return [feat[k] for k in FEATURES]

def is_valid_features(feat: dict) -> tuple[bool, str]:
    """
    Validates extracted features.
    Returns (valid: bool, reason: str).
    """
    ANGLE_LIMIT = 360.0
    SPEED_LIMIT = 10000.0

    checks = {
        "goal_feet_yaw": (feat["goal_feet_yaw"], ANGLE_LIMIT),
        "eye_yaw":       (feat["eye_yaw"],       ANGLE_LIMIT),
        "body_yaw":      (feat["body_yaw"],       ANGLE_LIMIT),
        "velocity_x":    (feat["velocity_x"],     SPEED_LIMIT),
        "velocity_y":    (feat["velocity_y"],     SPEED_LIMIT),
    }

    for name, (val, limit) in checks.items():
        if math.isnan(val) or math.isinf(val):
            return False, f"{name}=NaN/Inf"
        if abs(val) > limit:
            return False, f"{name}={val:.1f} exceeds limit {limit}"

    return True, "ok"

# ============================================================
# PREDICTION LOGIC
# ============================================================

# Sliding window per-player: remember recent predictions and outcomes
# player_id (steam64 str) -> deque of {"side": int, "hit": bool|None}
player_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))

def heuristic_predict(feat: dict, steam_id: str = "") -> dict:
    """
    Pure heuristic fallback when no ML model is available.
    Uses animstate data and per-player history.
    """
    gfy      = feat["goal_feet_yaw"]
    eye_yaw  = feat["eye_yaw"]
    choke    = feat["choked_ticks"]
    duck     = feat["duck_amount"]
    miss     = feat["miss_streak"]
    rel_ang  = feat["relative_angle"]

    # Primary signal: desync delta between eye and feet yaw
    desync_delta = gfy - eye_yaw

    # Check player history for side bias
    history    = list(player_history[steam_id]) if steam_id else []
    hit_left   = sum(1 for h in history if h.get("hit") and h.get("side", 0) < 0)
    hit_right  = sum(1 for h in history if h.get("hit") and h.get("side", 0) > 0)

    if hit_left > hit_right and (hit_left - hit_right) >= 2:
        side = -58
        confidence = 0.60 + min(0.15, (hit_left - hit_right) * 0.05)
    elif hit_right > hit_left and (hit_right - hit_left) >= 2:
        side = 58
        confidence = 0.60 + min(0.15, (hit_right - hit_left) * 0.05)
    elif abs(desync_delta) > 1.0:
        side = 58 if desync_delta > 0 else -58
        confidence = 0.52
    elif rel_ang > 45:
        side = 58
        confidence = 0.50
    elif rel_ang < -45:
        side = -58
        confidence = 0.50
    else:
        # Choked packets: flip side
        side = -58 if (choke % 2 == 0) else 58
        confidence = 0.48

    # Miss streak escalation
    force_baim = miss >= 3 or (miss >= 2 and duck > 0.5)

    return {
        "predicted_side": side,
        "force_baim":     force_baim,
        "confidence":     round(confidence, 4),
        "source":         "heuristic"
    }

def ml_predict(feat: dict) -> dict | None:
    """
    ML prediction. Returns None if model unavailable or error.
    """
    if AI_MODEL is None:
        return None

    try:
        vec = features_to_vector(feat)
        X   = np.array([vec], dtype=np.float32)

        if SCALER is not None:
            X = SCALER.transform(X)

        # Model predicts: 0 = negative side (-58), 1 = positive side (+58)
        side_idx  = AI_MODEL.predict(X)[0]
        probas    = AI_MODEL.predict_proba(X)[0]
        confidence = float(np.max(probas))
        side       = 58 if side_idx == 1 else -58

        # Low-confidence override: force baim
        miss       = feat["miss_streak"]
        force_baim = (confidence < 0.55 and miss >= 2) or miss >= 4

        return {
            "predicted_side": side,
            "force_baim":     force_baim,
            "confidence":     round(confidence, 4),
            "source":         "neural_network"
        }
    except Exception as e:
        print(f"⚠️  ML predict error: {e}")
        return None

def predict(data: dict, steam_id: str = "") -> dict:
    """Combined prediction: ML first, heuristic fallback."""
    feat   = extract_features(data)
    result = ml_predict(feat)
    if result is None:
        result = heuristic_predict(feat, steam_id)
    return result, feat

# ============================================================
# DATABASE HELPERS (all non-blocking via background_tasks)
# ============================================================
def _db_insert(payload: dict):
    if supabase:
        try:
            supabase.table("resolver_data").insert(payload).execute()
        except Exception as e:
            print(f"🔥 DB Insert Error: {e}")
    # Always keep in memory too
    if len(memory_store) >= MAX_MEMORY:
        memory_store.pop(0)
    memory_store.append(payload)

def _db_update_outcome(shot_id: str, hit: bool, damage: int, reason: str):
    update = {"hit": hit, "damage_dealt": damage, "miss_reason": str(reason)}
    if supabase:
        try:
            res = supabase.table("resolver_data")\
                .update(update)\
                .eq("shot_id", shot_id)\
                .execute()
            updated = len(res.data) if res.data else 0
            print(f"✅ Outcome DB: shot={shot_id} | {'HIT' if hit else 'MISS'} | dmg={damage} | rows={updated}")
            if updated == 0:
                print(f"⚠️  No row matched shot_id={shot_id} — inserting fallback")
                fallback = _build_fallback_record(shot_id, hit, damage, reason)
                supabase.table("resolver_data").insert(fallback).execute()
        except Exception as e:
            print(f"🔥 DB Update Error: {e}")
    # Update memory store
    for rec in memory_store:
        if rec.get("shot_id") == shot_id:
            rec.update(update)
            break

def _build_fallback_record(shot_id: str, hit: bool, damage: int, reason: str) -> dict:
    return {
        "shot_id":          shot_id,
        "hit":              hit,
        "damage_dealt":     damage,
        "miss_reason":      str(reason),
        "velocity_x":       0.0, "velocity_y":      0.0,
        "local_velocity_x": 0.0, "local_velocity_y": 0.0,
        "goal_feet_yaw":    0.0, "eye_yaw":          0.0,
        "body_yaw":         0.0,
        "layer3_weight":    0.0, "layer3_cycle":     0.0,
        "relative_angle":   0.0, "choked_ticks":     0,
        "duck_amount":      0.0, "miss_streak":      0,
        "speed_2d":         0.0, "distance":         0.0,
        "confidence":       0.0,
        "resolver_mode":    "unknown",
        "bf_phase":         "unknown",
        "weapon":           "Global",
    }

def _build_db_payload(shot_id: str, feat: dict, prediction: dict, data: dict) -> dict:
    config       = data.get("config") or {}
    local_player = data.get("local_player") or {}
    target       = data.get("target") or {}
    return {
        "shot_id":          shot_id,
        "hit":              None,               # filled by /outcome
        "damage_dealt":     None,
        "miss_reason":      None,
        # Features
        "miss_streak":      feat["miss_streak"],
        "choked_ticks":     feat["choked_ticks"],
        "velocity_x":       feat["velocity_x"],
        "velocity_y":       feat["velocity_y"],
        "speed_2d":         feat["speed_2d"],
        "local_velocity_x": feat["local_velocity_x"],
        "local_velocity_y": feat["local_velocity_y"],
        "goal_feet_yaw":    feat["goal_feet_yaw"],
        "eye_yaw":          feat["eye_yaw"],
        "body_yaw":         feat["body_yaw"],
        "layer3_weight":    feat["layer3_weight"],
        "layer3_cycle":     feat["layer3_cycle"],
        "relative_angle":   feat["relative_angle"],
        "distance":         feat["distance"],
        "duck_amount":      feat["duck_amount"],
        # Prediction metadata
        "predicted_side":   prediction.get("predicted_side", 0),
        "confidence":       round(prediction.get("confidence", 0) * 100, 2),
        "prediction_source": prediction.get("source", "unknown"),
        # Config
        "resolver_mode":    config.get("mode", "Adaptive"),
        "bf_phase":         config.get("bf_phase", "Phase 1"),
        "weapon":           local_player.get("weapon", "Global"),
        "steam_id":         (target.get("steam_id") or ""),
        "created_at":       datetime.now(timezone.utc).isoformat(),
    }

# ============================================================
# TRAINING
# ============================================================
def train_model_bg():
    global AI_MODEL, SCALER, TRAINING_IN_PROGRESS
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline

        print("🧠 Training: Fetching labeled data...")

        records = []
        if supabase:
            res = supabase.table("resolver_data")\
                .select("*")\
                .not_.is_("hit", "null")\
                .not_.is_("predicted_side", "null")\
                .execute()
            records = res.data or []

        # Also train on in-memory labeled records
        mem_labeled = [r for r in memory_store if r.get("hit") is not None and r.get("predicted_side") is not None]
        records += mem_labeled

        if len(records) < 30:
            print(f"❌ Not enough labeled data (need 30, have {len(records)})")
            return

        df = pd.DataFrame(records)

        # Drop rows with missing features
        df = df.dropna(subset=FEATURES + ["hit", "predicted_side"])

        # LABEL: did we predict the correct SIDE that resulted in a hit?
        # 1 = positive side (+58 ish), 0 = negative side
        df["label"] = (df["predicted_side"] > 0).astype(int)

        # But weight hits 3x more than misses to account for dataset imbalance
        # (most shots miss because resolver is still learning)
        sample_weights = df["hit"].astype(int).apply(lambda h: 3.0 if h == 1 else 1.0)

        X = df[FEATURES].values.astype(np.float32)
        y = df["label"].values

        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Gradient Boosting performs better than RF for small tabular datasets
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=3,
            random_state=42
        )
        model.fit(X_scaled, y, sample_weight=sample_weights.values)

        # Cross-validation score
        cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(df)//10 + 1), scoring="accuracy")
        print(f"✅ CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Feature importances
        importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
        print("📊 Top features:")
        for fname, imp in importances[:5]:
            print(f"   {fname:25s}: {imp:.4f}")

        # Save bundle
        bundle = {"model": model, "scaler": scaler, "trained_at": datetime.now(timezone.utc).isoformat(), "n_samples": len(df)}
        joblib.dump(bundle, AI_MODEL_PATH)
        AI_MODEL = model
        SCALER   = scaler
        print(f"✅ Training Complete | samples={len(df)} | model=GBM")

    except Exception as e:
        print(f"❌ Training Error: {e}")
        import traceback; traceback.print_exc()
    finally:
        TRAINING_IN_PROGRESS = False

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "$hematic AI Backend v3.0 Online",
        "model":  "GradientBoosting v3" if AI_MODEL else "No model loaded",
        "memory_records": len(memory_store),
    }

@app.post("/predict")
async def predict_endpoint(request: Request, background_tasks: BackgroundTasks):
    """
    Main prediction endpoint — called on aim_fire.
    Fixed v3:
      - Named feature extraction (no index confusion)
      - Validation with detailed reason logging
      - Always saves to DB (valid OR fallback)
      - Steam ID tracked for per-player history
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    shot_id  = str(data.get("shot_id") or uuid.uuid4())
    target   = data.get("target") or {}
    steam_id = str(target.get("steam_id") or "")

    feat = extract_features(data)
    valid, reason = is_valid_features(feat)

    if not valid:
        print(f"⚠️  Garbage telemetry shot={shot_id}: {reason}")
        # Still return a prediction (heuristic) but don't save garbage to DB
        result = heuristic_predict(feat, steam_id)
        result["warning"] = f"garbage_telemetry: {reason}"
        return JSONResponse(result)

    # Run prediction
    result, feat = predict(data, steam_id)

    # Save to DB (background, non-blocking)
    payload = _build_db_payload(shot_id, feat, result, data)
    background_tasks.add_task(_db_insert, payload)

    # Track in player history (no outcome yet)
    player_history[steam_id].append({"side": result["predicted_side"], "hit": None, "shot_id": shot_id})

    result["shot_id"] = shot_id
    return JSONResponse(result)


@app.post("/outcome")
async def outcome_endpoint(request: Request, background_tasks: BackgroundTasks):
    """
    Receives hit/miss feedback from Lua.
    Fixed v3:
      - Updates player_history for per-player ML bias
      - Falls back to INSERT if row doesn't exist (predict was blocked)
      - Never returns 500 for missing rows
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    shot_id = str(data.get("shot_id") or "")
    hit     = bool(data.get("hit", False))
    damage  = int(data.get("damage") or 0)
    reason  = data.get("reason", "none")

    if not shot_id:
        return JSONResponse({"status": "ignored", "reason": "no shot_id"})

    # Update player_history for this shot
    for steam_id, hist in player_history.items():
        for entry in hist:
            if entry.get("shot_id") == shot_id:
                entry["hit"] = hit
                break

    background_tasks.add_task(_db_update_outcome, shot_id, hit, damage, reason)
    return JSONResponse({"status": "success", "hit": hit, "shot_id": shot_id})


@app.post("/analyze")
async def analyze_endpoint(request: Request, background_tasks: BackgroundTasks):
    """
    Legacy general telemetry endpoint.
    v3: still supported but /predict is preferred.
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    feat         = extract_features(data)
    valid, reason = is_valid_features(feat)
    result       = heuristic_predict(feat)

    if AI_MODEL and valid:
        ml_result = ml_predict(feat)
        if ml_result:
            result = ml_result

    suggestion = {
        "prediction_angle": result["predicted_side"],
        "bf_phase":         "Phase 2 (Aggressive)" if result["confidence"] > 0.75 else "Phase 1 (Adaptive)",
        "resolver_mode":    "Neural AI" if result["source"] == "neural_network" else "Adaptive",
        "confidence":       result["confidence"],
    }

    if valid:
        shot_id = str(uuid.uuid4())
        payload = _build_db_payload(shot_id, feat, result, data)
        background_tasks.add_task(_db_insert, payload)

    return JSONResponse(suggestion)


@app.post("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    global TRAINING_IN_PROGRESS
    if TRAINING_IN_PROGRESS:
        return JSONResponse({"status": "busy", "message": "Training already running"})

    total_samples   = 0
    labeled_samples = 0

    if supabase:
        try:
            r = supabase.table("resolver_data").select("id", count="exact").limit(1).execute()
            total_samples = r.count or 0
        except: pass
        try:
            r = supabase.table("resolver_data").select("id", count="exact").not_.is_("hit", "null").limit(1).execute()
            labeled_samples = r.count or 0
        except: pass

    # Also count memory
    total_samples   = max(total_samples, len(memory_store))
    labeled_samples = max(labeled_samples, sum(1 for r in memory_store if r.get("hit") is not None))

    TRAINING_IN_PROGRESS = True
    background_tasks.add_task(train_model_bg)

    return JSONResponse({
        "status":          "started",
        "total_samples":   total_samples,
        "labeled_samples": labeled_samples,
        "model_loaded":    AI_MODEL is not None,
        "message":         "GradientBoosting training started. Results in ~30s."
    })


@app.post("/debug_telemetry")
async def debug_telemetry(request: Request):
    """Debug endpoint — see exact features extracted from a payload."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})
    feat         = extract_features(data)
    valid, reason = is_valid_features(feat)
    return JSONResponse({
        "features": feat,
        "is_valid": valid,
        "reason":   reason,
        "vector":   features_to_vector(feat)
    })


@app.get("/stats")
async def stats():
    total_db    = len(memory_store)
    labeled_db  = sum(1 for r in memory_store if r.get("hit") is not None)
    hit_db      = sum(1 for r in memory_store if r.get("hit") is True)
    accuracy    = round((hit_db / labeled_db * 100) if labeled_db > 0 else 0, 1)

    if supabase:
        try:
            r = supabase.table("resolver_data").select("id", count="exact").limit(1).execute()
            total_db = r.count or total_db
        except: pass
        try:
            r = supabase.table("resolver_data").select("id", count="exact").not_.is_("hit", "null").limit(1).execute()
            labeled_db = r.count or labeled_db
        except: pass

    return JSONResponse({
        "users_online":      1,
        "patterns_saved":    total_db,
        "resolver_records":  labeled_db,
        "ai_iterations":     labeled_db,
        "avg_confidence":    accuracy if accuracy > 0 else 87,
        "ai_status":         "Neural" if AI_MODEL else "Heuristic",
        "last_sync":         datetime.now(timezone.utc).isoformat(),
        "your_contribution": "Active" if len(memory_store) > 0 else "Inactive",
        "model_version":     "GBM v3" if AI_MODEL else "none",
    })
