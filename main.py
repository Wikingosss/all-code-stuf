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

app = FastAPI(title="$hematic AI Backend", version="3.2.0")

# ============================================================
# DATABASE
# ============================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://qpymjauuxmkhgtrfetts.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_secret_3JF1ydBbUo3_5ZzRIU9r1g_P7TfWWGE")

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
# SUPABASE SCHEMA
# Exact columns verified from Supabase screenshot.
# strip_to_schema() filters every payload before any DB write.
# To add a column: ALTER TABLE resolver_data ADD COLUMN ... + add key here.
# ============================================================
DB_COLUMNS = {
    "id", "created_at",
    "miss_streak", "confidence", "resolver_mode", "bf_phase", "weapon",
    "velocity_x", "velocity_y", "distance",
    "choked_ticks", "duck_amount", "shot_id",
    "goal_feet_yaw", "eye_yaw", "body_yaw",
    "layer3_weight", "layer3_cycle", "relative_angle",
    "local_velocity_x", "local_velocity_y",
    "hit", "damage_dealt", "miss_reason",
    "speed_2d", "prediction_source", "steam_id",
}

def strip_to_schema(payload: dict) -> dict:
    """Remove any key not in DB_COLUMNS to prevent PGRST204 errors."""
    return {k: v for k, v in payload.items() if k in DB_COLUMNS}

# ============================================================
# IN-MEMORY STORE + PENDING SHOTS
# ============================================================
memory_store: list[dict] = []
MAX_MEMORY = 5000

# shot_uuid → {ts: float, payload: dict}
# Holds /predict payloads so /outcome can do a full fallback insert.
pending_shots: dict[str, dict] = {}
PENDING_TTL = 120  # seconds

# ============================================================
# SHOT-ID MAPPING  ← ROOT CAUSE FIX
#
# Lua uses e.id — an integer that resets to 1 on every game/script restart.
# After a server restart shots 1..N already exist in DB from a prior session,
# so /outcome for "shot=1" updates the wrong row (rows=0 because it was from
# a previous server instance that generated a different UUID).
#
# Solution:
#   /predict  → maps lua_id → new UUID, returns UUID to Lua
#   Lua stores UUID in ml_cache[target].shot_id
#   /outcome  → Lua sends UUID (not the integer)
#   Fallback: if Lua sends integer, we remap via shot_id_map
# ============================================================
shot_id_map: dict[str, str] = {}   # lua integer str → uuid str

def resolve_shot_id(lua_id: str) -> str:
    if lua_id not in shot_id_map:
        shot_id_map[lua_id] = str(uuid.uuid4())
    return shot_id_map[lua_id]

# ============================================================
# AI MODEL
# ============================================================
AI_MODEL_PATH        = "resolver_ai_v3.joblib"
AI_MODEL             = None
SCALER               = None
TRAINING_LOCK        = threading.Lock()
TRAINING_IN_PROGRESS = False

FEATURES = [
    "miss_streak", "choked_ticks",
    "velocity_x", "velocity_y", "speed_2d",
    "goal_feet_yaw", "eye_yaw", "body_yaw",
    "layer3_weight", "layer3_cycle",
    "relative_angle", "distance", "duck_amount",
    "local_velocity_x", "local_velocity_y",
]

def load_ai_model():
    global AI_MODEL, SCALER
    if os.path.exists(AI_MODEL_PATH):
        try:
            bundle   = joblib.load(AI_MODEL_PATH)
            AI_MODEL = bundle.get("model")
            SCALER   = bundle.get("scaler")
            print(f"✅ AI Model Loaded | has_scaler={SCALER is not None}")
        except Exception as e:
            print(f"⚠️  Failed to load AI model: {e}")

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_features(data: dict) -> dict:
    target       = data.get("target") or {}
    config       = data.get("config") or {}
    local_player = data.get("local_player") or {}
    anim         = target.get("anim") or {}
    vel          = target.get("vel") or {}
    lp_vel       = local_player.get("vel") or {}

    vx = float(vel.get("x", 0) or 0)
    vy = float(vel.get("y", 0) or 0)

    return {
        "miss_streak":      float(config.get("miss_streak", 0) or 0),
        "choked_ticks":     float(target.get("choke", 0) or 0),
        "velocity_x":       vx,
        "velocity_y":       vy,
        "speed_2d":         math.sqrt(vx * vx + vy * vy),
        "goal_feet_yaw":    float(anim.get("goal_feet_yaw", 0) or 0),
        "eye_yaw":          float(anim.get("eye_yaw", 0) or 0),
        "body_yaw":         float(anim.get("body_yaw", 0) or 0),
        "layer3_weight":    float(anim.get("layer3_weight", 0) or 0),
        "layer3_cycle":     float(anim.get("layer3_cycle", 0) or 0),
        "relative_angle":   float(target.get("relative_angle", 0) or 0),
        "distance":         float(config.get("distance", 0) or 0),
        "duck_amount":      float(target.get("duck", 0) or 0),
        "local_velocity_x": float(lp_vel.get("x", 0) or 0),
        "local_velocity_y": float(lp_vel.get("y", 0) or 0),
    }

def features_to_vector(feat: dict) -> list:
    return [feat[k] for k in FEATURES]

def is_valid_features(feat: dict) -> tuple[bool, str]:
    checks = {
        "goal_feet_yaw": 360.0, "eye_yaw": 360.0, "body_yaw": 360.0,
        "velocity_x": 10_000.0, "velocity_y": 10_000.0,
    }
    for name, limit in checks.items():
        val = feat[name]
        if math.isnan(val) or math.isinf(val):
            return False, f"{name}=NaN/Inf"
        if abs(val) > limit:
            return False, f"{name}={val:.1f} exceeds {limit}"
    return True, "ok"

# ============================================================
# PREDICTION
# ============================================================
player_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))

def heuristic_predict(feat: dict, steam_id: str = "") -> dict:
    gfy, eye_yaw = feat["goal_feet_yaw"], feat["eye_yaw"]
    choke, duck  = feat["choked_ticks"],  feat["duck_amount"]
    miss, rel    = feat["miss_streak"],   feat["relative_angle"]

    desync = gfy - eye_yaw
    hist   = list(player_history[steam_id]) if steam_id else []
    hl     = sum(1 for h in hist if h.get("hit") and h.get("side", 0) < 0)
    hr     = sum(1 for h in hist if h.get("hit") and h.get("side", 0) > 0)

    if hl > hr and (hl - hr) >= 2:
        side, conf = -58, 0.60 + min(0.15, (hl - hr) * 0.05)
    elif hr > hl and (hr - hl) >= 2:
        side, conf =  58, 0.60 + min(0.15, (hr - hl) * 0.05)
    elif abs(desync) > 1.0:
        side, conf = (58 if desync > 0 else -58), 0.52
    elif rel > 45:
        side, conf =  58, 0.50
    elif rel < -45:
        side, conf = -58, 0.50
    else:
        side, conf = (-58 if choke % 2 == 0 else 58), 0.48

    return {
        "predicted_side": side,
        "force_baim":     miss >= 3 or (miss >= 2 and duck > 0.5),
        "confidence":     round(conf, 4),
        "source":         "heuristic",
    }

def ml_predict(feat: dict) -> dict | None:
    if AI_MODEL is None:
        return None
    try:
        X = np.array([features_to_vector(feat)], dtype=np.float32)
        if SCALER is not None:
            X = SCALER.transform(X)
        idx   = AI_MODEL.predict(X)[0]
        proba = AI_MODEL.predict_proba(X)[0]
        conf  = float(np.max(proba))
        side  = 58 if idx == 1 else -58
        miss  = feat["miss_streak"]
        return {
            "predicted_side": side,
            "force_baim":     (conf < 0.55 and miss >= 2) or miss >= 4,
            "confidence":     round(conf, 4),
            "source":         "neural_network",
        }
    except Exception as e:
        print(f"⚠️  ML predict error: {e}")
        return None

def predict(data: dict, steam_id: str = "") -> tuple[dict, dict]:
    feat   = extract_features(data)
    result = ml_predict(feat) or heuristic_predict(feat, steam_id)
    return result, feat

# ============================================================
# CLEANUP
# ============================================================
def _cleanup_pending():
    now   = time.time()
    stale = [sid for sid, v in pending_shots.items() if now - v["ts"] > PENDING_TTL]
    for sid in stale:
        del pending_shots[sid]
        for lua_id, uid in list(shot_id_map.items()):
            if uid == sid:
                del shot_id_map[lua_id]
                break
    if stale:
        print(f"🧹 Cleaned {len(stale)} orphaned pending shots")

# ============================================================
# DATABASE HELPERS
# ============================================================
def _db_insert(payload: dict):
    safe = strip_to_schema(payload)
    if supabase:
        try:
            supabase.table("resolver_data").insert(safe).execute()
        except Exception as e:
            print(f"🔥 DB Insert Error: {e}")
    if len(memory_store) >= MAX_MEMORY:
        memory_store.pop(0)
    memory_store.append(safe)

def _db_update_outcome(shot_id: str, hit: bool, damage: int, reason: str,
                        fallback_payload: dict | None = None):
    update = {"hit": hit, "damage_dealt": damage, "miss_reason": str(reason)}

    if supabase:
        try:
            res     = supabase.table("resolver_data") \
                              .update(update).eq("shot_id", shot_id).execute()
            updated = len(res.data) if res.data else 0
            print(f"✅ Outcome: shot={shot_id} | {'HIT' if hit else 'MISS'} "
                  f"| dmg={damage} | rows_updated={updated}")

            if updated == 0:
                # Use full /predict payload if available, else minimal zeros
                record = dict(fallback_payload) if fallback_payload \
                         else _build_minimal_fallback(shot_id, hit, damage, reason)
                record.update(update)
                supabase.table("resolver_data").insert(strip_to_schema(record)).execute()
                print(f"↩️  Fallback insert shot={shot_id} "
                      f"({'full' if fallback_payload else 'minimal'})")
        except Exception as e:
            print(f"🔥 DB Update Error: {e}")

    for rec in memory_store:
        if rec.get("shot_id") == shot_id:
            rec.update(update)
            break

    pending_shots.pop(shot_id, None)

def _build_minimal_fallback(shot_id: str, hit: bool, damage: int, reason: str) -> dict:
    """Zero-filled record used only when /predict payload is completely unavailable."""
    return {
        "shot_id": shot_id, "hit": hit, "damage_dealt": damage, "miss_reason": str(reason),
        "velocity_x": 0.0, "velocity_y": 0.0, "local_velocity_x": 0.0, "local_velocity_y": 0.0,
        "goal_feet_yaw": 0.0, "eye_yaw": 0.0, "body_yaw": 0.0,
        "layer3_weight": 0.0, "layer3_cycle": 0.0, "relative_angle": 0.0,
        "choked_ticks": 0, "duck_amount": 0.0, "miss_streak": 0,
        "speed_2d": 0.0, "distance": 0.0, "confidence": 0.0,
        "resolver_mode": "unknown", "bf_phase": "unknown", "weapon": "Global",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

def _build_db_payload(shot_id: str, feat: dict, prediction: dict, data: dict) -> dict:
    config       = data.get("config") or {}
    local_player = data.get("local_player") or {}
    target       = data.get("target") or {}
    return {
        "shot_id":           shot_id,
        "hit":               None,
        "damage_dealt":      None,
        "miss_reason":       None,
        "miss_streak":       feat["miss_streak"],
        "choked_ticks":      feat["choked_ticks"],
        "velocity_x":        feat["velocity_x"],
        "velocity_y":        feat["velocity_y"],
        "speed_2d":          feat["speed_2d"],
        "local_velocity_x":  feat["local_velocity_x"],
        "local_velocity_y":  feat["local_velocity_y"],
        "goal_feet_yaw":     feat["goal_feet_yaw"],
        "eye_yaw":           feat["eye_yaw"],
        "body_yaw":          feat["body_yaw"],
        "layer3_weight":     feat["layer3_weight"],
        "layer3_cycle":      feat["layer3_cycle"],
        "relative_angle":    feat["relative_angle"],
        "distance":          feat["distance"],
        "duck_amount":       feat["duck_amount"],
        "confidence":        round(prediction.get("confidence", 0) * 100, 2),
        "prediction_source": prediction.get("source", "unknown"),
        "resolver_mode":     config.get("mode", "Adaptive"),
        "bf_phase":          config.get("bf_phase", "Phase 1"),
        "weapon":            local_player.get("weapon", "Global"),
        "steam_id":          str(target.get("steam_id") or ""),
        "created_at":        datetime.now(timezone.utc).isoformat(),
    }

# ============================================================
# TRAINING
# ============================================================
def train_model_bg():
    global AI_MODEL, SCALER, TRAINING_IN_PROGRESS
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score

        print("🧠 Training: fetching labeled data...")
        records = []
        if supabase:
            res = supabase.table("resolver_data") \
                          .select("*").not_.is_("hit", "null").execute()
            records = res.data or []
        records += [r for r in memory_store if r.get("hit") is not None]

        if len(records) < 30:
            print(f"❌ Not enough data (need 30, have {len(records)})")
            return

        df        = pd.DataFrame(records)
        available = [f for f in FEATURES if f in df.columns]
        df        = df.dropna(subset=available + ["hit"])

        if len(df) < 30:
            print(f"❌ After dropna only {len(df)} rows — aborting")
            return

        df["label"]    = df["hit"].astype(int)
        sample_weights = df["hit"].astype(int).apply(lambda h: 3.0 if h else 1.0)
        X = df[available].values.astype(np.float32)
        y = df["label"].values

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=3, random_state=42,
        )
        model.fit(X_sc, y, sample_weight=sample_weights.values)

        cv = cross_val_score(model, X_sc, y, cv=min(5, len(df) // 10 + 1), scoring="accuracy")
        print(f"✅ CV Accuracy: {cv.mean():.3f} ± {cv.std():.3f}")

        top5 = sorted(zip(available, model.feature_importances_), key=lambda x: -x[1])[:5]
        print("📊 Top: " + ", ".join(f"{n}={v:.3f}" for n, v in top5))

        joblib.dump({"model": model, "scaler": scaler, "features": available,
                     "trained_at": datetime.now(timezone.utc).isoformat(),
                     "n_samples": len(df)}, AI_MODEL_PATH)
        AI_MODEL = model
        SCALER   = scaler
        print(f"✅ Done | n={len(df)} | features={len(available)}")

    except Exception as e:
        print(f"❌ Training Error: {e}")
        import traceback; traceback.print_exc()
    finally:
        TRAINING_IN_PROGRESS = False

# ============================================================
# STARTUP
# ============================================================
@app.on_event("startup")
async def startup():
    load_ai_model()
    if supabase:
        try:
            supabase.table("resolver_data").select("shot_id").limit(1).execute()
            print("✅ DB schema probe OK")
        except Exception as e:
            print(f"⚠️  DB schema probe failed: {e}")

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "status":          "$hematic AI Backend v3.2 Online",
        "model":           "GBM v3" if AI_MODEL else "No model",
        "memory_records":  len(memory_store),
        "pending_shots":   len(pending_shots),
        "shot_id_map_size": len(shot_id_map),
    }


@app.post("/predict")
async def predict_endpoint(request: Request, background_tasks: BackgroundTasks):
    """
    Called by Lua on aim_fire BEFORE the shot is confirmed.

    KEY CHANGE v3.2:
      Lua sends its integer e.id as shot_id.
      We map it → stable UUID and return the UUID in the response.
      Lua must store this UUID and send it in /outcome.
      This breaks the session-collision problem where shot_id=1
      matched a row from a previous server session.
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    _cleanup_pending()

    lua_id   = str(data.get("shot_id") or "")
    shot_id  = resolve_shot_id(lua_id) if lua_id else str(uuid.uuid4())

    target   = data.get("target") or {}
    steam_id = str(target.get("steam_id") or "")

    feat          = extract_features(data)
    valid, reason = is_valid_features(feat)

    if not valid:
        print(f"⚠️  Garbage telemetry lua_id={lua_id}: {reason}")
        result            = heuristic_predict(feat, steam_id)
        result["shot_id"] = shot_id
        result["warning"] = f"garbage_telemetry: {reason}"
        return JSONResponse(result)

    result, feat = predict(data, steam_id)
    payload      = _build_db_payload(shot_id, feat, result, data)

    # Keep full payload so /outcome can use it for fallback insert
    pending_shots[shot_id] = {"ts": time.time(), "payload": payload}

    background_tasks.add_task(_db_insert, payload)

    player_history[steam_id].append({
        "side": result["predicted_side"], "hit": None, "shot_id": shot_id,
    })

    result["shot_id"] = shot_id
    return JSONResponse(result)


@app.post("/outcome")
async def outcome_endpoint(request: Request, background_tasks: BackgroundTasks):
    """
    Called by Lua on aim_hit / aim_miss.

    Lua should send the UUID returned by /predict.
    Fallback: if Lua sends an integer we remap via shot_id_map.
    If /predict was never called we do a full fallback insert using
    whatever payload we cached, or a minimal zero-filled record.
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    raw_id  = str(data.get("shot_id") or "")
    hit     = bool(data.get("hit", False))
    damage  = int(data.get("damage") or 0)
    reason  = data.get("reason", "none")

    if not raw_id:
        return JSONResponse({"status": "ignored", "reason": "no shot_id"})

    # Remap integer → UUID if Lua is still sending the old format
    shot_id = shot_id_map.get(raw_id, raw_id)

    # Update player history
    for steam_id, hist in player_history.items():
        for entry in hist:
            if entry.get("shot_id") == shot_id:
                entry["hit"] = hit
                break

    pending          = pending_shots.get(shot_id)
    fallback_payload = pending["payload"] if pending else None

    background_tasks.add_task(
        _db_update_outcome, shot_id, hit, damage, reason, fallback_payload
    )

    return JSONResponse({"status": "success", "hit": hit, "shot_id": shot_id})


@app.post("/analyze")
async def analyze_endpoint(request: Request, background_tasks: BackgroundTasks):
    """Legacy telemetry endpoint — /predict preferred."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    feat          = extract_features(data)
    valid, reason = is_valid_features(feat)
    result        = (ml_predict(feat) if (AI_MODEL and valid) else None) \
                    or heuristic_predict(feat)

    if valid:
        shot_id = str(uuid.uuid4())
        background_tasks.add_task(_db_insert, _build_db_payload(shot_id, feat, result, data))

    return JSONResponse({
        "prediction_angle": result["predicted_side"],
        "bf_phase":  "Phase 2 (Aggressive)" if result["confidence"] > 0.75 else "Phase 1 (Adaptive)",
        "resolver_mode": "Neural AI" if result["source"] == "neural_network" else "Adaptive",
        "confidence": result["confidence"],
    })


@app.post("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    global TRAINING_IN_PROGRESS
    if TRAINING_IN_PROGRESS:
        return JSONResponse({"status": "busy", "message": "Training already running"})

    total_db   = len(memory_store)
    labeled_db = sum(1 for r in memory_store if r.get("hit") is not None)

    if supabase:
        try:
            r = supabase.table("resolver_data").select("id", count="exact").limit(1).execute()
            total_db = max(total_db, r.count or 0)
        except: pass
        try:
            r = supabase.table("resolver_data").select("id", count="exact") \
                        .not_.is_("hit", "null").limit(1).execute()
            labeled_db = max(labeled_db, r.count or 0)
        except: pass

    TRAINING_IN_PROGRESS = True
    background_tasks.add_task(train_model_bg)

    return JSONResponse({
        "status": "started", "total_samples": total_db, "labeled_samples": labeled_db,
        "model_loaded": AI_MODEL is not None, "message": "GBM training started — ~30s.",
    })


@app.post("/debug_telemetry")
async def debug_telemetry(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})
    feat          = extract_features(data)
    valid, reason = is_valid_features(feat)
    return JSONResponse({"features": feat, "is_valid": valid,
                         "reason": reason, "vector": features_to_vector(feat)})


@app.get("/stats")
async def stats():
    total_db   = len(memory_store)
    labeled_db = sum(1 for r in memory_store if r.get("hit") is not None)
    hit_db     = sum(1 for r in memory_store if r.get("hit") is True)
    accuracy   = round((hit_db / labeled_db * 100) if labeled_db > 0 else 0, 1)

    if supabase:
        try:
            r = supabase.table("resolver_data").select("id", count="exact").limit(1).execute()
            total_db = r.count or total_db
        except: pass
        try:
            r = supabase.table("resolver_data").select("id", count="exact") \
                        .not_.is_("hit", "null").limit(1).execute()
            labeled_db = r.count or labeled_db
        except: pass

    return JSONResponse({
        "users_online":      1,
        "patterns_saved":    total_db,
        "resolver_records":  labeled_db,
        "ai_iterations":     labeled_db,
        "avg_confidence":    accuracy if accuracy > 0 else 87,
        "ai_status":         "Neural" if AI_MODEL else "Heuristic",
        "pending_shots":     len(pending_shots),
        "shot_id_map_size":  len(shot_id_map),
        "last_sync":         datetime.now(timezone.utc).isoformat(),
        "your_contribution": "Active" if memory_store else "Inactive",
        "model_version":     "GBM v3" if AI_MODEL else "none",
    })


@app.delete("/pending/{shot_id}")
async def cancel_pending(shot_id: str):
    removed = pending_shots.pop(shot_id, None)
    return JSONResponse({"removed": removed is not None, "shot_id": shot_id})
