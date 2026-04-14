"""
$hematic Reborn — AI Backend v4.0
Architecture:
  /predict  → real-time angle prediction (ML or heuristic fallback)
  /outcome  → shot result (hit/miss/damage) for supervised learning
  /analyze  → periodic telemetry for drift detection
  /train    → trigger model retrain
  /stats    → live DB stats for UI panel
  /logo     → PNG asset serve

ML Stack:
  - GradientBoostingClassifier for side prediction (L/R)
  - Heuristic fallback per-player history-weighted
  - Feature set: 22 fields including desync delta, AA type encoding,
    sim_time delta, armor, health, hitgroup context
"""

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
from fastapi.responses import JSONResponse, Response
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

app = FastAPI(title="$hematic AI Backend", version="4.0.0")

# ============================================================
# CONFIG
# ============================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://qpymjauuxmkhgtrfetts.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_secret_c_w2VtdiQUGeyqjuxk294A_RUmkrvco")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase Connected")
    except Exception as e:
        print(f"❌ Supabase Error: {e}")
else:
    print("⚠️  No Supabase — memory-only mode")

@app.exception_handler(Exception)
async def _exc(request: Request, exc: Exception):
    print(f"🔥 [{request.url.path}]: {exc}")
    return JSONResponse(status_code=500, content={"error": str(exc)})

# ============================================================
# SCHEMA — add columns here when extending Supabase table
# ============================================================
DB_COLUMNS = {
    "id", "created_at", "shot_id", "steam_id",
    "hit", "damage_dealt", "miss_reason", "hitgroup",
    # resolver state
    "miss_streak", "resolver_mode", "bf_phase", "weapon", "confidence", "prediction_source",
    # target kinematics
    "velocity_x", "velocity_y", "speed_2d", "distance",
    "choked_ticks", "duck_amount", "health", "armor",
    # FFI animstate
    "goal_feet_yaw", "eye_yaw", "body_yaw",
    "layer3_weight", "layer3_cycle", "relative_angle", "desync_delta",
    # local player kinematics
    "local_velocity_x", "local_velocity_y", "local_duck", "local_shots_fired",
    # AA classification
    "aa_type",
}

def _strip(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k in DB_COLUMNS}

# ============================================================
# STATE
# ============================================================
memory_store: list[dict] = []
MAX_MEMORY = 8000

pending_shots: dict[str, dict] = {}   # uuid → {ts, payload}
PENDING_TTL = 120

shot_id_map: dict[str, str] = {}      # lua int str → uuid str

# Per-steam_id hit/miss side history — core of heuristic
player_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=32))

# Cached CV accuracy from last training run
last_cv_accuracy: float = 0.0

def _resolve_sid(lua_id: str) -> str:
    """Map Lua integer shot id → stable UUID for this server session."""
    if lua_id not in shot_id_map:
        shot_id_map[lua_id] = str(uuid.uuid4())
    return shot_id_map[lua_id]

# ============================================================
# MODEL
# ============================================================
MODEL_PATH           = "resolver_ai_v4.joblib"
AI_MODEL             = None
SCALER               = None
TRAINING_LOCK        = threading.Lock()
TRAINING_IN_PROGRESS = False

# Features in exact order expected by the model
FEATURES = [
    # kinematics
    "velocity_x", "velocity_y", "speed_2d", "distance",
    "local_velocity_x", "local_velocity_y",
    # animstate
    "goal_feet_yaw", "eye_yaw", "body_yaw",
    "desync_delta",       # goal_feet_yaw - eye_yaw  (key feature)
    "layer3_weight", "layer3_cycle",
    "relative_angle",
    # entity state
    "choked_ticks", "duck_amount", "health", "armor",
    # context
    "miss_streak", "local_duck", "local_shots_fired",
]

def _load_model():
    global AI_MODEL, SCALER, last_cv_accuracy
    if os.path.exists(MODEL_PATH):
        try:
            bundle = joblib.load(MODEL_PATH)
            AI_MODEL = bundle["model"]
            SCALER   = bundle.get("scaler")
            last_cv_accuracy = bundle.get("cv_accuracy", 0.0)
            print(f"✅ Model loaded | CV={last_cv_accuracy:.3f} | samples={bundle.get('n_samples')}")
        except Exception as e:
            print(f"⚠️  Model load failed: {e}")

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def _safe_f(v, fallback: float = 0.0) -> float:
    """Cast to float, return fallback on None/NaN/Inf."""
    try:
        f = float(v)
        return fallback if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return fallback

def extract_features(data: dict) -> dict:
    tgt  = data.get("target") or {}
    cfg  = data.get("config") or {}
    lp   = data.get("local_player") or {}
    anim = tgt.get("anim") or {}
    vel  = tgt.get("vel") or {}
    lvel = lp.get("vel") or {}

    vx = _safe_f(vel.get("x"))
    vy = _safe_f(vel.get("y"))
    gfy   = _safe_f(anim.get("goal_feet_yaw"))
    eyaw  = _safe_f(anim.get("eye_yaw"))

    return {
        "velocity_x":        vx,
        "velocity_y":        vy,
        "speed_2d":          math.sqrt(vx*vx + vy*vy),
        "distance":          _safe_f(cfg.get("distance")),
        "local_velocity_x":  _safe_f(lvel.get("x")),
        "local_velocity_y":  _safe_f(lvel.get("y")),
        "goal_feet_yaw":     gfy,
        "eye_yaw":           eyaw,
        "body_yaw":          _safe_f(anim.get("body_yaw")),
        "desync_delta":      gfy - eyaw,
        "layer3_weight":     _safe_f(anim.get("layer3_weight")),
        "layer3_cycle":      _safe_f(anim.get("layer3_cycle")),
        "relative_angle":    _safe_f(tgt.get("relative_angle")),
        "choked_ticks":      _safe_f(tgt.get("choke")),
        "duck_amount":       _safe_f(tgt.get("duck")),
        "health":            _safe_f(tgt.get("health"), 100.0),
        "armor":             _safe_f(tgt.get("armor")),
        "miss_streak":       _safe_f(cfg.get("miss_streak")),
        "local_duck":        _safe_f(lp.get("duck_amount")),
        "local_shots_fired": _safe_f(lp.get("shots_fired")),
    }

def _feat_vector(feat: dict) -> list:
    return [feat[k] for k in FEATURES]

YAW_LIMIT  = 3600.0  # accept unnormalized yaw from Gamesense
VEL_LIMIT  = 12000.0

def _validate(feat: dict) -> tuple[bool, str]:
    checks = {
        "goal_feet_yaw": YAW_LIMIT, "eye_yaw": YAW_LIMIT, "body_yaw": YAW_LIMIT,
        "velocity_x": VEL_LIMIT, "velocity_y": VEL_LIMIT,
        "local_velocity_x": VEL_LIMIT, "local_velocity_y": VEL_LIMIT,
    }
    for name, lim in checks.items():
        v = feat[name]
        if math.isnan(v) or math.isinf(v):
            return False, f"{name}=NaN/Inf"
        if abs(v) > lim:
            return False, f"{name}={v:.1f}>lim"
    return True, "ok"

# ============================================================
# PREDICTION
# ============================================================
def _heuristic(feat: dict, steam_id: str = "") -> dict:
    """
    Heuristic prediction used when model is absent or confidence too low.
    Priority order:
      1. Per-player side bias from recent HIT history
      2. Desync delta sign (goal_feet_yaw - eye_yaw)
      3. Relative angle to local player
      4. Choke parity (last resort)
    """
    hist  = list(player_history[steam_id]) if steam_id else []
    n_hit = len([h for h in hist if h["hit"] is True])

    # Count hits per side from history
    hl = sum(1 for h in hist if h["hit"] and h["side"] < 0)
    hr = sum(1 for h in hist if h["hit"] and h["side"] > 0)

    desync = feat["desync_delta"]
    rel    = feat["relative_angle"]
    choke  = feat["choked_ticks"]
    miss   = feat["miss_streak"]
    duck   = feat["duck_amount"]

    if n_hit >= 3 and abs(hl - hr) >= 2:
        side = -58.0 if hl > hr else 58.0
        conf = min(0.82, 0.62 + abs(hl - hr) * 0.04)
    elif abs(desync) > 2.0:
        side = 58.0 if desync > 0 else -58.0
        conf = min(0.70, 0.50 + abs(desync) / 116.0 * 0.20)
    elif abs(rel) > 30:
        side = 58.0 if rel > 0 else -58.0
        conf = 0.50
    else:
        side = -58.0 if (int(choke) % 2 == 0) else 58.0
        conf = 0.46

    return {
        "predicted_side": side,
        "force_baim":     miss >= 4 or (miss >= 2 and duck > 0.6),
        "confidence":     round(conf, 4),
        "source":         "heuristic",
    }

def _ml_predict(feat: dict) -> dict | None:
    if AI_MODEL is None:
        return None
    try:
        X = np.array([_feat_vector(feat)], dtype=np.float32)
        if SCALER is not None:
            X = SCALER.transform(X)
        proba = AI_MODEL.predict_proba(X)[0]
        idx   = int(np.argmax(proba))
        conf  = float(proba[idx])
        # classes_: [0=left, 1=right]  →  side: 0→-58, 1→+58
        side  = 58.0 if AI_MODEL.classes_[idx] == 1 else -58.0
        miss  = feat["miss_streak"]
        return {
            "predicted_side": side,
            "force_baim":     (conf < 0.55 and miss >= 2) or miss >= 5,
            "confidence":     round(conf, 4),
            "source":         "gbm",
        }
    except Exception as e:
        print(f"⚠️  ML predict: {e}")
        return None

def _predict(data: dict, steam_id: str = "") -> tuple[dict, dict]:
    feat   = extract_features(data)
    result = _ml_predict(feat) or _heuristic(feat, steam_id)
    return result, feat

# ============================================================
# DATABASE
# ============================================================
def _db_insert(payload: dict):
    safe = _strip(payload)
    if supabase:
        try:
            supabase.table("resolver_data").insert(safe).execute()
        except Exception as e:
            print(f"🔥 DB insert: {e}")
    if len(memory_store) >= MAX_MEMORY:
        memory_store.pop(0)
    memory_store.append(safe)

def _db_outcome(shot_id: str, hit: bool, damage: int, reason: str,
                hitgroup: int, fallback: dict | None):
    upd = {"hit": hit, "damage_dealt": damage,
           "miss_reason": str(reason), "hitgroup": hitgroup}
    if supabase:
        try:
            res  = supabase.table("resolver_data").update(upd).eq("shot_id", shot_id).execute()
            rows = len(res.data) if res.data else 0
            print(f"{'✅ HIT' if hit else '❌ MISS'} | shot={shot_id[:8]} | dmg={damage} | rows={rows}")
            if rows == 0:
                rec = dict(fallback) if fallback else _fallback_rec(shot_id, hit, damage, reason)
                rec.update(upd)
                supabase.table("resolver_data").insert(_strip(rec)).execute()
                print(f"↩️  Fallback insert | shot={shot_id[:8]}")
        except Exception as e:
            print(f"🔥 DB outcome: {e}")
    for rec in memory_store:
        if rec.get("shot_id") == shot_id:
            rec.update(upd)
            break
    pending_shots.pop(shot_id, None)

def _fallback_rec(shot_id: str, hit: bool, damage: int, reason: str) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {"shot_id": shot_id, "hit": hit, "damage_dealt": damage,
            "miss_reason": str(reason), "created_at": now,
            **{k: 0.0 for k in FEATURES}}

def _build_payload(shot_id: str, feat: dict, pred: dict, data: dict) -> dict:
    cfg = data.get("config") or {}
    lp  = data.get("local_player") or {}
    tgt = data.get("target") or {}
    return {
        "shot_id":           shot_id,
        "hit":               None,
        "damage_dealt":      None,
        "miss_reason":       None,
        "hitgroup":          None,
        "steam_id":          str(tgt.get("steam_id") or ""),
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
        "desync_delta":      feat["desync_delta"],
        "layer3_weight":     feat["layer3_weight"],
        "layer3_cycle":      feat["layer3_cycle"],
        "relative_angle":    feat["relative_angle"],
        "distance":          feat["distance"],
        "duck_amount":       feat["duck_amount"],
        "health":            feat["health"],
        "armor":             feat["armor"],
        "local_duck":        feat["local_duck"],
        "local_shots_fired": feat["local_shots_fired"],
        "confidence":        round(pred.get("confidence", 0) * 100, 2),
        "prediction_source": pred.get("source", "unknown"),
        "resolver_mode":     cfg.get("mode", "Adaptive"),
        "bf_phase":          cfg.get("bf_phase", "Phase 1"),
        "weapon":            cfg.get("weapon") or lp.get("weapon") or "Global",
        "aa_type":           data.get("aa_type") or "",
        "created_at":        datetime.now(timezone.utc).isoformat(),
    }

def _cleanup_pending():
    now   = time.time()
    stale = [sid for sid, v in pending_shots.items() if now - v["ts"] > PENDING_TTL]
    for sid in stale:
        pending_shots.pop(sid, None)
        for lk, uv in list(shot_id_map.items()):
            if uv == sid:
                shot_id_map.pop(lk, None)
                break

# ============================================================
# TRAINING
# ============================================================
def _train_bg():
    global AI_MODEL, SCALER, TRAINING_IN_PROGRESS, last_cv_accuracy
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        print("🧠 Training start...")
        records = []
        if supabase:
            try:
                res = supabase.table("resolver_data").select("*") \
                              .not_.is_("hit", "null").execute()
                records = res.data or []
            except Exception as e:
                print(f"⚠️  DB fetch: {e}")
        records += [r for r in memory_store if r.get("hit") is not None]

        if len(records) < 50:
            print(f"❌ Need ≥50 labeled rows, have {len(records)}")
            return

        df  = pd.DataFrame(records)
        avail = [f for f in FEATURES if f in df.columns]
        df  = df.dropna(subset=avail + ["hit"])
        df  = df[df["hit"].notna()]

        if len(df) < 50:
            print(f"❌ After clean: {len(df)} rows — abort")
            return

        # Label: 1=right (+58 side hit), 0=left
        # We derive side from goal_feet_yaw sign if available, else from hit flag only
        if "desync_delta" in df.columns:
            df["label"] = ((df["hit"] == True) & (df["desync_delta"] >= 0)).astype(int)
        else:
            df["label"] = df["hit"].astype(int)

        X  = df[avail].values.astype(np.float32)
        y  = df["label"].values
        sw = np.where(df["hit"].values, 1.5, 1.0)  # weight hits slightly more

        scaler = StandardScaler()
        Xs     = scaler.fit_transform(X)

        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.04,
            subsample=0.8, min_samples_leaf=4,
            max_features="sqrt", random_state=42,
        )
        model.fit(Xs, y, sample_weight=sw)

        cv_n = min(5, max(2, len(df) // 20))
        cv   = cross_val_score(model, Xs, y,
                               cv=StratifiedKFold(n_splits=cv_n, shuffle=True, random_state=0),
                               scoring="accuracy")
        cv_mean = float(cv.mean())
        print(f"✅ CV {cv_mean:.3f} ± {cv.std():.3f} | n={len(df)} | feats={len(avail)}")

        top = sorted(zip(avail, model.feature_importances_), key=lambda x: -x[1])[:6]
        print("📊 " + " | ".join(f"{n}={v:.3f}" for n, v in top))

        last_cv_accuracy = cv_mean
        joblib.dump({
            "model": model, "scaler": scaler, "features": avail,
            "cv_accuracy": cv_mean, "n_samples": len(df),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }, MODEL_PATH)
        AI_MODEL = model
        SCALER   = scaler
        print("✅ Model saved")

    except Exception as e:
        print(f"❌ Training: {e}")
        import traceback; traceback.print_exc()
    finally:
        TRAINING_IN_PROGRESS = False

# ============================================================
# STARTUP
# ============================================================
@app.on_event("startup")
async def _startup():
    _load_model()
    if supabase:
        try:
            supabase.table("resolver_data").select("shot_id").limit(1).execute()
            print("✅ DB probe OK")
        except Exception as e:
            print(f"⚠️  DB probe: {e}")

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "status":   "$hematic AI Backend v4.0",
        "model":    f"GBM | CV={last_cv_accuracy:.3f}" if AI_MODEL else "heuristic-only",
        "records":  len(memory_store),
        "pending":  len(pending_shots),
    }


@app.post("/predict")
async def predict_ep(request: Request, bg: BackgroundTasks):
    """
    Called by Lua on aim_fire (before shot registers).
    Maps Lua int shot_id → stable UUID per session.
    Returns: predicted_side, force_baim, confidence, shot_id (UUID).
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "bad json"})

    _cleanup_pending()

    lua_id   = str(data.get("shot_id") or "")
    shot_id  = _resolve_sid(lua_id) if lua_id else str(uuid.uuid4())
    tgt      = data.get("target") or {}
    steam_id = str(tgt.get("steam_id") or "")

    feat           = extract_features(data)
    valid, inv_rsn = _validate(feat)

    if not valid:
        print(f"⚠️  Bad telemetry [{lua_id}]: {inv_rsn}")
        res = _heuristic(feat, steam_id)
        res["shot_id"] = shot_id
        res["warning"] = inv_rsn
        return JSONResponse(res)

    pred, feat = _predict(data, steam_id)
    payload    = _build_payload(shot_id, feat, pred, data)

    pending_shots[shot_id] = {"ts": time.time(), "payload": payload}
    bg.add_task(_db_insert, payload)

    player_history[steam_id].append({
        "side": pred["predicted_side"], "hit": None, "shot_id": shot_id,
    })

    pred["shot_id"] = shot_id
    return JSONResponse(pred)


@app.post("/outcome")
async def outcome_ep(request: Request, bg: BackgroundTasks):
    """
    Called by Lua on aim_hit / aim_miss.
    Sends UUID from /predict response.
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "bad json"})

    raw_id   = str(data.get("shot_id") or "")
    if not raw_id:
        return JSONResponse({"status": "ignored"})

    shot_id  = shot_id_map.get(raw_id, raw_id)
    hit      = bool(data.get("hit", False))
    damage   = int(data.get("damage") or 0)
    reason   = data.get("reason") or "none"
    hitgroup = int(data.get("hitgroup") or 0)

    for steam_id, hist in player_history.items():
        for entry in hist:
            if entry.get("shot_id") == shot_id:
                entry["hit"] = hit
                break

    pending  = pending_shots.get(shot_id)
    fallback = pending["payload"] if pending else None

    bg.add_task(_db_outcome, shot_id, hit, damage, reason, hitgroup, fallback)
    return JSONResponse({"status": "ok", "hit": hit, "shot_id": shot_id})


@app.post("/analyze")
async def analyze_ep(request: Request, bg: BackgroundTasks):
    """Periodic telemetry — used for drift detection, not shot-specific."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "bad json"})

    feat = extract_features(data)
    valid, _ = _validate(feat)
    pred = (_ml_predict(feat) if (AI_MODEL and valid) else None) or _heuristic(feat)

    if valid:
        shot_id = str(uuid.uuid4())
        bg.add_task(_db_insert, _build_payload(shot_id, feat, pred, data))

    conf = pred["confidence"]
    return JSONResponse({
        "bf_phase":       "Phase 2 (Aggressive)" if conf > 0.72 else "Phase 1 (Adaptive)",
        "resolver_mode":  "Neural AI" if pred["source"] == "gbm" else "Adaptive",
        "override_baim":  pred["force_baim"],
        "confidence":     conf,
    })


@app.post("/train")
async def train_ep(bg: BackgroundTasks):
    global TRAINING_IN_PROGRESS
    if TRAINING_IN_PROGRESS:
        return JSONResponse({"status": "busy"})
    total   = len(memory_store)
    labeled = sum(1 for r in memory_store if r.get("hit") is not None)
    if supabase:
        try:
            r = supabase.table("resolver_data").select("id", count="exact").limit(1).execute()
            total = max(total, r.count or 0)
        except: pass
        try:
            r = supabase.table("resolver_data").select("id", count="exact") \
                        .not_.is_("hit", "null").limit(1).execute()
            labeled = max(labeled, r.count or 0)
        except: pass
    TRAINING_IN_PROGRESS = True
    bg.add_task(_train_bg)
    return JSONResponse({
        "status": "started", "total": total, "labeled": labeled,
        "has_model": AI_MODEL is not None,
    })


@app.get("/stats")
async def stats_ep():
    total   = len(memory_store)
    labeled = sum(1 for r in memory_store if r.get("hit") is not None)
    hits    = sum(1 for r in memory_store if r.get("hit") is True)
    mem_acc = round((hits / labeled * 100) if labeled > 0 else 0.0, 1)

    if supabase:
        try:
            r = supabase.table("resolver_data").select("id", count="exact").limit(1).execute()
            total = max(total, r.count or 0)
        except: pass
        try:
            r = supabase.table("resolver_data").select("id", count="exact") \
                        .not_.is_("hit", "null").limit(1).execute()
            labeled = max(labeled, r.count or 0)
        except: pass

    real_conf = round(last_cv_accuracy * 100, 1) if AI_MODEL and last_cv_accuracy > 0 \
                else mem_acc

    return JSONResponse({
        "users_online":      1,
        "patterns_saved":    total,
        "resolver_records":  labeled,
        "ai_iterations":     labeled,
        "avg_confidence":    real_conf,
        "ai_status":         "GBM" if AI_MODEL else "Heuristic",
        "last_sync":         datetime.now(timezone.utc).isoformat(),
        "your_contribution": "Active" if memory_store else "Inactive",
        "model_version":     f"GBM v4 | CV={last_cv_accuracy:.3f}" if AI_MODEL else "none",
    })


@app.post("/debug")
async def debug_ep(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "bad json"})
    feat  = extract_features(data)
    valid, reason = _validate(feat)
    return JSONResponse({"features": feat, "valid": valid, "reason": reason})


@app.delete("/pending/{shot_id}")
async def cancel_pending(shot_id: str):
    removed = pending_shots.pop(shot_id, None)
    return JSONResponse({"removed": removed is not None})
