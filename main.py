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
from fastapi.responses import JSONResponse, Response, RedirectResponse
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

app = FastAPI(title="$hematic AI Backend", version="4.0.0")

# ============================================================
# CONFIG
# ============================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://qpymjauuxmkhgtrfetts.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_secret_c_w2VtdiQUGeyqjuxk294A_RUmkrvco")
AUTO_TRAIN_ENABLED = os.environ.get("AUTO_TRAIN_ENABLED", "1") == "1"
AUTO_TRAIN_EVERY = max(10, int(os.environ.get("AUTO_TRAIN_EVERY", "120")))
AUTO_TRAIN_MIN_LABELED = max(30, int(os.environ.get("AUTO_TRAIN_MIN_LABELED", "120")))
AUTO_TRAIN_COOLDOWN_SEC = max(60, int(os.environ.get("AUTO_TRAIN_COOLDOWN_SEC", "300")))
PERIODIC_TRAIN_ENABLED = os.environ.get("PERIODIC_TRAIN_ENABLED", "1") == "1"
PERIODIC_TRAIN_INTERVAL_SEC = max(300, int(os.environ.get("PERIODIC_TRAIN_INTERVAL_SEC", "1800")))
TRAIN_MIN_LABELED = max(20, int(os.environ.get("TRAIN_MIN_LABELED", "50")))
TRAIN_MAX_ROWS = max(2000, int(os.environ.get("TRAIN_MAX_ROWS", "20000")))
TRAIN_HARD_MISS_ROWS = max(200, int(os.environ.get("TRAIN_HARD_MISS_ROWS", "2500")))
TRAIN_HARD_MISS_BOOST = max(1.0, float(os.environ.get("TRAIN_HARD_MISS_BOOST", "1.6")))
ANALYZE_DB_WRITE = os.environ.get("ANALYZE_DB_WRITE", "0") == "1"
ANALYZE_DB_EVERY_N = max(1, int(os.environ.get("ANALYZE_DB_EVERY_N", "20")))
LOGO_GITHUB_RAW_BASE = os.environ.get("LOGO_GITHUB_RAW_BASE", "https://raw.githubusercontent.com/Wikingosss/lua-/main").rstrip("/")

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
    "id", "created_at", "shot_id", "lua_shot_id", "steam_id", "discord_id",
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

pending_shots: dict[str, dict] = {}    # uuid → {ts, payload}
PENDING_TTL = 120

shot_id_map: dict[str, str] = {}       # lua int str → uuid str

# Outcomes that arrived before /predict inserted the row
pending_outcomes: dict[str, dict] = {} # lua_id → {upd, ts}

# Per-steam_id hit/miss side history — core of heuristic
player_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=32))

presence_store: dict[str, dict] = {}
PRESENCE_TTL_SEC = max(10, int(os.environ.get("PRESENCE_TTL_SEC", "45")))

# Cached CV accuracy from last training run
last_cv_accuracy: float = 0.0
last_train_at_ts: float = 0.0
last_train_labeled: int = 0
last_training_trigger: str = "startup"

training_status = {
    "in_progress": False,
    "last_trigger": "startup",
    "last_started_at": None,
    "last_finished_at": None,
    "last_error": None,
    "last_cv": 0.0,
    "last_samples": 0,
}

prediction_metrics = {
    "total": 0,
    "ml_used": 0,
    "heuristic_used": 0,
    "invalid_telemetry": 0,
}
analyze_counter = 0

HARD_MISS_REASONS = {
    "resolver", "?", "unknown", "prediction error", "prediction_error", "occlusion"
}

def _resolve_sid(lua_id: str) -> str:
    """Map Lua integer shot id → stable UUID for this server session."""
    if lua_id not in shot_id_map:
        shot_id_map[lua_id] = str(uuid.uuid4())
    return shot_id_map[lua_id]


def _is_uuid(v: str) -> bool:
    try:
        uuid.UUID(str(v))
        return True
    except Exception:
        return False


def _presence_cleanup(now_ts: float | None = None) -> None:
    now = now_ts if isinstance(now_ts, (int, float)) else time.time()
    expired = []
    for sid, row in presence_store.items():
        ts = float(row.get("last_seen") or 0.0)
        if (now - ts) > PRESENCE_TTL_SEC:
            expired.append(sid)
    for sid in expired:
        presence_store.pop(sid, None)


def _count_labeled_records() -> int:
    local_labeled = sum(1 for r in memory_store if r.get("hit") is not None)
    if not supabase:
        return local_labeled
    try:
        r = supabase.table("resolver_data").select("id", count="exact") \
                    .not_.is_("hit", "null").limit(1).execute()
        return max(local_labeled, r.count or 0)
    except Exception:
        return local_labeled

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
        miss = feat["miss_streak"]
        return {
            "hit_probability": conf,
            "force_baim":     (conf < 0.55 and miss >= 2) or miss >= 5,
            "confidence":     round(conf, 4),
            "source":         "gbm",
        }
    except Exception as e:
        print(f"⚠️  ML predict: {e}")
        return None

def _predict(data: dict, steam_id: str = "") -> tuple[dict, dict]:
    feat = extract_features(data)
    h = _heuristic(feat, steam_id)
    m = _ml_predict(feat)

    if m:
        prediction_metrics["ml_used"] += 1
        h["confidence"] = round(max(float(h.get("confidence", 0)), float(m.get("confidence", 0))), 4)
        h["force_baim"] = bool(h.get("force_baim") or m.get("force_baim"))
        h["source"] = "gbm+heuristic"
        return h, feat

    prediction_metrics["heuristic_used"] += 1
    return h, feat

# ============================================================
# DATABASE
# ============================================================
# Columns that must be INTEGER in Postgres (PostgREST rejects 0.0 for INTEGER)
_INT_COLS   = {"miss_streak", "choked_ticks", "health", "armor",
               "damage_dealt", "hitgroup", "local_shots_fired"}
_FLOAT_COLS = {
    "velocity_x","velocity_y","speed_2d","distance",
    "local_velocity_x","local_velocity_y",
    "goal_feet_yaw","eye_yaw","body_yaw","desync_delta",
    "layer3_weight","layer3_cycle","relative_angle",
    "duck_amount","local_duck","confidence",
}

def _coerce_types(d: dict) -> dict:
    """Ensure int columns get int values and float columns get float values."""
    for col in _INT_COLS:
        if col in d:
            try:
                d[col] = int(float(d[col])) if d[col] is not None else 0
            except (TypeError, ValueError):
                d[col] = 0
    for col in _FLOAT_COLS:
        if col in d:
            try:
                d[col] = float(d[col]) if d[col] is not None else 0.0
            except (TypeError, ValueError):
                d[col] = 0.0
    return d

def _db_insert(payload: dict):
    safe = _coerce_types(_strip(payload))
    if supabase:
        try:
            supabase.table("resolver_data").insert(safe).execute()
        except Exception as e:
            print(f"🔥 DB insert: {e}")
    if len(memory_store) >= MAX_MEMORY:
        memory_store.pop(0)
    memory_store.append(safe)

def _db_outcome(shot_id: str, hit: bool, damage: int, reason: str,
                hitgroup: int, fallback: dict | None, lua_id: str | None = None):
    upd = {"hit": hit, "damage_dealt": int(damage),
           "miss_reason": str(reason), "hitgroup": int(hitgroup)}
    rows = 0
    if supabase:
        try:
            res  = supabase.table("resolver_data").update(upd).eq("shot_id", shot_id).execute()
            rows = len(res.data) if res.data else 0
            if rows == 0 and lua_id:
                res2 = supabase.table("resolver_data").update(upd).eq("lua_shot_id", lua_id).execute()
                rows = len(res2.data) if res2.data else 0
            print(f"{'✅ HIT' if hit else '❌ MISS'} | shot={shot_id[:8]} | dmg={damage} | rows={rows}")
            if rows == 0:
                # Park outcome — /predict hasn't inserted the row yet
                # Will be applied when /predict arrives (see predict_ep)
                key = lua_id or shot_id
                pending_outcomes[key] = {"upd": upd, "ts": time.time()}
                print(f"⏳ Queued outcome | lua={key} (predict not yet arrived)")
        except Exception as e:
            print(f"🔥 DB outcome: {e}")
    if rows == 0 and fallback:
        rec = dict(fallback)
        rec.update(upd)
        _db_insert(rec)
    for rec in memory_store:
        if rec.get("shot_id") in (shot_id, lua_id) or rec.get("lua_shot_id") in (lua_id, shot_id):
            rec.update(upd)
            break
    pending_shots.pop(shot_id, None)
    if AUTO_TRAIN_ENABLED and not TRAINING_IN_PROGRESS:
        labeled_now = _count_labeled_records()
        now = time.time()
        if labeled_now >= AUTO_TRAIN_MIN_LABELED and \
           labeled_now - last_train_labeled >= AUTO_TRAIN_EVERY and \
           now - last_train_at_ts >= AUTO_TRAIN_COOLDOWN_SEC:
            _start_training("auto_outcome")

def _fallback_rec(shot_id: str, hit: bool, damage: int, reason: str) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    base = {"shot_id": shot_id, "hit": hit, "damage_dealt": int(damage),
            "miss_reason": str(reason), "created_at": now}
    for k in FEATURES:
        base[k] = 0 if k in _INT_COLS else 0.0
    return base

def _build_payload(shot_id: str, feat: dict, pred: dict, data: dict, lua_shot_id: str = "") -> dict:
    cfg = data.get("config") or {}
    lp  = data.get("local_player") or {}
    tgt = data.get("target") or {}
    user = data.get("user") or {}
    discord_id = str(
        user.get("discord_id")
        or cfg.get("discord_id")
        or tgt.get("discord_id")
        or ""
    )
    return {
        "shot_id":           shot_id,
        "lua_shot_id":       str(lua_shot_id or ""),
        "hit":               None,
        "damage_dealt":      None,
        "miss_reason":       None,
        "hitgroup":          None,
        "steam_id":          str(tgt.get("steam_id") or ""),
        "discord_id":        discord_id,
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
    # Clean stale pending outcomes (older than 30s — predict never arrived)
    stale_out = [k for k, v in pending_outcomes.items() if now - v["ts"] > 30]
    for k in stale_out:
        pending_outcomes.pop(k, None)


def _save_model_artifacts(cv_mean: float, n_samples: int, trigger: str, features: list[str]):
    if not supabase:
        return
    snapshot_id = None
    try:
        snap = {
            "model_version": "resolver_ai_v4",
            "cv_accuracy": float(cv_mean),
            "n_samples": int(n_samples),
            "features_json": {"features": features, "trigger": trigger},
            "notes": f"auto={AUTO_TRAIN_ENABLED} trigger={trigger}",
        }
        r = supabase.table("model_snapshots").insert(snap).execute()
        if r.data and len(r.data) > 0:
            snapshot_id = r.data[0].get("id")
    except Exception as e:
        print(f"⚠️ snapshot save: {e}")

    _activate_profile_from_cv(cv_mean, snapshot_id)


def _activate_profile_from_cv(cv_mean: float, snapshot_id: int | None = None):
    if not supabase:
        return
    try:
        settings = {
            "resolver_mode": "Neural AI" if cv_mean >= 0.58 else "Adaptive",
            "bf_phase": "[3] Phase 3 (Custom)" if cv_mean >= 0.75 else "[2] Phase 2 (Aggressive)" if cv_mean >= 0.62 else "[1] Phase 1 (Adaptive)",
            "min_confidence": round(float(cv_mean) * 100, 1),
        }
        p = {
            "source_model_snapshot_id": snapshot_id,
            "settings_json": settings,
            "is_active": True,
            "effective_from": datetime.now(timezone.utc).isoformat(),
        }
        supabase.table("resolver_profiles").update({"is_active": False}).eq("is_active", True).execute()
        supabase.table("resolver_profiles").insert(p).execute()
    except Exception as e:
        print(f"⚠️ profile save: {e}")


def _save_fallback_profile(reason: str, labeled_rows: int, hits: int, misses: int):
    if not supabase:
        return
    try:
        settings = {
            "resolver_mode": "Adaptive",
            "bf_phase": "[1] Phase 1 (Adaptive)",
            "min_confidence": 50.0,
            "fallback_reason": reason,
            "labeled_rows": int(labeled_rows),
            "hits": int(hits),
            "misses": int(misses),
        }
        supabase.table("resolver_profiles").update({"is_active": False}).eq("is_active", True).execute()
        supabase.table("resolver_profiles").insert({
            "source_model_snapshot_id": None,
            "settings_json": settings,
            "is_active": True,
            "effective_from": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        print(f"⚠️ fallback profile save: {e}")


def _periodic_training_loop():
    while True:
        try:
            time.sleep(PERIODIC_TRAIN_INTERVAL_SEC)
            if not PERIODIC_TRAIN_ENABLED:
                continue
            if TRAINING_IN_PROGRESS:
                continue
            labeled_now = _count_labeled_records()
            now = time.time()
            if labeled_now < TRAIN_MIN_LABELED:
                continue
            if now - last_train_at_ts < AUTO_TRAIN_COOLDOWN_SEC:
                continue
            _start_training("periodic_timer")
        except Exception as e:
            print(f"⚠️ periodic trainer: {e}")


def _start_training(trigger: str) -> bool:
    global TRAINING_IN_PROGRESS
    with TRAINING_LOCK:
        if TRAINING_IN_PROGRESS:
            return False
        TRAINING_IN_PROGRESS = True
        training_status["in_progress"] = True
        training_status["last_trigger"] = trigger
        training_status["last_started_at"] = datetime.now(timezone.utc).isoformat()
        training_status["last_error"] = None
    t = threading.Thread(target=_train_bg, args=(trigger,), daemon=True)
    t.start()
    return True

# ============================================================
# TRAINING
# ============================================================
def _train_bg(trigger: str = "manual"):
    global AI_MODEL, SCALER, TRAINING_IN_PROGRESS, last_cv_accuracy
    global last_train_at_ts, last_train_labeled, last_training_trigger
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        print("🧠 Training start...")
        records = []
        if supabase:
            try:
                res = supabase.table("resolver_data").select("*") \
                              .not_.is_("hit", "null").order("created_at", desc=True).limit(TRAIN_MAX_ROWS).execute()
                records = res.data or []

                res_hard = supabase.table("resolver_data").select("*") \
                               .eq("hit", False).in_("miss_reason", list(HARD_MISS_REASONS)) \
                               .order("created_at", desc=True).limit(TRAIN_HARD_MISS_ROWS).execute()
                hard_rows = res_hard.data or []
                if hard_rows:
                    records += hard_rows
            except Exception as e:
                print(f"⚠️  DB fetch: {e}")
        records += [r for r in memory_store if r.get("hit") is not None]

        if len(records) < TRAIN_MIN_LABELED:
            print(f"❌ Need ≥{TRAIN_MIN_LABELED} labeled rows, have {len(records)}")
            training_status["last_error"] = f"Need >={TRAIN_MIN_LABELED} labeled rows, have {len(records)}"
            return

        df  = pd.DataFrame(records)
        if "shot_id" in df.columns:
            df["shot_id"] = df["shot_id"].astype(str)
            df = df.sort_values("created_at") if "created_at" in df.columns else df
            df = df.drop_duplicates(subset=["shot_id"], keep="last")
        avail = [f for f in FEATURES if f in df.columns]
        df  = df.dropna(subset=avail + ["hit"])
        df  = df[df["hit"].notna()]

        if len(df) < TRAIN_MIN_LABELED:
            print(f"❌ After clean: {len(df)} rows — abort")
            training_status["last_error"] = f"After clean only {len(df)} rows"
            return

        # Label: 1 = hit, 0 = miss (supervised hit probability model)
        df["label"] = (df["hit"] == True).astype(int)
        label_counts = df["label"].value_counts(dropna=False).to_dict()
        if len(label_counts.keys()) < 2:
            hits = int(label_counts.get(1, 0))
            misses = int(label_counts.get(0, 0))
            msg = f"single_class_dataset hits={hits} misses={misses}"
            print(f"❌ {msg}")
            training_status["last_error"] = msg
            _save_fallback_profile("single_class_dataset", len(df), hits, misses)
            return

        X  = df[avail].values.astype(np.float32)
        y  = df["label"].values
        miss_reason_series = df["miss_reason"] if "miss_reason" in df.columns else pd.Series([""] * len(df))
        miss_reason_norm = miss_reason_series.fillna("").astype(str).str.lower()
        hard_mask = (df["label"] == 0) & (miss_reason_norm.isin(HARD_MISS_REASONS))
        sw = np.where(df["hit"].values, 1.2, 1.0)
        sw = np.where(hard_mask.values, sw * TRAIN_HARD_MISS_BOOST, sw)

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
        last_train_at_ts = time.time()
        last_train_labeled = _count_labeled_records()
        last_training_trigger = trigger
        joblib.dump({
            "model": model, "scaler": scaler, "features": avail,
            "cv_accuracy": cv_mean, "n_samples": len(df),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }, MODEL_PATH)
        AI_MODEL = model
        SCALER   = scaler
        _save_model_artifacts(cv_mean, len(df), trigger, avail)
        training_status["last_cv"] = cv_mean
        training_status["last_samples"] = int(len(df))
        print("✅ Model saved")

    except Exception as e:
        print(f"❌ Training: {e}")
        training_status["last_error"] = str(e)
        import traceback; traceback.print_exc()
    finally:
        TRAINING_IN_PROGRESS = False
        training_status["in_progress"] = False
        training_status["last_finished_at"] = datetime.now(timezone.utc).isoformat()

# ============================================================
# STARTUP
# ============================================================
@app.on_event("startup")
async def _startup():
    _load_model()
    if supabase:
        try:
            r = supabase.table("resolver_data").select("id", count="exact").limit(1).execute()
            total = r.count or 0
            print(f"✅ DB probe OK | total rows: {total}")
        except Exception as e:
            print(f"⚠️  DB probe: {e}")
    if PERIODIC_TRAIN_ENABLED:
        t = threading.Thread(target=_periodic_training_loop, daemon=True)
        t.start()
        print(f"✅ Periodic trainer enabled | interval={PERIODIC_TRAIN_INTERVAL_SEC}s")

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    _presence_cleanup()
    return {
        "status":   "$hematic AI Backend v4.0",
        "model":    f"GBM | CV={last_cv_accuracy:.3f}" if AI_MODEL else "heuristic-only",
        "records":  len(memory_store),
        "pending":  len(pending_shots),
        "presence_online": len(presence_store),
    }


@app.get("/logo")
async def logo_default_ep():
    return RedirectResponse(url=f"{LOGO_GITHUB_RAW_BASE}/shematiclogo.png", status_code=307)


@app.get("/logo/{filename}")
async def logo_file_ep(filename: str):
    name = str(filename or "").strip()
    if name == "" or ".." in name or "/" in name or "\\" in name:
        return JSONResponse(status_code=400, content={"error": "invalid filename"})
    if not name.lower().endswith(".png"):
        return JSONResponse(status_code=400, content={"error": "only .png allowed"})
    return RedirectResponse(url=f"{LOGO_GITHUB_RAW_BASE}/{name}", status_code=307)


@app.post("/presence/heartbeat")
async def presence_heartbeat_ep(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "bad json"})

    steam_id = str(data.get("steam_id") or "").strip()
    if steam_id == "" or steam_id == "0":
        return JSONResponse(status_code=400, content={"error": "steam_id required"})

    user_id = str(data.get("user_id") or data.get("discord_id") or "").strip()
    map_name = str(data.get("map") or "").strip()
    version = str(data.get("version") or "").strip()
    now = time.time()

    presence_store[steam_id] = {
        "steam_id": steam_id,
        "user_id": user_id,
        "map": map_name,
        "version": version,
        "last_seen": now,
    }
    _presence_cleanup(now)

    return JSONResponse({
        "status": "ok",
        "steam_id": steam_id,
        "ttl_sec": PRESENCE_TTL_SEC,
        "online": len(presence_store),
    })


@app.post("/presence/list")
async def presence_list_ep(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "bad json"})

    _presence_cleanup()

    raw = data.get("steam_ids") or []
    if not isinstance(raw, list):
        return JSONResponse(status_code=400, content={"error": "steam_ids must be list"})

    in_match = []
    seen = set()
    for sid in raw:
        s = str(sid or "").strip()
        if s == "" or s == "0" or s in seen:
            continue
        seen.add(s)
        in_match.append(s)

    active_steam_ids = [sid for sid in in_match if sid in presence_store]
    users = {}
    for sid in active_steam_ids:
        row = presence_store.get(sid) or {}
        users[sid] = {
            "user_id": str(row.get("user_id") or ""),
            "map": str(row.get("map") or ""),
            "version": str(row.get("version") or ""),
            "last_seen": float(row.get("last_seen") or 0.0),
        }

    return JSONResponse({
        "status": "ok",
        "active_steam_ids": active_steam_ids,
        "users": users,
        "online_total": len(presence_store),
        "ttl_sec": PRESENCE_TTL_SEC,
    })


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
    prediction_metrics["total"] += 1

    lua_id   = str(data.get("shot_id") or "")
    shot_id  = _resolve_sid(lua_id) if lua_id else str(uuid.uuid4())
    tgt      = data.get("target") or {}
    steam_id = str(tgt.get("steam_id") or "")

    feat           = extract_features(data)
    valid, inv_rsn = _validate(feat)

    if not valid:
        prediction_metrics["invalid_telemetry"] += 1
        print(f"⚠️  Bad telemetry [{lua_id}]: {inv_rsn}")
        res = _heuristic(feat, steam_id)
        res["shot_id"] = shot_id
        res["warning"] = inv_rsn
        return JSONResponse(res)

    pred, feat = _predict(data, steam_id)
    payload    = _build_payload(shot_id, feat, pred, data, lua_id)

    pending_shots[shot_id] = {"ts": time.time(), "payload": payload, "lua_id": lua_id, "uuid": shot_id}
    # Insert synchronously so row exists before we check pending_outcomes below
    _db_insert(payload)

    # Apply any queued outcome (arrived before this predict)
    queued = pending_outcomes.pop(lua_id, None) or pending_outcomes.pop(shot_id, None)
    if queued and supabase:
        try:
            r2 = supabase.table("resolver_data").update(queued["upd"]).eq("shot_id", shot_id).execute()
            print(f"✅ Applied queued outcome | shot={shot_id[:8]} | rows={len(r2.data) if r2.data else 0}")
        except Exception as e:
            print(f"🔥 Queued outcome apply: {e}")

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

    raw_id  = str(data.get("shot_id") or "")
    lua_id  = str(data.get("lua_id")  or raw_id)  # original Lua integer string
    if not raw_id and not lua_id:
        return JSONResponse({"status": "ignored"})

    # Resolve UUID: try shot_id_map with both raw_id and lua_id
    shot_id = raw_id if _is_uuid(raw_id) else (shot_id_map.get(raw_id) or shot_id_map.get(lua_id) or raw_id)

    hit      = bool(data.get("hit", False))
    damage   = int(data.get("damage") or 0)
    reason   = data.get("reason") or "none"
    hitgroup = int(data.get("hitgroup") or 0)

    for steam_id, hist in player_history.items():
        for entry in hist:
            if entry.get("shot_id") in (shot_id, lua_id):
                entry["hit"] = hit
                break

    # Try to find pending payload by UUID or lua integer
    pending  = pending_shots.get(shot_id)
    fallback = pending["payload"] if pending else None

    # Pass lua_id so _db_outcome can retry with integer if UUID row not found
    bg.add_task(_db_outcome, shot_id, hit, damage, reason, hitgroup, fallback, lua_id)
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

    global analyze_counter
    if valid and ANALYZE_DB_WRITE:
        analyze_counter += 1
        if analyze_counter % ANALYZE_DB_EVERY_N == 0:
            shot_id = str(uuid.uuid4())
            bg.add_task(_db_insert, _build_payload(shot_id, feat, pred, data))

    conf = pred["confidence"]
    # bf_phase strings MUST match Lua combobox options exactly
    if conf > 0.78:
        bf = "[3] Phase 3 (Custom)"
    elif conf > 0.62:
        bf = "[2] Phase 2 (Aggressive)"
    else:
        bf = "[1] Phase 1 (Adaptive)"
    return JSONResponse({
        "bf_phase":       bf,
        "resolver_mode":  "Neural AI" if str(pred.get("source", "")).find("gbm") != -1 else "Adaptive",
        "override_baim":  pred["force_baim"],
        "confidence":     conf,
    })


@app.post("/train")
async def train_ep(bg: BackgroundTasks):
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
    started = _start_training("manual_api")
    return JSONResponse({
        "status": "started" if started else "busy", "total": total, "labeled": labeled,
        "has_model": AI_MODEL is not None,
    })


@app.get("/training/status")
async def training_status_ep():
    return JSONResponse({
        "in_progress": TRAINING_IN_PROGRESS,
        "last_trigger": training_status.get("last_trigger"),
        "last_started_at": training_status.get("last_started_at"),
        "last_finished_at": training_status.get("last_finished_at"),
        "last_error": training_status.get("last_error"),
        "last_cv": round(float(training_status.get("last_cv", 0.0)), 4),
        "last_samples": int(training_status.get("last_samples", 0)),
        "last_labeled_seen": int(last_train_labeled),
        "periodic_train_enabled": PERIODIC_TRAIN_ENABLED,
        "periodic_train_interval_sec": PERIODIC_TRAIN_INTERVAL_SEC,
        "train_min_labeled": TRAIN_MIN_LABELED,
        "train_max_rows": TRAIN_MAX_ROWS,
    })


@app.get("/profile/active")
async def active_profile_ep():
    if not supabase:
        return JSONResponse({"status": "no_db", "settings": {}})
    try:
        r = supabase.table("resolver_profiles").select("*") \
                    .eq("is_active", True).order("created_at", desc=True).limit(1).execute()
        row = (r.data or [{}])[0] if r.data else {}
        return JSONResponse({
            "status": "ok" if row else "empty",
            "settings": row.get("settings_json") or {},
            "source_model_snapshot_id": row.get("source_model_snapshot_id"),
            "effective_from": row.get("effective_from") or row.get("created_at"),
        })
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e), "settings": {}})


@app.get("/profile/{discord_id}")
async def user_profile_ep(discord_id: str):
    did = str(discord_id or "").strip()
    if did == "":
        return JSONResponse(status_code=400, content={"error": "discord_id required"})

    total_sent = 0
    labeled = 0
    hits = 0
    avg_conf = 0.0
    recent = []

    local_rows = [r for r in memory_store if str(r.get("discord_id") or "") == did]
    if local_rows:
        total_sent = len(local_rows)
        labeled = sum(1 for r in local_rows if r.get("hit") is not None)
        hits = sum(1 for r in local_rows if r.get("hit") is True)
        conf_vals = [float(r.get("confidence") or 0.0) for r in local_rows]
        avg_conf = (sum(conf_vals) / len(conf_vals)) if conf_vals else 0.0
        recent = local_rows[-5:]

    if supabase:
        try:
            r_total = supabase.table("resolver_data").select("id", count="exact") \
                .eq("discord_id", did).limit(1).execute()
            total_sent = max(total_sent, r_total.count or 0)
        except Exception:
            pass
        try:
            r_lab = supabase.table("resolver_data").select("id", count="exact") \
                .eq("discord_id", did).not_.is_("hit", "null").limit(1).execute()
            labeled = max(labeled, r_lab.count or 0)
        except Exception:
            pass
        try:
            r_hit = supabase.table("resolver_data").select("id", count="exact") \
                .eq("discord_id", did).eq("hit", True).limit(1).execute()
            hits = max(hits, r_hit.count or 0)
        except Exception:
            pass
        try:
            r_conf = supabase.table("resolver_data").select("confidence") \
                .eq("discord_id", did).order("created_at", desc=True).limit(200).execute()
            rows_conf = r_conf.data or []
            if rows_conf:
                vals = [float(x.get("confidence") or 0.0) for x in rows_conf]
                avg_conf = sum(vals) / len(vals)
        except Exception:
            pass
        try:
            r_recent = supabase.table("resolver_data").select("created_at,hit,damage_dealt,miss_reason,confidence,weapon,bf_phase,prediction_source") \
                .eq("discord_id", did).order("created_at", desc=True).limit(5).execute()
            recent = r_recent.data or recent
        except Exception:
            pass

    hit_rate = round((hits / labeled * 100.0), 2) if labeled > 0 else 0.0

    return JSONResponse({
        "discord_id": did,
        "shots_sent": int(total_sent),
        "labeled_shots": int(labeled),
        "hits": int(hits),
        "hit_rate": hit_rate,
        "avg_confidence": round(float(avg_conf), 2),
        "training_in_progress": TRAINING_IN_PROGRESS,
        "model_version": f"GBM v4 | CV={last_cv_accuracy:.3f}" if AI_MODEL else "none",
        "recent": recent,
    })


@app.get("/profile")
async def user_profile_query_ep(discord_id: str = ""):
    return await user_profile_ep(discord_id)


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
    hit_rate  = round((hits / labeled * 100) if labeled > 0 else 0.0, 1)

    return JSONResponse({
        "users_online":      1,
        "patterns_saved":    total,
        "resolver_records":  labeled,
        "ai_iterations":     labeled,
        "avg_confidence":    real_conf,
        "hit_rate":          hit_rate,
        "ai_status":         "GBM" if AI_MODEL else "Heuristic",
        "last_sync":         datetime.now(timezone.utc).isoformat(),
        "your_contribution": "Active" if memory_store else "Inactive",
        "model_version":     f"GBM v4 | CV={last_cv_accuracy:.3f}" if AI_MODEL else "none",
        "ml_usage_rate":     round((prediction_metrics["ml_used"] / prediction_metrics["total"] * 100) if prediction_metrics["total"] else 0.0, 2),
        "heuristic_usage_rate": round((prediction_metrics["heuristic_used"] / prediction_metrics["total"] * 100) if prediction_metrics["total"] else 0.0, 2),
        "invalid_telemetry_rate": round((prediction_metrics["invalid_telemetry"] / prediction_metrics["total"] * 100) if prediction_metrics["total"] else 0.0, 2),
        "training_in_progress": TRAINING_IN_PROGRESS,
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
