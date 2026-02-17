# energy/app.py

from fastapi import FastAPI, Header, HTTPException, Query, Request, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Tuple
from sqlalchemy import create_engine, text
import os
import time
import json
import traceback
import math
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from pathlib import Path
from fastapi.staticfiles import StaticFiles

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

load_dotenv()

app = FastAPI(title="Energy API", version="0.2.0")
app.mount("/static", StaticFiles(directory="/opt/energy/static"), name="static")

# =========================
# CONFIG
# =========================
DASHBOARD_PATH = Path("/opt/energy/templates/dashboard.html")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")

# Token opcional (si no está, no valida)
API_TOKEN = (os.getenv("ENERGY_API_TOKEN") or os.getenv("ENERGY_API_KEY") or "").strip()

# Moneda fija (para evitar RON->EUR). Si quieres volver a moneda por mes, se puede.
FIXED_CURRENCY = (os.getenv("ENERGY_CURRENCY") or "RON").strip() or "RON"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# =========================
# MODELS
# =========================

class ChannelPayload(BaseModel):
    # Intervalos (deltas / energía acumulada por intervalo)
    kwh_import: Optional[float] = 0.0
    kwh_export: Optional[float] = 0.0
    kvarh_ind: Optional[Dict[str, float]] = Field(default_factory=dict)  # {"X1":..., "X2":..., "X3":...}
    kvarh_cap: Optional[Dict[str, float]] = Field(default_factory=dict)

    # Instantáneos (todo lo que mande el smartmeter, sin procesar)
    instant: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class Payload(BaseModel):
    """
    Nuevo esquema:
      - meter_id: id único del contador (recomendado)
      - device_id: compat legacy (si llega, se usa como meter_id)
      - edge_id/site_id: opcionales (para trazabilidad multi-raspberry / multi-site)

    Compatibilidad: aceptamos payloads antiguos con device_id.
    """
    meter_id: Optional[str] = None
    device_id: Optional[str] = None  # legacy
    edge_id: Optional[str] = None
    site_id: Optional[str] = None

    ts_from: int
    ts_to: int
    channels: Dict[str, ChannelPayload]

    def resolved_meter_id(self) -> str:
        mid = (self.meter_id or self.device_id or "").strip()
        return mid


class TariffPayload(BaseModel):
    # Se mantiene por compatibilidad, pero el backend responderá/guardará FIXED_CURRENCY
    currency: str = "RON"
    price_active_kwh: float = 0.0
    price_ind_x1: float = 0.0
    price_ind_x2: float = 0.0
    price_ind_x3: float = 0.0
    price_cap_x1: float = 0.0
    price_cap_x2: float = 0.0
    price_cap_x3: float = 0.0


# =========================
# DB INIT (CREATE TABLES)
# =========================

@app.on_event("startup")
def startup_init():
    """
    Crea tablas necesarias si no existen.
    NO toca tu tabla telemetry_intervals existente.
    """
    with engine.begin() as conn:
        # --- meters ---
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS meters (
                id TEXT PRIMARY KEY,
                model TEXT,
                wiring TEXT,
                measurement_mode TEXT,
                is_enabled BOOLEAN NOT NULL DEFAULT TRUE
            );
        """))

        # --- telemetry_sources (opcional, para trazabilidad multi-edge) ---
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS telemetry_sources (
                meter_id TEXT NOT NULL REFERENCES meters(id) ON DELETE CASCADE,
                edge_id  TEXT,
                site_id  TEXT,
                last_seen_ts INTEGER NOT NULL DEFAULT 0,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (meter_id)
            );
        """))

        # --- instants ---
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS telemetry_instants (
                id BIGSERIAL PRIMARY KEY,
                meter_id TEXT NOT NULL REFERENCES meters(id) ON DELETE CASCADE,
                channel TEXT NOT NULL,
                ts INTEGER NOT NULL,
                data JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (meter_id, channel, ts)
            );
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_telemetry_instants_meter_ts
            ON telemetry_instants (meter_id, ts DESC);
        """))

        # --- tariffs monthly ---
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tariffs_monthly (
                meter_id TEXT NOT NULL REFERENCES meters(id) ON DELETE CASCADE,
                channel  TEXT NOT NULL,
                ym       TEXT NOT NULL, -- 'YYYY-MM'
                currency TEXT NOT NULL DEFAULT 'RON',

                price_active_kwh DOUBLE PRECISION NOT NULL DEFAULT 0,

                price_ind_x1     DOUBLE PRECISION NOT NULL DEFAULT 0,
                price_ind_x2     DOUBLE PRECISION NOT NULL DEFAULT 0,
                price_ind_x3     DOUBLE PRECISION NOT NULL DEFAULT 0,

                price_cap_x1     DOUBLE PRECISION NOT NULL DEFAULT 0,
                price_cap_x2     DOUBLE PRECISION NOT NULL DEFAULT 0,
                price_cap_x3     DOUBLE PRECISION NOT NULL DEFAULT 0,

                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (meter_id, channel, ym)
            );
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tariffs_monthly_meter_channel_ym
            ON tariffs_monthly (meter_id, channel, ym DESC);
        """))


# =========================
# AUTH HELPERS
# =========================

def _auth_ok(
    authorization: Optional[str],
    x_api_key: Optional[str],
    token_qs: Optional[str] = None,
) -> bool:
    """
    Si API_TOKEN está vacío => no valida (modo abierto).

    Si está definido, acepta:
      - Header: X-API-Key: <TOKEN>
      - Header: Authorization: Bearer <TOKEN>
      - Query:  ?token=<TOKEN>
    """
    if not API_TOKEN:
        return True

    if x_api_key and x_api_key.strip() == API_TOKEN:
        return True

    if authorization and authorization.strip() == f"Bearer {API_TOKEN}":
        return True

    if token_qs and token_qs.strip() == API_TOKEN:
        return True

    return False


def require_key(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    token_qs = request.query_params.get("token")
    if not _auth_ok(authorization, x_api_key, token_qs):
        raise HTTPException(status_code=401, detail="Invalid API key")


# =========================
# HELPERS
# =========================

def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _sanitize_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_sanitize_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _sanitize_jsonable(v) for k, v in obj.items()}
    return str(obj)


def _compute_kvar_from_p_pf(p_w: Any, pf: Any) -> float:
    """
    Q (var) = P (W) * tan(acos(pf))
    """
    try:
        p = float(p_w) if p_w is not None else 0.0
        pfv = float(pf) if pf is not None else 0.0
        if not math.isfinite(p) or not math.isfinite(pfv):
            return 0.0
        pfv = max(-1.0, min(1.0, pfv))
        if abs(pfv) < 1e-6:
            return 0.0
        phi = math.acos(pfv)
        q = p * math.tan(phi)
        if not math.isfinite(q):
            return 0.0
        return q
    except Exception:
        return 0.0


def _get_tz(tz_str: Optional[str]):
    if not tz_str:
        return timezone.utc
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(tz_str)
    except Exception:
        return timezone.utc


def _period_bounds(period: str, tz_str: Optional[str]) -> Tuple[int, int]:
    tz = _get_tz(tz_str)
    now_local = datetime.now(tz)

    if period == "day":
        start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        end_local = start_local + timedelta(days=1)
    elif period == "month":
        start_local = now_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if start_local.month == 12:
            end_local = start_local.replace(year=start_local.year + 1, month=1)
        else:
            end_local = start_local.replace(month=start_local.month + 1)
    else:
        raise ValueError("period must be 'day' or 'month'")

    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)
    return int(start_utc.timestamp()), int(end_utc.timestamp())


def _ym_now_utc() -> str:
    now = datetime.now(timezone.utc)
    return f"{now.year:04d}-{now.month:02d}"


def _normalize_ym(ym: str) -> str:
    ym = (ym or "").strip()
    # Acepta "YYYY-MM" (lo que devuelve <input type="month">)
    if len(ym) == 7 and ym[4] == "-":
        return ym
    # fallback
    return _ym_now_utc()


# =========================
# BASIC ROUTES
# =========================

@app.get("/")
def read_root():
    return {"status": "ok", "service": "energy-api"}


@app.get("/health", dependencies=[Depends(require_key)])
def health():
    return {"status": "ok"}


# =========================
# INGEST
# =========================

@app.post("/api/energy", dependencies=[Depends(require_key)])
async def ingest(payload: Payload):
    try:
        meter_id = payload.resolved_meter_id()
        if not meter_id:
            raise HTTPException(status_code=400, detail="meter_id required (or legacy device_id)")

        with engine.begin() as conn:
            # meters upsert (mínimo)
            conn.execute(
                text("""
                    INSERT INTO meters (id, model, wiring, measurement_mode, is_enabled)
                    VALUES (:id, :model, :wiring, :mode, TRUE)
                    ON CONFLICT (id) DO NOTHING
                """),
                {"id": meter_id, "model": "unknown", "wiring": "unknown", "mode": "unknown"},
            )

            # guardamos trazabilidad de fuente (edge/site opcional)
            if payload.edge_id or payload.site_id:
                conn.execute(
                    text("""
                        INSERT INTO telemetry_sources (meter_id, edge_id, site_id, last_seen_ts)
                        VALUES (:m, :e, :s, :ts)
                        ON CONFLICT (meter_id) DO UPDATE SET
                            edge_id = COALESCE(EXCLUDED.edge_id, telemetry_sources.edge_id),
                            site_id = COALESCE(EXCLUDED.site_id, telemetry_sources.site_id),
                            last_seen_ts = GREATEST(telemetry_sources.last_seen_ts, EXCLUDED.last_seen_ts),
                            updated_at = now()
                    """),
                    {"m": meter_id, "e": payload.edge_id, "s": payload.site_id, "ts": int(time.time())},
                )

            for channel, v in (payload.channels or {}).items():
                if v is None:
                    v = ChannelPayload()

                kwh_import = _to_float(v.kwh_import, 0.0)
                kwh_export = _to_float(v.kwh_export, 0.0)

                kvarh_ind = v.kvarh_ind or {}
                i1 = _to_float(kvarh_ind.get("X1"), 0.0)
                i2 = _to_float(kvarh_ind.get("X2"), 0.0)
                i3 = _to_float(kvarh_ind.get("X3"), 0.0)

                kvarh_cap = v.kvarh_cap or {}
                c1 = _to_float(kvarh_cap.get("X1"), 0.0)
                c2 = _to_float(kvarh_cap.get("X2"), 0.0)
                c3 = _to_float(kvarh_cap.get("X3"), 0.0)

                # Intervalos (si llega vacío por un payload de "instant only", lo dejamos en 0)
                conn.execute(
                    text("""
                        INSERT INTO telemetry_intervals (
                            meter_id, channel, ts_from, ts_to,
                            kwh_import, kwh_export,
                            kvarh_ind_x1, kvarh_ind_x2, kvarh_ind_x3,
                            kvarh_cap_x1, kvarh_cap_x2, kvarh_cap_x3
                        )
                        VALUES (
                            :meter_id, :channel, :ts_from, :ts_to,
                            :kwh_import, :kwh_export,
                            :i1, :i2, :i3,
                            :c1, :c2, :c3
                        )
                        ON CONFLICT (meter_id, channel, ts_from, ts_to)
                        DO UPDATE SET
                            kwh_import   = EXCLUDED.kwh_import,
                            kwh_export   = EXCLUDED.kwh_export,
                            kvarh_ind_x1 = EXCLUDED.kvarh_ind_x1,
                            kvarh_ind_x2 = EXCLUDED.kvarh_ind_x2,
                            kvarh_ind_x3 = EXCLUDED.kvarh_ind_x3,
                            kvarh_cap_x1 = EXCLUDED.kvarh_cap_x1,
                            kvarh_cap_x2 = EXCLUDED.kvarh_cap_x2,
                            kvarh_cap_x3 = EXCLUDED.kvarh_cap_x3
                    """),
                    {
                        "meter_id": meter_id,
                        "channel": channel,
                        "ts_from": int(payload.ts_from),
                        "ts_to": int(payload.ts_to),
                        "kwh_import": kwh_import,
                        "kwh_export": kwh_export,
                        "i1": i1, "i2": i2, "i3": i3,
                        "c1": c1, "c2": c2, "c3": c3,
                    },
                )

                # Instants (si llegan)
                inst = v.instant or {}
                if inst:
                    inst_clean = _sanitize_jsonable(inst)
                    inst_json = json.dumps(inst_clean, ensure_ascii=False)
                    conn.execute(
                        text("""
                            INSERT INTO telemetry_instants (meter_id, channel, ts, data)
                            VALUES (:meter_id, :channel, :ts, (:data)::jsonb)
                            ON CONFLICT (meter_id, channel, ts)
                            DO UPDATE SET data = EXCLUDED.data
                        """),
                        {
                            "meter_id": meter_id,
                            "channel": channel,
                            "ts": int(payload.ts_to),
                            "data": inst_json,
                        },
                    )

        return {"ok": True, "meter_id": meter_id}

    except HTTPException:
        raise
    except Exception as e:
        print("!!! ERROR in /api/energy:", repr(e))
        traceback.print_exc()
        raise


# =========================
# API FOR DASHBOARD
# =========================

@app.get("/api/meters", dependencies=[Depends(require_key)])
def api_meters() -> List[str]:
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT id FROM meters ORDER BY id ASC")).fetchall()
        return [r[0] for r in rows]


@app.get("/api/latest_interval", dependencies=[Depends(require_key)])
def api_latest_interval(meter_id: str = Query(...)):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (channel)
                channel, ts_from, ts_to,
                kwh_import, kwh_export,
                kvarh_ind_x1, kvarh_ind_x2, kvarh_ind_x3,
                kvarh_cap_x1, kvarh_cap_x2, kvarh_cap_x3
            FROM telemetry_intervals
            WHERE meter_id = :m
            ORDER BY channel, ts_to DESC
        """), {"m": meter_id}).fetchall()

        out = []
        for r in rows:
            out.append({
                "channel": r[0],
                "ts_from": int(r[1]) if r[1] is not None else None,
                "ts_to": int(r[2]) if r[2] is not None else None,
                "kwh_import": float(r[3] or 0.0),
                "kwh_export": float(r[4] or 0.0),
                "kvarh_ind": {"X1": float(r[5] or 0.0), "X2": float(r[6] or 0.0), "X3": float(r[7] or 0.0)},
                "kvarh_cap": {"X1": float(r[8] or 0.0), "X2": float(r[9] or 0.0), "X3": float(r[10] or 0.0)},
            })
        return {"meter_id": meter_id, "rows": out, "server_ts": int(time.time())}


@app.get("/api/totals_interval", dependencies=[Depends(require_key)])
def api_totals_interval(
    meter_id: str = Query(...),
    period: str = Query("day"),
    tz: str = Query("UTC"),
):
    if period not in ("day", "month"):
        raise HTTPException(status_code=400, detail="period must be 'day' or 'month'")

    start_ts, end_ts = _period_bounds(period, tz)

    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT
                channel,
                MIN(ts_from) AS ts_from,
                MAX(ts_to)   AS ts_to,
                COALESCE(SUM(kwh_import),0) AS kwh_import,
                COALESCE(SUM(kwh_export),0) AS kwh_export,
                COALESCE(SUM(kvarh_ind_x1),0) AS kvarh_ind_x1,
                COALESCE(SUM(kvarh_ind_x2),0) AS kvarh_ind_x2,
                COALESCE(SUM(kvarh_ind_x3),0) AS kvarh_ind_x3,
                COALESCE(SUM(kvarh_cap_x1),0) AS kvarh_cap_x1,
                COALESCE(SUM(kvarh_cap_x2),0) AS kvarh_cap_x2,
                COALESCE(SUM(kvarh_cap_x3),0) AS kvarh_cap_x3
            FROM telemetry_intervals
            WHERE meter_id = :m
              AND ts_from >= :start_ts
              AND ts_from <  :end_ts
            GROUP BY channel
            ORDER BY channel ASC
        """), {"m": meter_id, "start_ts": start_ts, "end_ts": end_ts}).fetchall()

        out = []
        for r in rows:
            out.append({
                "channel": r[0],
                "ts_from": int(r[1]) if r[1] is not None else start_ts,
                "ts_to": int(r[2]) if r[2] is not None else end_ts,
                "kwh_import": float(r[3] or 0.0),
                "kwh_export": float(r[4] or 0.0),
                "kvarh_ind": {"X1": float(r[5] or 0.0), "X2": float(r[6] or 0.0), "X3": float(r[7] or 0.0)},
                "kvarh_cap": {"X1": float(r[8] or 0.0), "X2": float(r[9] or 0.0), "X3": float(r[10] or 0.0)},
            })

        return {
            "meter_id": meter_id,
            "period": period,
            "tz": tz,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "rows": out,
            "server_ts": int(time.time()),
        }


@app.get("/api/latest_instant", dependencies=[Depends(require_key)])
def api_latest_instant(meter_id: str = Query(...), channel: str = Query("T")):
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT ts, data
            FROM telemetry_instants
            WHERE meter_id = :m AND channel = :c
            ORDER BY ts DESC
            LIMIT 1
        """), {"m": meter_id, "c": channel}).fetchone()

        if not row:
            return {"meter_id": meter_id, "channel": channel, "ts": None, "data": None}

        ts = int(row[0])
        data = row[1] or {}

        try:
            p = data.get("active_power_w")
            pf = data.get("power_factor")
            data = dict(data)
            data["reactive_power_var"] = _compute_kvar_from_p_pf(p, pf)
        except Exception:
            data = dict(data) if isinstance(data, dict) else {}
            data["reactive_power_var"] = 0.0

        return {"meter_id": meter_id, "channel": channel, "ts": ts, "data": data}


# =========================
# SERIES (FOR CHARTS)
# =========================

ALLOWED_METRICS = {
    # instants (JSONB data)
    "voltage_v": ("instants", "voltage_v"),
    "current_a": ("instants", "current_a"),
    "power_factor": ("instants", "power_factor"),
    "active_power_w": ("instants", "active_power_w"),
    "reactive_power_var": ("instants", "reactive_power_var"),  # lo calculamos al vuelo igual que en latest_instant

    # intervals (table columns)
    "energy_active_kwh": ("intervals", "kwh_import"),          # activa consumida
    "energy_reactive_kvarh": ("intervals", "kvarh_total"),     # inductiva+capacitiva (X1+X2+X3)
}

@app.get("/api/series", dependencies=[Depends(require_key)])
def api_series(
    meter_id: str = Query(...),
    metric: str = Query(...),
    channel: str = Query("T"),
    start_ts: int = Query(..., ge=0),
    end_ts: int = Query(..., ge=0),
    step_s: int = Query(300, ge=10, le=86400),
    tz: str = Query("UTC"),
):
    """
    Devuelve puntos para un gráfico: buckets de step_s entre start_ts y end_ts.
    - metric: voltage_v/current_a/power_factor/energy_active_kwh/energy_reactive_kvarh/...
    - channel: T/L1/L2/L3
    - start_ts/end_ts: epoch seconds (se interpretan como instantes UTC; tz es solo informativo ahora)
    """
    if end_ts <= start_ts:
        raise HTTPException(status_code=400, detail="end_ts must be > start_ts")

    metric = (metric or "").strip()
    if metric not in ALLOWED_METRICS:
        raise HTTPException(status_code=400, detail=f"metric not allowed: {metric}")

    kind, key = ALLOWED_METRICS[metric]

    # Bucket start = floor(ts/step)*step
    def _bucket(ts_col: str) -> str:
        return f"(FLOOR({ts_col}::double precision / :step_s) * :step_s)::bigint"

    def _safe_num(v):
        if v is None:
            return None
        try:
            f = float(v)
            return f if math.isfinite(f) else None
        except Exception:
            return None

    with engine.begin() as conn:
        if kind == "instants":
            if key == "reactive_power_var":
                q = text(f"""
                    SELECT
                        {_bucket("ts")} AS bts,
                        AVG(
                            COALESCE((data->>'active_power_w')::double precision, 0.0)
                            *
                            TAN(ACOS(
                                GREATEST(-1.0, LEAST(1.0,
                                    COALESCE((data->>'power_factor')::double precision, 0.0)
                                ))
                            ))
                        ) AS v
                    FROM telemetry_instants
                    WHERE meter_id = :m
                      AND channel = :c
                      AND ts >= :start_ts
                      AND ts <  :end_ts
                    GROUP BY bts
                    ORDER BY bts ASC
                """)
                rows = conn.execute(q, {
                    "m": meter_id, "c": channel,
                    "start_ts": start_ts, "end_ts": end_ts,
                    "step_s": step_s
                }).fetchall()
            else:
                # Nota: si algún valor no es casteable a float, Postgres puede fallar.
                # Asumimos que guardamos números en JSON.
                q = text(f"""
                    SELECT
                        {_bucket("ts")} AS bts,
                        AVG((data->>:k)::double precision) AS v
                    FROM telemetry_instants
                    WHERE meter_id = :m
                      AND channel = :c
                      AND ts >= :start_ts
                      AND ts <  :end_ts
                      AND (data ? :k)
                    GROUP BY bts
                    ORDER BY bts ASC
                """)
                rows = conn.execute(q, {
                    "m": meter_id, "c": channel,
                    "start_ts": start_ts, "end_ts": end_ts,
                    "step_s": step_s,
                    "k": key,
                }).fetchall()

            points = [{"ts": int(r[0]), "v": _safe_num(r[1])} for r in rows]

        else:
            if key == "kwh_import":
                q = text(f"""
                    SELECT
                        {_bucket("ts_from")} AS bts,
                        COALESCE(SUM(kwh_import),0) AS v
                    FROM telemetry_intervals
                    WHERE meter_id = :m
                      AND channel = :c
                      AND ts_from >= :start_ts
                      AND ts_from <  :end_ts
                    GROUP BY bts
                    ORDER BY bts ASC
                """)
                rows = conn.execute(q, {
                    "m": meter_id, "c": channel,
                    "start_ts": start_ts, "end_ts": end_ts,
                    "step_s": step_s
                }).fetchall()
                points = [{"ts": int(r[0]), "v": float(r[1] or 0.0)} for r in rows]

            elif key == "kvarh_total":
                q = text(f"""
                    SELECT
                        {_bucket("ts_from")} AS bts,
                        COALESCE(SUM(
                            kvarh_ind_x1 + kvarh_ind_x2 + kvarh_ind_x3 +
                            kvarh_cap_x1 + kvarh_cap_x2 + kvarh_cap_x3
                        ),0) AS v
                    FROM telemetry_intervals
                    WHERE meter_id = :m
                      AND channel = :c
                      AND ts_from >= :start_ts
                      AND ts_from <  :end_ts
                    GROUP BY bts
                    ORDER BY bts ASC
                """)
                rows = conn.execute(q, {
                    "m": meter_id, "c": channel,
                    "start_ts": start_ts, "end_ts": end_ts,
                    "step_s": step_s
                }).fetchall()
                points = [{"ts": int(r[0]), "v": float(r[1] or 0.0)} for r in rows]

            else:
                raise HTTPException(status_code=400, detail="unsupported intervals metric")

        return {
            "meter_id": meter_id,
            "metric": metric,
            "channel": channel,
            "start_ts": int(start_ts),
            "end_ts": int(end_ts),
            "step_s": int(step_s),
            "tz": tz,
            "points": points,
            "server_ts": int(time.time()),
        }


# =========================
# TARIFFS (MONTHLY)
# =========================

@app.get("/api/tariffs", dependencies=[Depends(require_key)])
def api_get_tariffs(
    meter_id: str = Query(...),
    channel: str = Query("T"),
    ym: Optional[str] = Query(default=None),
):
    ym_n = _normalize_ym(ym or _ym_now_utc())

    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT
                currency,
                price_active_kwh,
                price_ind_x1, price_ind_x2, price_ind_x3,
                price_cap_x1, price_cap_x2, price_cap_x3
            FROM tariffs_monthly
            WHERE meter_id = :m AND channel = :c AND ym = :ym
        """), {"m": meter_id, "c": channel, "ym": ym_n}).fetchone()

        if not row:
            return {
                "meter_id": meter_id,
                "channel": channel,
                "ym": ym_n,
                "currency": FIXED_CURRENCY,
                "price_active_kwh": 0.0,
                "price_ind_x1": 0.0, "price_ind_x2": 0.0, "price_ind_x3": 0.0,
                "price_cap_x1": 0.0, "price_cap_x2": 0.0, "price_cap_x3": 0.0,
            }

        return {
            "meter_id": meter_id,
            "channel": channel,
            "ym": ym_n,
            "currency": FIXED_CURRENCY,  # forzado
            "price_active_kwh": float(row[1] or 0.0),
            "price_ind_x1": float(row[2] or 0.0),
            "price_ind_x2": float(row[3] or 0.0),
            "price_ind_x3": float(row[4] or 0.0),
            "price_cap_x1": float(row[5] or 0.0),
            "price_cap_x2": float(row[6] or 0.0),
            "price_cap_x3": float(row[7] or 0.0),
        }


@app.put("/api/tariffs", dependencies=[Depends(require_key)])
def api_put_tariffs(
    payload: TariffPayload,
    meter_id: str = Query(...),
    channel: str = Query("T"),
    ym: Optional[str] = Query(default=None),
):
    ym_n = _normalize_ym(ym or _ym_now_utc())

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO tariffs_monthly (
                meter_id, channel, ym,
                currency,
                price_active_kwh,
                price_ind_x1, price_ind_x2, price_ind_x3,
                price_cap_x1, price_cap_x2, price_cap_x3
            ) VALUES (
                :m, :c, :ym,
                :currency,
                :pa,
                :i1, :i2, :i3,
                :c1, :c2, :c3
            )
            ON CONFLICT (meter_id, channel, ym) DO UPDATE SET
                currency = EXCLUDED.currency,
                price_active_kwh = EXCLUDED.price_active_kwh,
                price_ind_x1 = EXCLUDED.price_ind_x1,
                price_ind_x2 = EXCLUDED.price_ind_x2,
                price_ind_x3 = EXCLUDED.price_ind_x3,
                price_cap_x1 = EXCLUDED.price_cap_x1,
                price_cap_x2 = EXCLUDED.price_cap_x2,
                price_cap_x3 = EXCLUDED.price_cap_x3,
                updated_at = now()
        """), {
            "m": meter_id,
            "c": channel,
            "ym": ym_n,
            "currency": FIXED_CURRENCY,  # forzado
            "pa": float(payload.price_active_kwh or 0.0),
            "i1": float(payload.price_ind_x1 or 0.0),
            "i2": float(payload.price_ind_x2 or 0.0),
            "i3": float(payload.price_ind_x3 or 0.0),
            "c1": float(payload.price_cap_x1 or 0.0),
            "c2": float(payload.price_cap_x2 or 0.0),
            "c3": float(payload.price_cap_x3 or 0.0),
        })

    return {"ok": True, "meter_id": meter_id, "channel": channel, "ym": ym_n, "currency": FIXED_CURRENCY}


# =========================
# SIMPLE DASHBOARD (HTML)
# =========================

@app.get("/dashboard", response_class=HTMLResponse, dependencies=[Depends(require_key)])
def dashboard():
    if not DASHBOARD_PATH.exists():
        raise HTTPException(status_code=500, detail=f"dashboard.html not found at {DASHBOARD_PATH}")
    html = DASHBOARD_PATH.read_text(encoding="utf-8")
    return HTMLResponse(html)
