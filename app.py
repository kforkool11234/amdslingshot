"""
IoT Smart Meter Simulator
=========================
Reads rows from smart_grid_dataset.csv and publishes a structured JSON
payload to MQTT topic  vpp/telemetry/main_bus  every 2 seconds.

Run:
    python app.py

Requires:
    pip install flask paho-mqtt pandas
    A Mosquitto broker listening on localhost:1883
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone

import pandas as pd
import paho.mqtt.client as mqtt
from flask import Flask, jsonify, render_template

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
MQTT_BROKER   = "localhost"
MQTT_PORT     = 1883
MQTT_TOPIC    = "vpp/telemetry/main_bus"
ASSET_ID      = "mumbai_campus_node_01"
CSV_PATH      = os.path.join(os.path.dirname(__file__), "smart_grid_dataset.csv")
PUBLISH_INTERVAL = 2          # seconds between rows
NOMINAL_FREQ_HZ  = 50.0       # fixed; not in CSV

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Shared state  (thread-safe via a lock)
# ──────────────────────────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_state = {
    "last_payload": None,   # dict — last JSON payload sent
    "row_index":    0,      # current row pointer
    "total_rows":   0,      # total rows in CSV
    "mqtt_connected": False,
    "publish_count":  0,
    "last_error":     None,
}

# ──────────────────────────────────────────────────────────────────────────────
# Load CSV once at startup
# ──────────────────────────────────────────────────────────────────────────────
log.info("Loading CSV: %s", CSV_PATH)
_df = pd.read_csv(CSV_PATH)

# Normalise column names (strip whitespace)
_df.columns = [c.strip() for c in _df.columns]

# Expected columns — map internal key → CSV column name
COL_MAP = {
    "timestamp":    "Timestamp",
    "voltage":      "Voltage (V)",
    "current":      "Current (A)",
    "power_cons":   "Power Consumption (kW)",
    "react_power":  "Reactive Power (kVAR)",
    "power_factor": "Power Factor",
    "solar":        "Solar Power (kW)",
    "wind":         "Wind Power (kW)",
    "grid_supply":  "Grid Supply (kW)",
    "v_fluct":      "Voltage Fluctuation (%)",
}

# Verify columns exist
for key, col in COL_MAP.items():
    if col not in _df.columns:
        log.warning("Column '%s' not found in CSV — will default to 0.0", col)

with _state_lock:
    _state["total_rows"] = len(_df)

log.info("CSV loaded: %d rows, %d columns", len(_df), len(_df.columns))


# ──────────────────────────────────────────────────────────────────────────────
# Helper: safely read a cell value (float)
# ──────────────────────────────────────────────────────────────────────────────
def _safe_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        val = row[col]
        return round(float(val), 6) if pd.notna(val) else default
    except (KeyError, ValueError, TypeError):
        return default


def _safe_str(row: pd.Series, col: str, default: str = "") -> str:
    try:
        val = row[col]
        return str(val) if pd.notna(val) else default
    except (KeyError, ValueError):
        return default


# ──────────────────────────────────────────────────────────────────────────────
# Build JSON payload from a CSV row
# ──────────────────────────────────────────────────────────────────────────────
def build_payload(row: pd.Series) -> dict:
    """Construct the full telemetry payload from a single CSV row."""
    ts_raw = _safe_str(row, COL_MAP["timestamp"])
    # Use CSV timestamp if available, otherwise fall back to UTC now
    if ts_raw:
        timestamp = ts_raw
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    voltage     = _safe_float(row, COL_MAP["voltage"])
    current_a   = _safe_float(row, COL_MAP["current"])
    power_cons  = _safe_float(row, COL_MAP["power_cons"])
    react_pwr   = _safe_float(row, COL_MAP["react_power"])
    pwr_factor  = _safe_float(row, COL_MAP["power_factor"])
    solar       = _safe_float(row, COL_MAP["solar"])
    wind        = _safe_float(row, COL_MAP["wind"])
    grid_supply = _safe_float(row, COL_MAP["grid_supply"])
    v_fluct     = _safe_float(row, COL_MAP["v_fluct"])

    payload = {
        "header": {
            "asset_id":  ASSET_ID,
            "timestamp": timestamp,
            "status":    "online",
        },
        "grid_state": {
            "voltage_v":           round(voltage,    4),
            "current_a":           round(current_a,  4),
            "frequency_hz":        NOMINAL_FREQ_HZ,
            "power_factor":        round(pwr_factor, 6),
            "voltage_fluctuation": round(v_fluct,    6),
        },
        "energy_flow": {
            "load_kw":            round(power_cons,  4),
            "solar_gen_kw":       round(solar,       4),
            "wind_gen_kw":        round(wind,        4),
            "grid_supply_kw":     round(grid_supply, 4),
            "reactive_power_kvar":round(react_pwr,   4),
        },
    }
    return payload


# ──────────────────────────────────────────────────────────────────────────────
# MQTT client setup
# ──────────────────────────────────────────────────────────────────────────────
def _on_connect(client, userdata, flags, reason_code, properties=None):
    """Callback for successful (or failed) broker connection."""
    if reason_code == 0:
        log.info("MQTT connected to %s:%d", MQTT_BROKER, MQTT_PORT)
        with _state_lock:
            _state["mqtt_connected"] = True
            _state["last_error"] = None
    else:
        log.error("MQTT connect failed — reason code %s", reason_code)
        with _state_lock:
            _state["mqtt_connected"] = False
            _state["last_error"] = f"Connect failed: rc={reason_code}"


def _on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
    log.warning("MQTT disconnected (rc=%s) — will retry automatically", reason_code)
    with _state_lock:
        _state["mqtt_connected"] = False


def _on_publish(client, userdata, mid, reason_code=None, properties=None):
    with _state_lock:
        _state["publish_count"] += 1


mqtt_client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id=ASSET_ID,
    clean_session=True,
)
mqtt_client.on_connect    = _on_connect
mqtt_client.on_disconnect = _on_disconnect
mqtt_client.on_publish    = _on_publish

# Enable automatic reconnect (every 1 s, up to 60 s back-off)
mqtt_client.reconnect_delay_set(min_delay=1, max_delay=60)


def _connect_mqtt():
    """Initial blocking connect; retries every 5 s on failure."""
    while True:
        try:
            mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            mqtt_client.loop_start()   # start background network thread
            break
        except OSError as exc:
            log.error("Cannot reach broker at %s:%d — %s. Retrying in 5 s…",
                      MQTT_BROKER, MQTT_PORT, exc)
            with _state_lock:
                _state["last_error"] = str(exc)
            time.sleep(5)


# ──────────────────────────────────────────────────────────────────────────────
# Background publisher thread
# ──────────────────────────────────────────────────────────────────────────────
def publisher_thread():
    """Iterate through CSV rows endlessly, publishing one row every 2 seconds."""
    total = len(_df)
    idx   = 0

    while True:
        row     = _df.iloc[idx]
        payload = build_payload(row)
        json_str = json.dumps(payload, indent=2)

        with _state_lock:
            connected = _state["mqtt_connected"]

        if connected:
            result = mqtt_client.publish(MQTT_TOPIC, json_str, qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                log.info("Published row %d/%d → %s", idx + 1, total, MQTT_TOPIC)
            else:
                log.warning("Publish failed (rc=%d) for row %d", result.rc, idx)
        else:
            log.warning("Broker not connected — skipping row %d", idx)

        with _state_lock:
            _state["last_payload"] = payload
            _state["row_index"]    = idx + 1

        idx = (idx + 1) % total   # wrap around
        time.sleep(PUBLISH_INTERVAL)


# ──────────────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/status")
def status():
    """Human-readable debug page showing the last published payload."""
    with _state_lock:
        payload    = _state["last_payload"]
        row_index  = _state["row_index"]
        total_rows = _state["total_rows"]

    raw_json = json.dumps(payload, indent=2) if payload else None
    return render_template(
        "status.html",
        payload=payload,
        raw_json=raw_json,
        row_index=row_index,
        total_rows=total_rows,
    )


@app.route("/api/status")
def api_status():
    """JSON endpoint — returns the full last payload plus metadata."""
    with _state_lock:
        data = {
            "last_payload":   _state["last_payload"],
            "row_index":      _state["row_index"],
            "total_rows":     _state["total_rows"],
            "mqtt_connected": _state["mqtt_connected"],
            "publish_count":  _state["publish_count"],
            "last_error":     _state["last_error"],
            "mqtt_topic":     MQTT_TOPIC,
            "asset_id":       ASSET_ID,
        }
    return jsonify(data)


@app.route("/")
def index():
    return (
        "<h2>IoT Smart Meter Simulator</h2>"
        "<ul>"
        "<li><a href='/status'>/status</a> — Live telemetry dashboard</li>"
        "<li><a href='/api/status'>/api/status</a> — Raw JSON API</li>"
        "</ul>"
    ), 200


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Connect MQTT (blocks until first successful connect or broker available)
    log.info("Connecting to MQTT broker %s:%d …", MQTT_BROKER, MQTT_PORT)
    _connect_mqtt()

    # 2. Start the CSV publisher as a daemon thread
    t = threading.Thread(target=publisher_thread, name="csv-publisher", daemon=True)
    t.start()
    log.info("Publisher thread started — publishing every %ds", PUBLISH_INTERVAL)

    # 3. Start Flask dev server
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
