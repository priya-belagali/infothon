"""Reads LIVE weather forecasts (Open-Meteo: free, no API key) and turns severe weather
into port disruptions that feed the same pipeline as a manual port closure.

ASSUMPTIONS (tune freely): a day counts as 'severe' if wind gusts >= 60 km/h
or rainfall >= 60 mm. Number of severe days in the forecast window = disruption days."""
import json
import urllib.parse
import urllib.request

import pandas as pd

# Approximate port coordinates (latitude, longitude)
PORT_COORDS = {
    "P1": (13.08, 80.29),   # Chennai
    "P2": (18.95, 72.95),   # Mumbai (JNPT)
    "P3": (22.74, 69.70),   # Mundra
    "P4": (17.69, 83.30),   # Visakhapatnam
}
GUST_LIMIT_KMH = 60
RAIN_LIMIT_MM = 60
FORECAST_DAYS = 5
URL = "https://api.open-meteo.com/v1/forecast"


def _get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def fetch_port_forecast(port_id):
    """Daily forecast lists for one port: dates, wind gusts (km/h), rainfall (mm)."""
    lat, lon = PORT_COORDS[port_id]
    query = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "daily": "wind_gusts_10m_max,precipitation_sum",
        "timezone": "auto", "forecast_days": FORECAST_DAYS,
    })
    daily = _get_json(f"{URL}?{query}")["daily"]
    return {"dates": daily["time"], "gusts": daily["wind_gusts_10m_max"],
            "rain": daily["precipitation_sum"]}


def analyze_forecast(fc):
    """Turn a forecast into a summary: worst gust, worst rain, and number of severe days."""
    gusts = [g or 0 for g in fc["gusts"]]
    rain = [r or 0 for r in fc["rain"]]
    severe = sum(1 for g, r in zip(gusts, rain) if g >= GUST_LIMIT_KMH or r >= RAIN_LIMIT_MM)
    return {"gust_max": round(max(gusts), 1), "rain_max": round(max(rain), 1),
            "severe_days": severe, "source": "Open-Meteo forecast", "error": None}


def load_weather(fetch=fetch_port_forecast):
    """{port_id: summary} for every port. A failed port gets an error note, never a crash."""
    out = {}
    for port_id in PORT_COORDS:
        try:
            out[port_id] = analyze_forecast(fetch(port_id))
        except Exception as exc:  # no internet, API down, etc.
            out[port_id] = {"gust_max": None, "rain_max": None, "severe_days": 0,
                            "source": "unavailable", "error": str(exc)[:120]}
    return out


def with_demo_storm(info, port_id="P1", days=3, gust=95.0):
    """Pretend a cyclone hits one port (real weather is usually calm, so demos need this)."""
    info = {k: dict(v) for k, v in info.items()}
    for pid in PORT_COORDS:
        info.setdefault(pid, {"gust_max": None, "rain_max": None, "severe_days": 0,
                              "source": "n/a", "error": None})
    info[port_id].update({"gust_max": gust, "severe_days": max(days, info[port_id]["severe_days"]),
                          "source": "DEMO storm (not real)", "error": None})
    return info


def to_delay_map(info):
    return {pid: v["severe_days"] for pid, v in info.items() if v.get("severe_days", 0) > 0}


def merge_delays(*maps):
    """Combine several {port_id: days} maps, keeping the longest delay per port."""
    out = {}
    for m in maps:
        for pid, d in m.items():
            out[pid] = max(out.get(pid, 0), d)
    return out


def weather_table(info, port_names):
    rows = [{"Port": port_names.get(pid, pid), "Max gust (km/h)": v["gust_max"],
             "Max rain (mm/day)": v["rain_max"], "Severe days": v["severe_days"],
             "Source": v["source"] if not v["error"] else f"unavailable ({v['error']})"}
            for pid, v in info.items()]
    return pd.DataFrame(rows)
