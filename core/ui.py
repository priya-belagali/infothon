"""Visual helpers: colours, CSS, HTML cards and charts (Altair ships with Streamlit)."""
import altair as alt
import pandas as pd

LIME, GREEN, AMBER, RED, TEAL = "#9ef01a", "#2ecc71", "#f5a623", "#ff6b6b", "#4cc9f0"
LABEL_COLORS = {"HIGH": RED, "MEDIUM": AMBER, "LOW": GREEN}
_RISK_SCALE = alt.Scale(domain=["HIGH", "MEDIUM", "LOW"], range=[RED, AMBER, GREEN])

CSS = """<style>
.hero{background:linear-gradient(120deg,#06120d 0%,#0f3b24 55%,#1d6b2f 100%);border:1px solid #2ecc7155;
border-radius:16px;padding:22px 28px;margin-bottom:18px;box-shadow:0 6px 24px rgba(0,0,0,.25)}
.hero h1{color:#b6f55a;margin:0;font-size:2rem;padding:0}
.hero p{color:#d6f5dc;margin:4px 0 0 0;font-size:1rem}
.hero .tag{display:inline-block;margin-top:10px;margin-right:8px;padding:2px 12px;border-radius:999px;
background:#9ef01a22;border:1px solid #9ef01a88;color:#b6f55a;font-size:.75rem;letter-spacing:.06em}
[data-testid="stMetric"]{background:rgba(158,240,26,.07);border:1px solid rgba(158,240,26,.30);
border-left:5px solid #9ef01a;border-radius:12px;padding:14px 16px}
.badge{display:inline-block;padding:2px 12px;border-radius:999px;font-weight:700;font-size:.8rem;color:#06120d}
.flow{display:flex;align-items:stretch;gap:6px;margin:8px 0 18px 0;flex-wrap:wrap}
.flow .step{flex:1;min-width:130px;border-radius:12px;padding:10px 12px;background:rgba(158,240,26,.07);
border:1px solid rgba(158,240,26,.30)}
.flow .step.bad{border-color:#ff6b6b;background:rgba(255,107,107,.10)}
.flow .step.good{border-color:#9ef01a;background:rgba(158,240,26,.16);box-shadow:0 0 12px rgba(158,240,26,.25)}
.flow .arrow{align-self:center;color:#9ef01a;font-size:1.4rem;font-weight:700}
.flow .icon{display:inline-block;width:26px;height:26px;line-height:26px;border-radius:50%;background:#9ef01a;color:#06120d;font-weight:800;font-size:.85rem;text-align:center}
.flow .title{font-weight:700;margin-top:2px}
.flow .sub{font-size:.8rem;opacity:.8}
.opt{position:relative;border-radius:14px;padding:16px;border:1px solid rgba(158,240,26,.30);
background:rgba(158,240,26,.06);height:100%}
.opt.rec{border:2px solid #9ef01a;background:rgba(158,240,26,.14);box-shadow:0 0 16px rgba(158,240,26,.25)}
.opt h4{margin:0 0 4px 0}
.opt .big{font-size:1.6rem;font-weight:800;color:#9ef01a;margin-bottom:6px}
.opt .ribbon{display:inline-block;margin-bottom:6px;padding:1px 10px;border-radius:999px;
background:#9ef01a;color:#06120d;font-size:.7rem;font-weight:800;letter-spacing:.05em}
.opt .note{font-size:.8rem;opacity:.75;margin-top:8px;font-style:italic}
.ev{border-left:4px solid #4cc9f0;background:rgba(76,201,240,.08);border-radius:8px;padding:8px 12px;margin:6px 0}
.stButton>button{border-radius:10px;font-weight:700}
.rec-card{background:rgba(158,240,26,.07);border:1px solid rgba(158,240,26,.30);border-left:5px solid #9ef01a;
border-radius:12px;padding:14px 16px;margin-top:10px}
.rec-card .lbl{font-size:.85rem;opacity:.75}
.rec-card .val{font-size:1.35rem;font-weight:800;line-height:1.25;white-space:normal;overflow-wrap:anywhere}
.rec-card .sub{font-size:.85rem;opacity:.8;margin-top:4px;white-space:normal;overflow-wrap:anywhere}
@keyframes sail{0%{left:-340px;opacity:0}6%{opacity:1}94%{opacity:1;visibility:visible}
100%{left:calc(100vw + 20px);opacity:0;visibility:hidden}}
.ship-fly{position:fixed;top:30vh;left:-340px;width:300px;z-index:99999;pointer-events:none;
animation:sail 4.5s ease-in-out forwards;filter:drop-shadow(0 0 10px rgba(158,240,26,.55));text-align:center}
.ship-fly img{width:100%;display:block}
.ship-fly .cap{color:#b6f55a;font-weight:800;font-size:.95rem;text-shadow:0 2px 6px #000}
.card-detail-box{background:rgba(158,240,26,.05);border:1px solid rgba(158,240,26,.35);
border-radius:14px;padding:18px 20px;margin:12px 0 20px 0;box-shadow:0 4px 18px rgba(0,0,0,.20)}
.card-detail-box h3{margin-top:0;margin-bottom:8px}
</style>"""


# ---------------- HTML pieces ----------------
def inject_css():
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)


def badge(label):
    return f'<span class="badge" style="background:{LABEL_COLORS.get(label, TEAL)}">{label}</span>'


def hero(title, subtitle, tags=()):
    import streamlit as st
    tag_html = "".join(f'<span class="tag">{t}</span>' for t in tags)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{tag_html}</div>',
                unsafe_allow_html=True)


def flow_html(steps):
    """steps = [(icon, title, subtitle, kind)] where kind is '', 'bad' or 'good'."""
    parts = []
    for i, (icon, title, sub, kind) in enumerate(steps):
        parts.append(f'<div class="step {kind}"><div class="icon">{icon}</div>'
                     f'<div class="title">{title}</div><div class="sub">{sub}</div></div>')
        if i < len(steps) - 1:
            parts.append('<div class="arrow">&rarr;</div>')
    return '<div class="flow">' + "".join(parts) + "</div>"


def option_card_html(o, recommended):
    cls = " rec" if recommended else ""
    ribbon = '<div class="ribbon">RECOMMENDED</div>' if recommended else ""
    verdict = (f'<div style="font-weight:700;color:{RED}">Breaches promise</div>' if o.breach
               else f'<div style="font-weight:700;color:{GREEN}">Meets promise</div>')
    risk = f"<div>ML breach risk: {o.breach_risk:.0%}</div>" if o.breach_risk is not None else ""
    return (f'<div class="opt{cls}">{ribbon}<h4>{o.name}</h4><div class="big">Rs {o.cost:,.0f}</div>'
            f'<div>Residual delay: {o.delay_days:g} day(s)</div><div>Feasibility: {o.feasibility:.0%}</div>'
            f'{verdict}{risk}<div class="note">{o.note}</div></div>')


def rec_card_html(label, text, sub=""):
    """Card that shows the FULL text (wraps onto several lines; st.metric cuts long values with '...')."""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="rec-card"><div class="lbl">{label}</div><div class="val">{text}</div>{sub_html}</div>'


def evidence_html(lines):
    return "".join(f'<div class="ev">{i}. {line}</div>' for i, line in enumerate(lines, 1))


# ---------------- charts ----------------
def risk_bar_chart(df, top=12):
    d = df.head(top)[["order_id", "risk_score", "risk_label"]]
    return (alt.Chart(d).mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
            .encode(y=alt.Y("order_id:N", sort="-x", title=None),
                    x=alt.X("risk_score:Q", title="Risk score (0-100)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("risk_label:N", scale=_RISK_SCALE, legend=alt.Legend(title="Risk")),
                    tooltip=["order_id", "risk_score", "risk_label"])
            .properties(width="container", height=max(200, 26 * len(d))))


def risk_donut(df):
    c = df.groupby("risk_label").size().reset_index(name="count")
    return (alt.Chart(c).mark_arc(innerRadius=55)
            .encode(theta="count:Q",
                    color=alt.Color("risk_label:N", scale=_RISK_SCALE, legend=alt.Legend(title="Risk")),
                    tooltip=["risk_label", "count"])
            .properties(width="container", height=260))


def factor_chart(factors):
    d = pd.DataFrame(factors, columns=["factor", "value"])
    d["effect"] = d["value"].apply(lambda v: "Raises risk" if v >= 0 else "Lowers risk")
    return (alt.Chart(d).mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
            .encode(y=alt.Y("factor:N", sort="-x", title=None),
                    x=alt.X("value:Q", title="Contribution to late-shipment prediction"),
                    color=alt.Color("effect:N", legend=None,
                                    scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[RED, GREEN])),
                    tooltip=["factor", "value"])
            .properties(width="container", height=140))


def sim_chart(base, sim):
    rows = []
    for metric, key in [("Orders affected", "orders_affected"), ("Delivery breaches", "breaches")]:
        rows += [{"metric": metric, "scenario": "Baseline", "value": base[key]},
                 {"metric": metric, "scenario": "Simulated", "value": sim[key]}]
    d = pd.DataFrame(rows)
    return (alt.Chart(d).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(x=alt.X("scenario:N", title=None, sort=["Baseline", "Simulated"]),
                    y=alt.Y("value:Q", title=None),
                    color=alt.Color("scenario:N", legend=None,
                                    scale=alt.Scale(domain=["Baseline", "Simulated"], range=[TEAL, RED])),
                    column=alt.Column("metric:N", title=None),
                    tooltip=["metric", "scenario", "value"])
            .properties(width=170, height=240))


def gust_chart(info, port_names, limit):
    rows = [{"port": port_names.get(p, p), "gust": v["gust_max"]} for p, v in info.items()
            if v.get("gust_max") is not None]
    if not rows:
        return None
    d = pd.DataFrame(rows)
    d["status"] = d["gust"].apply(lambda g: "Severe" if g >= limit else "Normal")
    bars = (alt.Chart(d).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(x=alt.X("port:N", title=None), y=alt.Y("gust:Q", title="Max wind gust (km/h)"),
                    color=alt.Color("status:N", legend=alt.Legend(title="Status"),
                                    scale=alt.Scale(domain=["Severe", "Normal"], range=[RED, GREEN])),
                    tooltip=["port", "gust"]))
    rule = (alt.Chart(pd.DataFrame({"limit": [limit]}))
            .mark_rule(strokeDash=[6, 4], color=AMBER, size=2).encode(y="limit:Q"))
    return (bars + rule).properties(width="container", height=280)


def model_chart(metrics):
    rows = []
    for name, key in [("Simple rule", "rule"), ("ML model", "ml")]:
        for metric in ("precision", "recall"):
            rows.append({"model": name, "metric": metric.title(), "value": float(metrics[key][metric])})
    d = pd.DataFrame(rows)
    return (alt.Chart(d).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(x=alt.X("model:N", title=None, sort=["Simple rule", "ML model"]),
                    y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("model:N", legend=None,
                                    scale=alt.Scale(domain=["Simple rule", "ML model"], range=[TEAL, LIME])),
                    column=alt.Column("metric:N", title=None),
                    tooltip=["model", "metric", alt.Tooltip("value:Q", format=".2f")])
            .properties(width=170, height=240))


def port_map_df(ports, coords, delay_map):
    rows = []
    for _, p in ports.iterrows():
        lat, lon = coords[p.port_id]
        hit = p.port_id in delay_map
        rows.append({"port": p.port_name, "lat": lat, "lon": lon,
                     "color": RED if hit else GREEN, "size": 90000 if hit else 45000})
    return pd.DataFrame(rows)


# ---------------- risk gauge (pure SVG, no extra libraries) ----------------
import math

RISK_ZONES = [(0.0, 0.2, GREEN), (0.2, 0.5, AMBER), (0.5, 1.0, RED)]   # matches risk_label thresholds
PROB_ZONES = [(0.0, 0.3, GREEN), (0.3, 0.6, AMBER), (0.6, 1.0, RED)]   # illustrative bands


def _pt(cx, cy, r, f):
    ang = math.pi * (1 - f)
    return cx + r * math.cos(ang), cy - r * math.sin(ang)


def _arc(cx, cy, r, f1, f2):
    x1, y1 = _pt(cx, cy, r, f1)
    x2, y2 = _pt(cx, cy, r, f2)
    return f"M {x1:.1f} {y1:.1f} A {r} {r} 0 0 1 {x2:.1f} {y2:.1f}"


def gauge_html(title, value_text, fraction, zones, caption=""):
    """Half-circle dial. fraction is 0-1; zones = [(start, end, colour)]."""
    fraction = max(0.0, min(1.0, float(fraction)))
    cx, cy, r = 110, 105, 85
    paths = "".join(f'<path d="{_arc(cx, cy, r, a, b)}" stroke="{c}" stroke-width="16" fill="none" opacity="0.9"/>'
                    for a, b, c in zones)
    color = next((c for a, b, c in zones if fraction < b), zones[-1][2])
    nx, ny = _pt(cx, cy, r - 24, fraction)
    svg = (f'<svg viewBox="0 0 220 138" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:300px">'
           f'{paths}'
           f'<text x="{cx - r}" y="{cy + 18}" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.6">0</text>'
           f'<text x="{cx + r}" y="{cy + 18}" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.6">max</text>'
           f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{color}" stroke-width="4" stroke-linecap="round"/>'
           f'<circle cx="{cx}" cy="{cy}" r="7" fill="{color}"/>'
           f'<text x="{cx}" y="{cy + 30}" text-anchor="middle" font-size="26" font-weight="800" fill="{color}">{value_text}</text>'
           f'</svg>')
    return (f'<div style="text-align:center"><div style="font-weight:700;margin-bottom:2px">{title}</div>{svg}'
            f'<div style="font-size:.8rem;opacity:.75">{caption}</div></div>')


# ---------------- supply-route map (pydeck ships with Streamlit) ----------------
FACTORY = {"name": "Factory (demo location)", "lat": 12.97, "lon": 77.59}


def route_map_frames(ports, routes, shipments, coords, delay_map):
    """Two tables: one arc per port -> factory, and one dot per port plus the factory."""
    per_port = shipments.merge(routes[["route_id", "port_id"]], on="route_id").groupby("port_id").size()
    arcs, nodes = [], []
    for _, p in ports.iterrows():
        lat, lon = coords[p.port_id]
        hit = p.port_id in delay_map
        color = [255, 107, 107] if hit else [46, 204, 113]
        n = int(per_port.get(p.port_id, 0))
        status = f"Disrupted for {delay_map[p.port_id]} day(s)" if hit else "Operating normally"
        arcs.append({"port": p.port_name, "slon": lon, "slat": lat, "flon": FACTORY["lon"],
                     "flat": FACTORY["lat"], "color": color, "width": 2 + n,
                     "status": status, "shipments": n})
        nodes.append({"port": p.port_name, "lon": lon, "lat": lat, "color": color,
                      "status": status, "shipments": n})
    nodes.append({"port": FACTORY["name"], "lon": FACTORY["lon"], "lat": FACTORY["lat"],
                  "color": [76, 201, 240], "status": "Receives all shipments", "shipments": "-"})
    return pd.DataFrame(arcs), pd.DataFrame(nodes)


def route_map(ports, routes, shipments, coords, delay_map, dark=True):
    import pydeck as pdk
    arcs, nodes = route_map_frames(ports, routes, shipments, coords, delay_map)
    arc_layer = pdk.Layer("ArcLayer", data=arcs,
                          get_source_position="[slon, slat]", get_target_position="[flon, flat]",
                          get_source_color="color", get_target_color="color",
                          get_width="width", pickable=True, auto_highlight=True)
    dot_layer = pdk.Layer("ScatterplotLayer", data=nodes, get_position="[lon, lat]",
                          get_fill_color="color", get_radius=22000, radius_min_pixels=7,
                          pickable=True)
    view = pdk.ViewState(latitude=17.5, longitude=79.0, zoom=3.8, pitch=35)
    map_style = "dark" if dark else "light"
    if dark:
        tooltip = {"html": "<b>{port}</b><br/>{status}<br/>Shipments in transit: {shipments}",
                   "style": {"backgroundColor": "#0c2118", "color": "#e6f4ea", "border": "1px solid #2ecc7155"}}
    else:
        tooltip = {"html": "<b>{port}</b><br/>{status}<br/>Shipments in transit: {shipments}",
                   "style": {"backgroundColor": "#ffffff", "color": "#10261a", "border": "1px solid #2e8b5755",
                             "boxShadow": "0 2px 8px rgba(0,0,0,0.15)"}}
    try:
        return pdk.Deck(layers=[arc_layer, dot_layer], initial_view_state=view, tooltip=tooltip,
                        map_provider="carto", map_style=map_style)
    except TypeError:       # older pydeck without map_provider
        return pdk.Deck(layers=[arc_layer, dot_layer], initial_view_state=view, tooltip=tooltip,
                        map_style=map_style)


# ---------------- approval animation ----------------
import base64
from functools import lru_cache


@lru_cache(maxsize=4)
def _gif_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def ship_html(path, caption=""):
    """Ship sails left -> right across the screen (no box), then disappears.
    Uses a transparent GIF; position:fixed so it passes over the page."""
    cap = f'<div class="cap">{caption}</div>' if caption else ""
    return (f'<div class="ship-fly"><img alt="Container ship" '
            f'src="data:image/gif;base64,{_gif_b64(str(path))}"/>{cap}</div>')


# ---------------- dark / light mode ----------------
DARK_THEME = {"theme.base": "dark", "theme.primaryColor": "#9ef01a", "theme.backgroundColor": "#06120d",
              "theme.secondaryBackgroundColor": "#0c2118", "theme.textColor": "#e6f4ea"}
LIGHT_THEME = {"theme.base": "light", "theme.primaryColor": "#2e8b57", "theme.backgroundColor": "#f6fbf7",
               "theme.secondaryBackgroundColor": "#e3f1e6", "theme.textColor": "#10261a"}


def theme_needs_change(dark):
    import streamlit as st
    return (st.get_option("theme.base") == "dark") != bool(dark)


def apply_theme(dark):
    """Switch the whole app theme. Uses Streamlit's internal config setter, so it is wrapped in a
    try/except: returns True if it worked (caller then reruns), False if this version refuses."""
    import streamlit as st
    try:
        for key, value in (DARK_THEME if dark else LIGHT_THEME).items():
            st._config.set_option(key, value)
        return True
    except Exception:
        return False
