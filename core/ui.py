"""Visual helpers: colours, CSS, HTML cards and charts (Altair ships with Streamlit)."""
import altair as alt
import pandas as pd

LIME, GREEN, AMBER, RED, TEAL = "#9ef01a", "#2ecc71", "#f5a623", "#ff6b6b", "#4cc9f0"
LABEL_COLORS = {"HIGH": RED, "MEDIUM": AMBER, "LOW": GREEN}
LABEL_EMOJI = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}
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
.flow .icon{font-size:1.5rem}
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
            parts.append('<div class="arrow">➜</div>')
    return '<div class="flow">' + "".join(parts) + "</div>"


def option_card_html(o, recommended):
    cls = " rec" if recommended else ""
    ribbon = '<div class="ribbon">★ RECOMMENDED</div>' if recommended else ""
    verdict = "❌ Breaches promise" if o.breach else "✅ Meets promise"
    risk = f"<div>ML breach risk: {o.breach_risk:.0%}</div>" if o.breach_risk is not None else ""
    return (f'<div class="opt{cls}">{ribbon}<h4>{o.name}</h4><div class="big">Rs {o.cost:,.0f}</div>'
            f'<div>Residual delay: {o.delay_days:g} day(s)</div><div>Feasibility: {o.feasibility:.0%}</div>'
            f'<div>{verdict}</div>{risk}<div class="note">{o.note}</div></div>')


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
