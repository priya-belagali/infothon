"""The planner's screen. Run with:  python -m streamlit run app/streamlit_app.py"""
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from agent import pipeline
from core import analysis, decisions, model, simulator, ui, weather
from core.db import DB_PATH

st.set_page_config(page_title="Supply Chain Disruption Agent", layout="wide")
ui.inject_css()
SHIP_GIF = Path(__file__).resolve().parent / "Container_ship_transparent.gif"

# On a fresh server there is no database yet, so build the synthetic data once.
if not DB_PATH.exists():
    root = Path(__file__).resolve().parent.parent
    subprocess.run([sys.executable, str(root / "data" / "generate_synthetic.py")], check=True)
decisions.init_table()


@st.cache_resource
def get_model():
    return model.train_model()          # trains once, then reused


@st.cache_data(ttl=1800, show_spinner="Reading live weather...")
def get_weather():
    return weather.load_weather()          # cached 30 minutes so we don't hammer the API


mdl = get_model()
net = analysis.load_network()
port_names = net["ports"].set_index("port_id")["port_name"].to_dict()

# ---------------- sidebar ----------------
P_DASH, P_ORDER, P_SIM = "Dashboard", "Order detail", "What-if simulator"
P_WEATHER, P_LOG, P_MODEL = "Weather", "Decision log", "Model check"

st.sidebar.markdown("## Disruption Agent")
st.sidebar.caption("Predict · Trace · Approve")

# ---- dark / light mode switch ----
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = st.get_option("theme.base") != "light"
_toggle = getattr(st.sidebar, "toggle", st.sidebar.checkbox)
dark_mode = _toggle("Dark mode", key="dark_mode")
if ui.theme_needs_change(dark_mode) and st.session_state.get("_theme_tried") != dark_mode:
    st.session_state["_theme_tried"] = dark_mode      # try once per switch, never loop
    if ui.apply_theme(dark_mode):
        (getattr(st, "rerun", None) or st.experimental_rerun)()
if ui.theme_needs_change(dark_mode):
    st.sidebar.caption("Theme switch isn't available in this version - use the menu (three dots, top right) > Settings > Theme.")

page = st.sidebar.radio("Go to", [P_DASH, P_ORDER, P_SIM, P_WEATHER, P_LOG, P_MODEL])
stale_demo = st.sidebar.checkbox("Simulate stale data (+9 hours)")
extra_hours = 9 if stale_demo else 0

st.sidebar.markdown("**Disruption sources**")
use_db_event = st.sidebar.checkbox("Port closure from database (demo)", value=True)
use_weather = st.sidebar.checkbox("Live weather (Open-Meteo)")
demo_storm = st.sidebar.checkbox("Demo: inject a storm at Chennai")

db_map = analysis.live_delay_map(net) if use_db_event else {}
raw_weather = get_weather() if use_weather else {}
weather_failed = [pid for pid, v in raw_weather.items() if v.get("error")]
weather_info = raw_weather
if demo_storm:
    weather_info = weather.with_demo_storm(weather_info, "P1", 3)
weather_map = weather.to_delay_map(weather_info)
delay_map = weather.merge_delays(db_map, weather_map)

st.sidebar.markdown("**Live disruptions**")
if not delay_map:
    st.sidebar.success("None right now")
for pid, d in db_map.items():
    st.sidebar.error(f"{port_names[pid]}: port closed {d} day(s)")
for pid, d in weather_map.items():
    st.sidebar.warning(f"{port_names[pid]}: severe weather {d} day(s)")



def toast_on_change(key, value, on_true, on_false=None):
    """Pop a toast only when a sidebar switch CHANGES (not on every rerun)."""
    prev = st.session_state.get(f"_prev_{key}")
    if prev is not None and prev != value:
        msg = on_true if value else on_false
        if msg:
            st.toast(msg)
    st.session_state[f"_prev_{key}"] = value


toast_on_change("stale", stale_demo,
                "Safe mode ON: shipment data is 9 hours old.", "Shipment data is fresh again.")
toast_on_change("storm", demo_storm, "Demo storm injected at Chennai.", "Demo storm cleared.")
toast_on_change("dbevent", use_db_event,
                "Database port closure included.", "Database port closure removed.")
toast_on_change("weather", use_weather,
                f"Weather unavailable for {len(weather_failed)} port(s). Check your internet connection."
                if weather_failed else "Live weather loaded.",
                "Live weather switched off.")

df = analysis.assess(net, delay_map, mdl, extra_hours)


@st.dialog("Confirm Action Approval", width="small")
def confirm_approve_dialog(order_id, chosen, evidence_txt):
    st.markdown("### Are you sure you want to approve?")
    st.write(f"You are approving **{chosen.name}** for order **{order_id}**.")
    st.markdown(f"- **Est. Cost:** Rs {chosen.cost:,.0f}  \n- **Residual Delay:** {chosen.delay_days:g} day(s)")
    st.warning("**Note:** Once submitted, this decision cannot be changed.")

    col1, col2 = st.columns(2)
    if col1.button("Yes, Approve", type="primary", use_container_width=True):
        decisions.log_decision(
            order_id, chosen.name, "APPROVED by planner",
            f"{chosen.name}: est. Rs {chosen.cost:,.0f}, residual delay {chosen.delay_days:g}d",
            evidence_txt, chosen.cost
        )
        st.session_state[f"_just_approved_{order_id}"] = True
        st.toast(f"Decision logged: {chosen.name} for {order_id}")
        st.rerun()
    if col2.button("Cancel", use_container_width=True):
        st.rerun()


@st.dialog("Confirm Action Rejection", width="small")
def confirm_reject_dialog(order_id, chosen, evidence_txt):
    st.markdown("### Are you sure you want to reject?")
    st.write(f"You are rejecting the recommended action **{chosen.name}** for order **{order_id}**.")
    st.warning("**Note:** Once submitted, this decision cannot be changed.")

    col1, col2 = st.columns(2)
    if col1.button("Yes, Reject", type="primary", use_container_width=True):
        decisions.log_decision(
            order_id, chosen.name, "REJECTED by planner",
            "Planner rejected the recommendation", evidence_txt, chosen.cost
        )
        st.session_state[f"_just_rejected_{order_id}"] = True
        st.toast(f"Rejection logged for {order_id}")
        st.rerun()
    if col2.button("Cancel", use_container_width=True):
        st.rerun()


# ---------------- pages ----------------
if page == P_DASH:
    ui.hero("Supply Chain Disruption Agent",
            "Predict the delay. Trace the impact. Choose the next move.",
            tags=["SYNTHETIC DEMO DATA", "HUMAN-APPROVED ACTIONS ONLY"])

    if "dash_card_selected" not in st.session_state:
        st.session_state["dash_card_selected"] = None

    selected_card = st.session_state.get("dash_card_selected")
    high_risk_count = int((df.risk_label == "HIGH").sum()) if not df.empty else 0
    breach_count = int(df.breach.sum()) if not df.empty else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        is_active = (selected_card == "disruptions")
        with st.container(border=True):
            st.metric("Active disruptions", len(delay_map),
                      help="Click to inspect disrupted ports and weather conditions")
            if st.button("Hide details" if is_active else "View details", key="btn_card_disruptions",
                         type="primary" if is_active else "secondary", use_container_width=True):
                st.session_state["dash_card_selected"] = None if is_active else "disruptions"
                st.rerun()

    with k2:
        is_active = (selected_card == "orders")
        with st.container(border=True):
            st.metric("Orders affected", len(df),
                      help="Click to inspect all affected customer orders")
            if st.button("Hide details" if is_active else "View details", key="btn_card_orders",
                         type="primary" if is_active else "secondary", use_container_width=True):
                st.session_state["dash_card_selected"] = None if is_active else "orders"
                st.rerun()

    with k3:
        is_active = (selected_card == "high_risk")
        with st.container(border=True):
            st.metric("High risk", high_risk_count,
                      help="Click to inspect priority high-risk orders")
            if st.button("Hide details" if is_active else "View details", key="btn_card_high_risk",
                         type="primary" if is_active else "secondary", use_container_width=True):
                st.session_state["dash_card_selected"] = None if is_active else "high_risk"
                st.rerun()

    with k4:
        is_active = (selected_card == "breaches")
        with st.container(border=True):
            st.metric("Potential breaches", breach_count,
                      help="Click to inspect orders exceeding buffer days")
            if st.button("Hide details" if is_active else "View details", key="btn_card_breaches",
                         type="primary" if is_active else "secondary", use_container_width=True):
                st.session_state["dash_card_selected"] = None if is_active else "breaches"
                st.rerun()

    # ---- Interactive Card Details Section ----
    if selected_card is not None:
        with st.container(border=True):
            d_hdr, d_cls = st.columns([6, 1])
            with d_cls:
                if st.button("Close", key="btn_close_dash_card_detail", use_container_width=True):
                    st.session_state["dash_card_selected"] = None
                    st.rerun()

            if selected_card == "disruptions":
                with d_hdr:
                    st.markdown("### Active Disruptions Details")
                    st.caption("Live logistics ports experiencing closures or severe weather.")
                if not delay_map:
                    st.success("No active disruptions. All supply lanes are operating normally.")
                else:
                    per_port = net["shipments"].merge(net["routes"][["route_id", "port_id"]], on="route_id").groupby("port_id").size()
                    d_rows = []
                    for pid, days in delay_map.items():
                        causes = []
                        if pid in db_map:
                            causes.append(f"Port closure event ({db_map[pid]}d)")
                        if pid in weather_map:
                            w = weather_info.get(pid, {})
                            gust = f"{w.get('gust_max', '-')} km/h gust" if w.get("gust_max") else ""
                            rain = f"{w.get('rain_max', '-')} mm rain" if w.get("rain_max") else ""
                            causes.append(f"Severe weather ({weather_map[pid]}d: {', '.join(filter(None, [gust, rain]))})")
                        orders_linked = len(df[df.port_id == pid]) if not df.empty else 0
                        coords = weather.PORT_COORDS.get(pid, (None, None))
                        d_rows.append({
                            "Port ID": pid,
                            "Port Name": port_names.get(pid, pid),
                            "Total Delay": f"{days} day(s)",
                            "Root Cause": " & ".join(causes) if causes else "Manual override",
                            "In-Transit Shipments": int(per_port.get(pid, 0)),
                            "Exposed Orders": orders_linked,
                            "Coordinates": f"{coords[0]}, {coords[1]}" if coords[0] else "-",
                        })
                    st.dataframe(pd.DataFrame(d_rows), use_container_width=True, hide_index=True)

            elif selected_card == "orders":
                with d_hdr:
                    st.markdown("### Affected Orders Details")
                    st.caption("All customer orders impacted by delays along inbound supply routes.")
                if df.empty:
                    st.success("No orders are exposed to current disruptions.")
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Affected", len(df))
                    m2.metric("Total Customers", df["customer"].nunique())
                    m3.metric("Total SKUs", df["sku"].nunique())
                    m4.metric("Total Units", f"{df['qty'].sum():,}")

                    o_view = df.copy()
                    o_view["ml_prob_pct"] = (o_view["ml_prob"] * 100).round(0)
                    o_view["breach_status"] = o_view["breach"].map(lambda b: "Breach" if b else "OK")
                    disp_cols = ["order_id", "customer", "sku", "qty", "port_name", "delay_days", "buffer_days", "cover_days", "risk_score", "risk_label", "ml_prob_pct", "breach_status"]
                    st.dataframe(
                        o_view[disp_cols], use_container_width=True, hide_index=True,
                        column_config={
                            "order_id": "Order ID",
                            "customer": "Customer",
                            "sku": "SKU",
                            "qty": "Quantity",
                            "port_name": "Inbound Port",
                            "delay_days": "Delay (days)",
                            "buffer_days": "Buffer (days)",
                            "cover_days": "Cover (days)",
                            "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%d"),
                            "risk_label": "Risk Tier",
                            "ml_prob_pct": st.column_config.ProgressColumn("ML Late Prob (%)", min_value=0, max_value=100, format="%d%%"),
                            "breach_status": "Status",
                        }
                    )

            elif selected_card == "high_risk":
                with d_hdr:
                    st.markdown("### High Risk Orders Details")
                    st.caption("Critical orders requiring priority planner review and mitigation.")
                high_df = df[df.risk_label == "HIGH"] if not df.empty else pd.DataFrame()
                if high_df.empty:
                    st.success("No orders currently flagged as HIGH risk.")
                else:
                    st.warning(f"{len(high_df)} orders are at HIGH risk. Disruption delay significantly threatens promised fulfillment.")
                    h_view = high_df.copy()
                    h_view["ml_prob_pct"] = (h_view["ml_prob"] * 100).round(0)
                    h_view["breach_status"] = h_view["breach"].map(lambda b: "Breach" if b else "OK")
                    disp_cols = ["order_id", "customer", "sku", "qty", "port_name", "delay_days", "buffer_days", "cover_days", "risk_score", "ml_prob_pct", "breach_status"]
                    st.dataframe(
                        h_view[disp_cols], use_container_width=True, hide_index=True,
                        column_config={
                            "order_id": "Order ID",
                            "customer": "Customer",
                            "sku": "SKU",
                            "qty": "Quantity",
                            "port_name": "Disrupted Port",
                            "delay_days": "Delay (days)",
                            "buffer_days": "Buffer (days)",
                            "cover_days": "Cover (days)",
                            "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%d"),
                            "ml_prob_pct": st.column_config.ProgressColumn("ML Late Prob (%)", min_value=0, max_value=100, format="%d%%"),
                            "breach_status": "Status",
                        }
                    )
                    st.info("**Next step:** Switch to the Order detail page to approve mitigation plans (e.g. inventory transfer or reroute).")

            elif selected_card == "breaches":
                with d_hdr:
                    st.markdown("### Potential Delivery Breaches Details")
                    st.caption("Orders where disruption delay exceeds customer delivery safety buffer.")
                breach_df = df[df.breach == True] if not df.empty else pd.DataFrame()
                if breach_df.empty:
                    st.success("Zero delivery breaches predicted! All current delays are absorbed by buffer days.")
                else:
                    st.error(f"{len(breach_df)} orders are projected to breach customer SLAs unless expedited or rerouted.")
                    b_view = breach_df.copy()
                    b_view["shortfall_days"] = b_view["delay_days"] - b_view["buffer_days"]
                    b_view["ml_prob_pct"] = (b_view["ml_prob"] * 100).round(0)
                    disp_cols = ["order_id", "customer", "sku", "qty", "port_name", "delay_days", "buffer_days", "shortfall_days", "cover_days", "risk_score", "ml_prob_pct"]
                    st.dataframe(
                        b_view[disp_cols], use_container_width=True, hide_index=True,
                        column_config={
                            "order_id": "Order ID",
                            "customer": "Customer",
                            "sku": "SKU",
                            "qty": "Quantity",
                            "port_name": "Disrupted Port",
                            "delay_days": "Delay (days)",
                            "buffer_days": "Buffer (days)",
                            "shortfall_days": "Days Overdue (Shortfall)",
                            "cover_days": "Cover (days)",
                            "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%d"),
                            "ml_prob_pct": st.column_config.ProgressColumn("ML Late Prob (%)", min_value=0, max_value=100, format="%d%%"),
                        }
                    )

    if df.empty:
        st.success("No orders are exposed to the current disruptions.")
    else:
        if df.stale.any():
            st.warning("SAFE MODE - shipment data is stale. Prediction reliability may be reduced; "
                       "verify manually before approving high-impact actions.")
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Highest-risk orders")
            st.altair_chart(ui.risk_bar_chart(df))
        with right:
            st.subheader("Risk mix")
            st.altair_chart(ui.risk_donut(df))

    st.subheader("Supply routes")
    try:
        st.pydeck_chart(ui.route_map(net["ports"], net["routes"], net["shipments"],
                                     weather.PORT_COORDS, delay_map, dark=dark_mode))
        st.caption(f"Arcs run from each port to the factory. Red = disrupted, green = operating. "
                   f"Thicker arc = more shipments in transit. Hover for details. (Theme: {'Dark' if dark_mode else 'Light'})")
    except Exception:
        map_df = ui.port_map_df(net["ports"], weather.PORT_COORDS, delay_map)
        try:
            st.map(map_df, latitude="lat", longitude="lon", color="color", size="size")
        except Exception:
            st.map(map_df[["lat", "lon"]])
        st.caption("Red = disrupted port, green = operating normally.")

    if not df.empty:
        st.subheader("All exposed orders")
        view = df.copy()
        view["risk_label"] = view["risk_label"]
        view["ml_prob"] = (view["ml_prob"] * 100).round(0)
        view["breach"] = view["breach"].map(lambda b: "Breach" if b else "OK")
        cols = ["order_id", "customer", "sku", "risk_score", "risk_label", "ml_prob",
                "delay_days", "buffer_days", "cover_days", "breach", "port_name"]
        st.dataframe(
            view[cols], use_container_width=True, hide_index=True,
            column_config={
                "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%d"),
                "ml_prob": st.column_config.ProgressColumn("ML late prob (%)", min_value=0, max_value=100, format="%d"),
            })
        st.caption("Risk score is rule-based (not a calibrated probability). "
                   "ML late prob is the model's late-shipment estimate.")

elif page == P_ORDER:
    if df.empty:
        ui.hero("Why is this order at risk?", "Evidence first, action second.")
        st.info("Nothing to show - no exposed orders.")
        st.stop()
    default = df.index[df.order_id == "O-1042"].tolist()
    order_id = st.selectbox("Choose an order", df.order_id.tolist(), index=default[0] if default else 0)
    res = pipeline.run_agent(net, mdl, order_id, delay_map, extra_hours)
    row, rec = res["row"], res["recommended"]

    ui.hero(f"Order {order_id}", f"{row.customer} · {row.sku} · quantity {row.qty}",
            tags=[f"{row.risk_label} RISK",
                  f"CONFIDENCE: {res['confidence']}"])

    st.markdown(ui.flow_html([
        ("1", "Disruption", f"{row.port_name}: {row.delay_days} day(s)", "bad"),
        ("2", "Shipment delay", f"{row.shipment_id} +{row.delay_days}d", "bad"),
        ("3", "Stock cover", f"{row.sku}: {row.cover_days}d",
         "bad" if row.cover_days < row.delay_days else ""),
        ("4", "Order at risk", f"Buffer {row.buffer_days}d", "bad" if row.breach else ""),
        ("5", "Recovery", rec.name if rec else "Escalate to human", "good"),
    ]), unsafe_allow_html=True)

    if res["stale"]:
        st.warning(f"DATA QUALITY WARNING: shipment information is {res['hours_old']:.0f} hours old. "
                   "Refresh the data or verify the shipment manually.")

    tab_over, tab_evid, tab_opts, tab_dec = st.tabs(["Overview", "Evidence", "Options", "Decision"])

    with tab_over:
        g1, g2, g3 = st.columns(3)
        g1.markdown(ui.gauge_html("Rule-based risk", f"{row.risk_score}/100", row.risk_score / 100,
                                  ui.RISK_ZONES, "Ranks orders for review"), unsafe_allow_html=True)
        g2.markdown(ui.gauge_html("ML late probability", f"{row.ml_prob:.0%}", float(row.ml_prob),
                                  ui.PROB_ZONES, "Model estimate, not calibrated"), unsafe_allow_html=True)
        with g3:
            st.metric("Confidence", res["confidence"])
            if rec:
                detail = f"Est. cost Rs {rec.cost:,.0f} · residual delay {rec.delay_days:g} day(s)"
                st.markdown(ui.rec_card_html("Recommended action", rec.name, detail), unsafe_allow_html=True)
            else:
                st.markdown(ui.rec_card_html("Recommended action", "Escalate to a human planner",
                                             "No feasible recovery plan was found."), unsafe_allow_html=True)

    with tab_evid:
        e1, e2 = st.columns([3, 2])
        with e1:
            st.subheader("Evidence")
            st.markdown(ui.evidence_html(res["evidence"]), unsafe_allow_html=True)
            if res["llm_summary"]:
                st.info(res["llm_summary"])
            with st.expander("Agent steps (tool calls)"):
                for step in res["steps"]:
                    st.code(step)
        with e2:
            st.subheader("What drives the prediction")
            st.altair_chart(ui.factor_chart(res["ml_factors"]))

    with tab_opts:
        st.subheader("Recovery options")
        cols = st.columns(len(res["options"]))
        for col, o in zip(cols, res["options"]):
            col.markdown(ui.option_card_html(o, rec is not None and o.name == rec.name),
                         unsafe_allow_html=True)
        if rec is None:
            st.error("No feasible plan - escalate to a human planner.")

    with tab_dec:
        st.subheader("Planner decision")
        if rec is None:
            st.error("No feasible plan - escalate to a human planner.")
        else:
            names = [o.name for o in res["options"]]
            chosen_name = st.selectbox("Action to approve (default = recommendation)", names,
                                       index=names.index(rec.name))
            chosen = next(o for o in res["options"] if o.name == chosen_name)
            verified = True
            if res["stale"]:
                verified = st.checkbox("I have manually verified the shipment data")
            evidence_txt = (f"Risk {row.risk_score}/100 · {row.shipment_id} +{row.delay_days}d · "
                            f"cover {row.cover_days}d")

            # Check if this order already has an immutable decision logged
            existing_decision = decisions.get_order_decision(order_id)
            is_decided = existing_decision is not None

            if is_decided:
                dec_status = existing_decision["status"]
                dec_act = existing_decision["action"]
                dec_time = existing_decision["decided_at"]
                dec_reason = existing_decision["reason"]
                if "APPROVED" in dec_status:
                    st.success(
                        f"**Decision Recorded: {dec_status}**\n\n"
                        f"- **Action:** {dec_act}  \n"
                        f"- **Decided at:** {dec_time}  \n"
                        f"- **Details:** {dec_reason}  \n\n"
                        f"*This decision has been finalized and cannot be changed.*"
                    )
                else:
                    st.error(
                        f"**Decision Recorded: {dec_status}**\n\n"
                        f"- **Action:** {dec_act}  \n"
                        f"- **Decided at:** {dec_time}  \n"
                        f"- **Details:** {dec_reason}  \n\n"
                        f"*This decision has been finalized and cannot be changed.*"
                    )
                if SHIP_GIF.exists() and st.session_state.get(f"_just_approved_{order_id}"):
                    st.markdown(ui.ship_html(SHIP_GIF, "Approved - cargo is on the move"),
                                unsafe_allow_html=True)
                    st.session_state[f"_just_approved_{order_id}"] = False

            c1, c2 = st.columns(2)
            btn_disabled = is_decided or (not verified)
            help_tooltip = "Decision is locked for this order and cannot be changed" if is_decided else None

            if c1.button("APPROVE", disabled=btn_disabled, use_container_width=True, help=help_tooltip):
                confirm_approve_dialog(order_id, chosen, evidence_txt)

            if c2.button("REJECT", disabled=btn_disabled, use_container_width=True, help=help_tooltip):
                confirm_reject_dialog(order_id, chosen, evidence_txt)

elif page == P_SIM:
    ui.hero("What-if disruption simulator", "Test a disruption before committing to a recovery plan.",
            tags=["ILLUSTRATIVE - COMPUTED FROM SYNTHETIC DATA"])
    c1, c2 = st.columns(2)
    port_id = c1.selectbox("Disruption: port closure at", list(port_names), format_func=port_names.get)
    days = c2.slider("Duration (days)", 1, 7, 3)
    if st.button("SIMULATE IMPACT"):
        base, sim = simulator.simulate(net, mdl, port_id, days, base_map=delay_map)
        m1, m2, m3 = st.columns(3)
        m1.metric("Orders affected", sim["orders_affected"],
                  delta=f"{sim['orders_affected'] - base['orders_affected']:+d}", delta_color="inverse")
        m2.metric("Potential delivery breaches", sim["breaches"],
                  delta=f"{sim['breaches'] - base['breaches']:+d}", delta_color="inverse")
        m3.metric("Est. recovery cost", f"Rs {sim['recovery_cost']:,.0f}",
                  delta=f"Rs {sim['recovery_cost'] - base['recovery_cost']:+,.0f}", delta_color="inverse")
        st.toast("Simulation complete")
        st.altair_chart(ui.sim_chart(base, sim))
        if sim["unresolved"]:
            st.error(f"{sim['unresolved']} breaching order(s) have no feasible plan - escalate to a human.")
        st.caption("Computed from the synthetic demo data - not real-world results.")

elif page == P_WEATHER:
    ui.hero("Port weather", "Severe weather becomes a disruption the agent can reason about.")
    if not weather_info:
        st.info("Tick 'Live weather (Open-Meteo)' in the sidebar to read real forecasts, "
                "or 'Demo: inject a storm' to simulate one.")
    else:
        chart = ui.gust_chart(weather_info, port_names, weather.GUST_LIMIT_KMH)
        if chart is not None:
            st.altair_chart(chart)
            st.caption(f"Dashed line = severe-weather limit ({weather.GUST_LIMIT_KMH} km/h gusts).")
        st.dataframe(weather.weather_table(weather_info, port_names),
                     use_container_width=True, hide_index=True)
        st.caption(f"A day is 'severe' if gusts >= {weather.GUST_LIMIT_KMH} km/h or rain >= "
                   f"{weather.RAIN_LIMIT_MM} mm. Severe days in the next {weather.FORECAST_DAYS} days "
                   "become disruption days. These thresholds are assumptions, not validated.")

elif page == P_LOG:
    ui.hero("Decision log", "Every approval and rejection, with the evidence behind it.")
    log = decisions.list_decisions()
    if log.empty:
        st.info("No decisions yet. Approve or reject something on the Order detail page.")
    else:
        approved = log["status"].str.contains("APPROVED")
        m1, m2, m3 = st.columns(3)
        m1.metric("Decisions logged", len(log))
        m2.metric("Approved", int(approved.sum()))
        m3.metric("Approved spend", f"Rs {log.loc[approved, 'est_cost'].sum():,.0f}")
        st.dataframe(log, use_container_width=True, hide_index=True)
        with st.expander("Demo controls: reset decision log"):
            st.caption("Clearing the log resets the locked status on orders so you can demonstrate the approval/rejection flow again.")
            if st.button("Reset / Clear All Decisions"):
                decisions.clear_decisions()
                st.toast("Decision history cleared.")
                st.rerun()

else:
    ui.hero("Model check: rules vs ML", "Prove value before claiming accuracy.",
            tags=["TIME-BASED SPLIT", "NO FUTURE LEAKAGE"])
    m = mdl.metrics
    st.write(f"Model: **{m['kind']}** · trained on the oldest {m['train_rows']} shipments, "
             f"tested on the newest {m['test_rows']}.")
    st.altair_chart(ui.model_chart(m))
    st.table(pd.DataFrame({"Precision": [m["rule"]["precision"], m["ml"]["precision"]],
                           "Recall": [m["rule"]["recall"], m["ml"]["recall"]]},
                          index=["Simple rule", "ML model"]).round(3))
    st.caption("Synthetic data: these scores prove the pipeline works, not real-world accuracy.")
