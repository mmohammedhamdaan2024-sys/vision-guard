"""
Vision Guard — Page 1: Main Surveillance Dashboard
Aggregated real-time metrics, Plotly interactive telemetry charts, and visual evidence gallery.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database.db_manager import DatabaseManager
from utils.ui_components import inject_custom_css, render_surveillance_header, render_metric_card, require_auth

st.set_page_config(page_title="Dashboard | Vision Guard", page_icon="📊", layout="wide")
inject_custom_css()
require_auth()

db = DatabaseManager("data/visionguard.db")
metrics = db.get_dashboard_metrics()

render_surveillance_header(
    title="MAIN TRAFFIC SURVEILLANCE DASHBOARD",
    subtitle="Real-time road safety telemetry, violation event stream, and visual evidence archive.",
    tag="TELEMETRY ACTIVE"
)

# Row 1: KPI Counter Cards
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    render_metric_card("Total Vehicles", f"{metrics['total_vehicles']:,}", "All scanned vehicles")
with c2:
    render_metric_card("Violations Logged", f"{metrics['total_violations']:,}", "Total safety alerts")
with c3:
    render_metric_card("Helmet Alerts", f"{metrics['helmet_alerts']:,}", "Two-wheeler non-compliance")
with c4:
    render_metric_card("Seat-Belt Alerts", f"{metrics['seatbelt_alerts']:,}", "Occupant restraint alerts")
with c5:
    render_metric_card("Tint & Mirror Flags", f"{metrics['tint_mirror_alerts']:,}", "Visibility reviews")
with c6:
    render_metric_card("Pending Reviews", f"{metrics['pending_reviews']:,}", "Awaiting officer sign-off")

st.markdown("<br/>", unsafe_allow_html=True)

# Row 2: Plotly Charts
col_chart1, col_chart2 = st.columns([1, 1])

with col_chart1:
    st.markdown("#### 🍩 Violations by Category")
    cat_data = db.get_violations_by_category()
    if cat_data:
        df_cat = pd.DataFrame(cat_data)
        fig_donut = px.pie(
            df_cat,
            values="count",
            names="violation_type",
            hole=0.55,
            color_discrete_sequence=["#EF4444", "#3A86FF", "#F59E0B", "#10B981", "#8B5CF6"]
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("No violation records logged yet.")

with col_chart2:
    st.markdown("#### 📈 Daily Violation Trends (Past 7 Days)")
    trend_data = db.get_daily_violation_trends(7)
    if trend_data:
        df_trend = pd.DataFrame(trend_data)
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=df_trend["date"], y=df_trend["helmet"],
            name="Helmet Alerts", marker_color="#EF4444"
        ))
        fig_trend.add_trace(go.Bar(
            x=df_trend["date"], y=df_trend["seatbelt"],
            name="Seat-belt Alerts", marker_color="#3A86FF"
        ))
        fig_trend.add_trace(go.Bar(
            x=df_trend["date"], y=df_trend["others"],
            name="Tint / Mirror / Plate", marker_color="#F59E0B"
        ))
        fig_trend.update_layout(
            barmode="stack",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            margin=dict(t=20, b=20, l=20, r=20),
            xaxis=dict(gridcolor="#1E293B"),
            yaxis=dict(gridcolor="#1E293B"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Trend telemetry pending data collection.")

st.markdown("<br/>", unsafe_allow_html=True)

# Row 3: Visual Evidence Gallery
st.markdown("### 📸 Recent Visual Evidence Gallery")
recent_violations = db.list_violations(limit=4)

if recent_violations:
    gcols = st.columns(4)
    for idx, v in enumerate(recent_violations):
        with gcols[idx]:
            evid_path = v.get("evidence_path", "")
            if evid_path and os.path.exists(evid_path):
                st.image(evid_path, use_container_width=True)
            else:
                st.markdown(
                    f"""
                    <div style="background: #1C2541; height: 160px; border-radius: 8px; border: 1px dashed #3A86FF44;
                                display: flex; align-items: center; justify-content: center; text-align: center; color: #94A3B8; font-size: 0.8rem;">
                        📷 Evidence Snapshot<br/>[{v.get('tracking_id')}]
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown(f"**{v.get('violation_type')}**")
            st.caption(f"Plate: `{v.get('plate_number', 'UNREADABLE')}` | Conf: `{v.get('confidence', 0.0):.2f}`")
            status = v.get("status", "Pending Review")
            badge_class = "badge-approved" if status == "Approved" else ("badge-critical" if status == "Rejected" else "badge-pending")
            st.markdown(f"<span class='{badge_class}'>{status}</span>", unsafe_allow_html=True)
else:
    st.info("No recent evidence images available.")

st.markdown("<br/>", unsafe_allow_html=True)

# Row 4: Recent Activity Table
st.markdown("### 📋 Recent Detection Event Stream")
all_violations = db.list_violations(limit=10)
if all_violations:
    table_rows = []
    for r in all_violations:
        table_rows.append({
            "Violation ID": f"VIO-{r['id']:04d}",
            "Tracking ID": r["tracking_id"],
            "Vehicle Type": str(r.get("vehicle_type", "Unknown")).capitalize(),
            "Registration Plate": r.get("plate_number", "UNREADABLE"),
            "Violation Category": r["violation_type"],
            "Severity": r["severity"],
            "Confidence": f"{r['confidence']:.2f}",
            "Review Status": r["status"],
            "Timestamp": r["timestamp"]
        })
    df_table = pd.DataFrame(table_rows)
    st.dataframe(df_table, use_container_width=True, hide_index=True)
