"""
Vision Guard — Page 6: Analytics & Reports
Statistical audit summaries, compliance rate analytics, CSV data exports, and PDF report generation.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database.db_manager import DatabaseManager
from utils.report_generator import generate_audit_report_pdf
from utils.ui_components import inject_custom_css, render_surveillance_header, render_metric_card, require_auth

st.set_page_config(page_title="Analytics & Reports | Vision Guard", page_icon="📈", layout="wide")
inject_custom_css()
require_auth()

db = DatabaseManager("data/visionguard.db")
metrics = db.get_dashboard_metrics()

render_surveillance_header(
    title="TRAFFIC SAFETY ANALYTICS & AUDIT REPORTS",
    subtitle="Macro telemetry analysis, violation distribution statistics, data exports, and audit report generation.",
    tag="STATISTICAL AUDIT"
)

# Metric Summary
c1, c2, c3, c4 = st.columns(4)
with c1:
    compliance_rate = 100.0 - (metrics["total_violations"] / max(1, metrics["total_vehicles"]) * 100.0)
    compliance_rate = max(0.0, min(100.0, compliance_rate))
    render_metric_card("Compliance Score", f"{compliance_rate:.1f}%", "Overall road safety index")
with c2:
    render_metric_card("Total Citations", f"{metrics['total_violations']:,}", "Recorded safety deviations")
with c3:
    render_metric_card("Review Workload", f"{metrics['pending_reviews']:,}", "Cases pending manual adjudication")
with c4:
    render_metric_card("Approved Fines", f"₹{metrics['approved_fines']:,.2f}", "Total confirmed revenue")

st.markdown("<br/>", unsafe_allow_html=True)

# Charts Row
ch1, ch2 = st.columns(2)

with ch1:
    st.markdown("#### 📊 Violation Breakdown by Category")
    cat_data = db.get_violations_by_category()
    if cat_data:
        df_cat = pd.DataFrame(cat_data)
        fig_bar = px.bar(
            df_cat,
            x="violation_type",
            y="count",
            color="violation_type",
            text="count",
            color_discrete_sequence=["#EF4444", "#3A86FF", "#F59E0B", "#10B981", "#8B5CF6"]
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            xaxis=dict(title="Category", gridcolor="#1E293B"),
            yaxis=dict(title="Count", gridcolor="#1E293B"),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with ch2:
    st.markdown("#### 🎯 Adjudication Status Distribution")
    challans = db.list_challans(limit=500)
    if challans:
        df_ch = pd.DataFrame(challans)
        status_counts = df_ch["review_status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig_pie = px.pie(
            status_counts,
            values="Count",
            names="Status",
            color="Status",
            color_discrete_map={"Approved": "#10B981", "Pending Review": "#F59E0B", "Rejected": "#EF4444", "Under Investigation": "#3A86FF"}
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# Data Export Section
st.markdown("### 💾 Export Data & Official Audit Reports")
ex1, ex2 = st.columns(2)

with ex1:
    st.markdown(
        """
        <div class="metric-card">
            <h4 style="color: #00E5FF; margin-bottom: 6px;">📥 Export Dataset to CSV</h4>
            <p style="font-size: 0.85rem; color: #94A3B8;">
                Download complete audit registry containing vehicle tracking IDs, timestamps, categories, and fine amounts.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if violations := db.list_violations(limit=1000):
        df_export = pd.DataFrame(violations)
        csv_data = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV Dataset",
            data=csv_data,
            file_name="visionguard_violation_audit_export.csv",
            mime="text/csv",
            use_container_width=True
        )

with ex2:
    st.markdown(
        """
        <div class="metric-card">
            <h4 style="color: #00E5FF; margin-bottom: 6px;">📑 Generate Executive PDF Audit Report</h4>
            <p style="font-size: 0.85rem; color: #94A3B8;">
                Compile a professional printable PDF report summarizing enforcement statistics and safety index.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Generate Executive Audit PDF Report", use_container_width=True):
        reports_dir = "data/reports"
        os.makedirs(reports_dir, exist_ok=True)
        rpt_path = os.path.join(reports_dir, "VisionGuard_Executive_Audit_Report.pdf")
        cat_data = db.get_violations_by_category()
        viol_list = db.list_violations(limit=20)
        generate_audit_report_pdf(metrics, cat_data, rpt_path)

        with open(rpt_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Compiled Audit Report (.pdf)",
                data=f.read(),
                file_name=os.path.basename(rpt_path),
                mime="application/pdf",
                use_container_width=True
            )
