"""
Vision Guard — UI Components & Surveillance Dark Theme Styling
Shared CSS, glassmorphism cards, glowing badges, and navigation headers.
"""

import streamlit as st


def inject_custom_css():
    """Injects high-tech dark surveillance command center styling."""
    css = """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Global Theme */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #0B132B;
        color: #E2E8F0;
    }

    /* Top surveillance banner */
    .surveillance-header {
        background: linear-gradient(135deg, #1C2541 0%, #0B132B 100%);
        border: 1px solid #3A86FF44;
        border-radius: 10px;
        padding: 18px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .surveillance-title {
        color: #FFFFFF;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .surveillance-tag {
        background: rgba(0, 229, 255, 0.15);
        color: #00E5FF;
        border: 1px solid #00E5FF55;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }

    /* Metric cards */
    .metric-card {
        background: #1C2541;
        border: 1px solid #3A86FF33;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #00E5FF;
    }

    .metric-title {
        color: #94A3B8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 600;
    }

    .metric-val {
        color: #FFFFFF;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-sub {
        color: #38BDF8;
        font-size: 0.75rem;
        margin-top: 4px;
    }

    /* Badges */
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid #EF4444;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
    }

    .badge-approved {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ADE80;
        border: 1px solid #22C55E;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
    }

    .badge-pending {
        background-color: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid #F59E0B;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
    }

    /* Custom Streamlit adjustments */
    .stButton>button {
        background-color: #4361EE;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 8px 16px;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        background-color: #3B82F6;
        box-shadow: 0 0 12px rgba(67, 97, 238, 0.6);
        color: #FFFFFF;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #121829;
        border-right: 1px solid #3A86FF22;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161F38;
        padding: 6px;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #94A3B8;
        border-radius: 6px;
        padding: 8px 16px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #4361EE !important;
        color: #FFFFFF !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_surveillance_header(title: str, subtitle: str, tag: str = "LIVE SURVEILLANCE FEED"):
    """Renders a command-center header banner."""
    html = f"""
    <div class="surveillance-header">
        <div>
            <div class="surveillance-title">
                🛡️ {title}
            </div>
            <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">
                {subtitle}
            </div>
        </div>
        <div class="surveillance-tag">
            ● {tag}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtext: str = ""):
    """Renders a styled metric card."""
    html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-val">{value}</div>
        {f'<div class="metric-sub">{subtext}</div>' if subtext else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def require_auth():
    """Ensures user is logged in before accessing protected pages."""
    if not st.session_state.get("authenticated", False):
        st.warning("⚠️ Access Restricted: Please authenticate on the Login page.")
        st.stop()
