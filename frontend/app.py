"""
Main Streamlit App - Citizen Grievance Management System
Entry point for the application with multi-page navigation
"""

import streamlit as st
from streamlit_option_menu import option_menu

# Page configuration
st.set_page_config(
    page_title="Grievance Management Portal",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .priority-p1 {
        background-color: #ff4757;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        display: inline-block;
    }
    .priority-p2 {
        background-color: #ffa502;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        display: inline-block;
    }
    .priority-p3 {
        background-color: #1f77b4;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        display: inline-block;
    }
    .priority-p4 {
        background-color: #2ca02c;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        display: inline-block;
    }
    .emergency-badge {
        background-color: #ff0000;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 0.3rem;
        font-weight: bold;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("🏛️ Grievance Portal")
    st.write("---")
    
    selected = option_menu(
        menu_title="",
        options=["Home", "Submit Grievance", "Batch Upload", "Analytics", "About"],
        icons=["house", "pencil", "upload", "bar-chart", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#f8f9fa"},
            "icon": {"color": "#667eea", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#667eea"},
        }
    )

# Import pages
if selected == "Home":
    from pages_content import home
    home.show()
elif selected == "Submit Grievance":
    from pages_content import submit_grievance
    submit_grievance.show()
elif selected == "Batch Upload":
    from pages_content import batch_upload
    batch_upload.show()
elif selected == "Analytics":
    from pages_content import analytics
    analytics.show()
elif selected == "About":
    from pages_content import about
    about.show()
