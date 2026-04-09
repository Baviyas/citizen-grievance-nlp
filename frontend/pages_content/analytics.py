"""
Analytics Page - Statistics and Insights
"""

import streamlit as st
from utils.api_client import get_api_client
from utils.ui_components import (
    render_priority_distribution,
    render_department_distribution,
    render_sentiment_distribution
)
import pandas as pd

def show():
    """Display analytics page"""
    st.title("📊 Analytics & Insights")
    st.markdown("---")
    
    # Check if data is available in session
    if "grievance_history" not in st.session_state:
        st.session_state.grievance_history = []
    
    # Load sample data info
    api_client = get_api_client()
    stats = api_client.get_statistics()
    
    if not stats:
        st.error("❌ Unable to load system statistics")
        return
    
    # Overview cards
    st.subheader("📈 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Departments", len(stats.get("departments", [])))
    
    with col2:
        st.metric("Priority Levels", len(stats.get("priority_tiers", [])))
    
    with col3:
        st.metric("Sentiment Types", len(stats.get("sentiment_types", [])))
    
    with col4:
        st.metric("API Status", "🟢 Online" if api_client.health_check() else "🔴 Offline")
    
    st.markdown("---")
    
    # System Configuration
    st.subheader("⚙️ System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Available Departments:**")
        for dept in stats.get("departments", []):
            st.write(f"• {dept}")
    
    with col2:
        st.markdown("**Priority Tiers:**")
        priority_info = {
            "P1": "Critical (2 hours)",
            "P2": "High (24 hours)",
            "P3": "Medium (3 days)",
            "P4": "Low (7 days)"
        }
        for tier, info in priority_info.items():
            st.write(f"• {tier}: {info}")
    
    st.markdown("---")
    
    # Model Information
    st.subheader("🤖 AI Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        models = stats.get("models", {})
        st.markdown("**Deployed Models:**")
        for model_type, model_name in models.items():
            st.write(f"• **{model_type}**: {model_name}")
    
    with col2:
        st.markdown("**Model Performance:**")
        st.info("""
        - **Routing Model**: Logistic Regression
        - **Accuracy**: 99.8%
        - **Sentiment Model**: DistilBERT-based
        - **Classes**: 4 core departments
        """)
    
    st.markdown("---")
    
    # Sample Analysis
    st.subheader("📊 Sample Grievance Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Department Distribution**
        
        Grievances are automatically routed to:
        - Environment
        - Transport
        - Social & Health Services
        - Non-Complaint
        """)
    
    with col2:
        st.markdown("""
        **Sentiment Analysis**
        
        Complaints are analyzed for:
        - 🔴 Critical
        - ⚠️ Negative
        - 😐 Neutral
        - ✅ Positive
        """)
    
    with col3:
        st.markdown("""
        **Urgency Scoring**
        
        Calculated based on:
        - Sentiment (40%)
        - Keywords (25%)
        - Confidence (20%)
        - Recency (15%)
        """)
    
    st.markdown("---")
    
    # Data Load Sample
    st.subheader("📁 Available Data")
    
    # Load actual CSV data for analysis
    import os
    from pathlib import Path
    
    try:
        processed_file = Path(__file__).parent.parent.parent / "outputs" / "grievance_with_urgency.csv"
        if processed_file.exists():
            df = pd.read_csv(processed_file)
            
            st.markdown(f"**Processed Grievance Data**: {len(df)} records")
            
            # Display sample data
            st.dataframe(
                df.head(10),
                use_container_width=True,
                hide_index=True
            )
            
            # Statistics from actual data
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if "department" in df.columns or "Department" in df.columns:
                    dept_col = "department" if "department" in df.columns else "Department"
                    st.metric(
                        "Total Grievances",
                        len(df),
                        delta=None
                    )
            
            with col2:
                if "department" in df.columns or "Department" in df.columns:
                    dept_col = "department" if "department" in df.columns else "Department"
                    st.metric(
                        "Unique Departments",
                        df[dept_col].nunique() if dept_col in df.columns else "N/A"
                    )
            
            with col3:
                if "urgency_score" in df.columns:
                    st.metric(
                        "Avg Urgency Score",
                        f"{df['urgency_score'].mean():.2f}",
                        delta=None
                    )
        else:
            st.info("📁 Processed data file not found. Submit grievances to populate data.")
    
    except Exception as e:
        st.warning(f"Could not load sample data: {str(e)}")
