"""
Home Page - Dashboard and Overview
"""

import streamlit as st
from utils.api_client import get_api_client
from datetime import datetime

def show():
    """Display home page"""
    st.title("🏛️ Citizen Grievance Management Portal")
    st.markdown("---")
    
    # Welcome section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to the Grievance Management System
        
        Our AI-powered platform helps efficiently manage and route citizen grievances
        to the appropriate departments. We use advanced NLP and machine learning to:
        
        - 🤖 **Automatically categorize** complaints into departments
        - 💭 **Analyze sentiment** to understand complaint urgency
        - ⚡ **Prioritize cases** based on severity and emergency indicators
        - 📊 **Track and manage** complaints throughout their lifecycle
        """)
    
    with col2:
        st.markdown("""
        ### Quick Stats
        """)
        
        # Try to get API stats
        api_client = get_api_client()
        stats = api_client.get_statistics()
        
        if stats:
            st.metric("System Status", "🟢 Online")
            st.metric("Departments", len(stats.get("departments", [])))
            st.metric("Priority Tiers", len(stats.get("priority_tiers", [])))
    
    st.markdown("---")
    
    # Feature cards
    st.subheader("📋 Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 📝 Submit Grievance
        
        File individual complaints quickly and receive
        immediate analysis including department routing,
        sentiment analysis, and priority assessment.
        """)
        if st.button("Submit Grievance →", key="btn_submit"):
            st.session_state.page = "submit"
    
    with col2:
        st.markdown("""
        #### 📤 Batch Upload
        
        Process multiple grievances at once using
        a CSV file. Ideal for bulk migrations or
        historical data analysis.
        """)
        if st.button("Upload File →", key="btn_batch"):
            st.session_state.page = "batch"
    
    with col3:
        st.markdown("""
        #### 📊 Analytics
        
        View comprehensive statistics, trends,
        and insights about filed grievances across
        departments and priority levels.
        """)
        if st.button("View Analytics →", key="btn_analytics"):
            st.session_state.page = "analytics"
    
    st.markdown("---")
    
    # System Information
    st.subheader("ℹ️ System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**API Status**")
        if api_client.health_check():
            st.success("✅ Connected and Operational")
        else:
            st.error("❌ Connection Failed - Check API Server")
    
    with col2:
        st.info("**Available Departments**")
        if stats:
            for dept in stats.get("departments", []):
                st.write(f"• {dept}")
    
    with col3:
        st.info("**Priority Tiers**")
        priority_info = {
            "P1": "🔴 Critical (2 hours)",
            "P2": "🟠 High (24 hours)",
            "P3": "🔵 Medium (3 days)",
            "P4": "🟢 Low (7 days)"
        }
        for tier, info in priority_info.items():
            st.write(f"{info}")
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: gray; margin-top: 2rem;'>
        <p>AI-Driven Citizen Grievance Management System | Last Updated: 2026</p>
    </div>
    """, unsafe_allow_html=True)
