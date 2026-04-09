"""
UI Component utilities for Streamlit app
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

def render_grievance_card(grievance: Dict[str, Any]) -> None:
    """Render a grievance result card"""
    with st.container():
        # Header with ID and emergency badge
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.subheader(f"📋 {grievance['grievance_id']}")
        
        with col3:
            if grievance.get("emergency", False):
                st.markdown(
                    '<div class="emergency-badge">🚨 EMERGENCY</div>',
                    unsafe_allow_html=True
                )
        
        st.write(f"**Complaint:** {grievance['description'][:200]}...")
        
        # Key Information
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Department", grievance["predicted_department"])
        with col2:
            st.metric("Confidence", f"{grievance['confidence']*100:.1f}%")
        with col3:
            st.metric("Sentiment", grievance["sentiment"].capitalize())
        with col4:
            priority = grievance["priority_tier"]
            priority_colors = {
                "P1": "🔴",
                "P2": "🟠",
                "P3": "🔵",
                "P4": "🟢"
            }
            st.metric("Priority", f"{priority_colors.get(priority, '')} {priority}")
        
        # Urgency and SLA
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Urgency Score", f"{grievance['urgency_score']:.2f}/100")
        with col2:
            st.metric("SLA", grievance["sla"])
        
        # Recommendations
        st.write("**Recommended Actions:**")
        for i, rec in enumerate(grievance.get("recommendations", []), 1):
            st.write(f"{i}. {rec}")
        
        st.divider()


def render_batch_results(results: Dict[str, Any]) -> None:
    """Render batch processing results"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Processed",
            results["total_processed"],
            delta=None
        )
    with col2:
        st.metric(
            "Successful",
            results["successful"],
            delta=f"{(results['successful']/results['total_processed']*100):.1f}%",
            delta_color="off"
        )
    with col3:
        st.metric(
            "Failed",
            results["failed"],
            delta=None,
            delta_color="off"
        )
    
    st.divider()
    
    # Display results as table
    if results["results"]:
        df = pd.DataFrame([
            {
                "ID": r["grievance_id"],
                "Department": r["predicted_department"],
                "Priority": r["priority_tier"],
                "Urgency Score": r["urgency_score"],
                "Sentiment": r["sentiment"],
                "Emergency": "🚨 Yes" if r["emergency"] else "No"
            }
            for r in results["results"]
        ])
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Priority": st.column_config.TextColumn(width="small"),
                "Urgency Score": st.column_config.NumberColumn(format="%.2f"),
            }
        )
        
        # Download results as CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="grievance_analysis_results.csv",
            mime="text/csv"
        )


def render_priority_distribution(grievances: list) -> None:
    """Render priority distribution chart"""
    import plotly.graph_objects as go
    
    priority_counts = {}
    for g in grievances:
        p = g["priority_tier"]
        priority_counts[p] = priority_counts.get(p, 0) + 1
    
    if not priority_counts:
        st.info("No grievances to display")
        return
    
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(priority_counts.keys()),
                y=list(priority_counts.values()),
                marker=dict(
                    color=["#ff4757", "#ffa502", "#1f77b4", "#2ca02c"]
                )
            )
        ]
    )
    
    fig.update_layout(
        title="Priority Distribution",
        xaxis_title="Priority Tier",
        yaxis_title="Count",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_department_distribution(grievances: list) -> None:
    """Render department distribution chart"""
    import plotly.graph_objects as go
    
    dept_counts = {}
    for g in grievances:
        dept = g["predicted_department"]
        dept_counts[dept] = dept_counts.get(dept, 0) + 1
    
    if not dept_counts:
        st.info("No grievances to display")
        return
    
    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(dept_counts.keys()),
                values=list(dept_counts.values())
            )
        ]
    )
    
    fig.update_layout(
        title="Department Distribution",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_sentiment_distribution(grievances: list) -> None:
    """Render sentiment distribution chart"""
    import plotly.graph_objects as go
    
    sentiment_counts = {}
    for g in grievances:
        sent = g["sentiment"].capitalize()
        sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1
    
    if not sentiment_counts:
        st.info("No grievances to display")
        return
    
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(sentiment_counts.keys()),
                y=list(sentiment_counts.values()),
                marker=dict(
                    color=["#ff4757", "#ffa502", "#1f77b4", "#2ca02c"]
                )
            )
        ]
    )
    
    fig.update_layout(
        title="Sentiment Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Count",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
