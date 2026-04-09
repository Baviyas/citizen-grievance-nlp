"""
About Page - Project Information
"""

import streamlit as st
from datetime import datetime

def show():
    """Display about page"""
    st.title("ℹ️ About This System")
    st.markdown("---")
    
    # Project Overview
    st.subheader("🎯 Project Overview")
    
    st.markdown("""
    The **AI-Driven Citizen Grievance & Sentiment Analysis System** is an intelligent platform
    designed to streamline government complaint management using advanced Natural Language Processing
    and Machine Learning technologies.
    
    ### Key Features
    
    ✨ **Intelligent Routing**
    - Automatically categorizes complaints into 4 core departments
    - Uses Logistic Regression with 99.8% accuracy
    - Real-time prediction on submitted complaints
    
    💭 **Sentiment Analysis**
    - DistilBERT-based sentiment classification
    - Detects critical, negative, neutral, and positive sentiments
    - Helps identify urgent complaints automatically
    
    ⚡ **Urgency Scoring**
    - Weighted scoring system considering:
      - Sentiment strength (40%)
      - Emergency keywords (25%)
      - Model confidence (20%)
      - Recency factor (15%)
    - Assigns priority tiers (P1-P4) with SLAs
    
    📊 **Analytics & Tracking**
    - Batch processing for multiple grievances
    - Detailed analytics and visualizations
    - CSV import/export capabilities
    """)
    
    st.markdown("---")
    
    # Technical Architecture
    st.subheader("🏗️ Technical Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Frontend**
        - Streamlit
        - Plotly
        - Pandas
        """)
    
    with col2:
        st.markdown("""
        **Backend**
        - FastAPI
        - Uvicorn
        - Pydantic
        """)
    
    with col3:
        st.markdown("""
        **ML/NLP**
        - scikit-learn
        - Transformers
        - PyTorch
        """)
    
    st.markdown("---")
    
    # Data Flow
    st.subheader("🔄 Processing Pipeline")
    
    st.markdown("""
    ```
    User Input
        ↓
    [FastAPI Backend]
        ↓
    ├─ Text Preprocessing
    ├─ Department Routing (Logistic Regression)
    ├─ Sentiment Analysis (DistilBERT)
    ├─ Urgency Calculation
    └─ Priority Assignment
        ↓
    [Response with Recommendations]
        ↓
    [Streamlit Frontend Display]
    ```
    """)
    
    st.markdown("---")
    
    # Department Categories
    st.subheader("🏢 Department Categories")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Environment**
        - Pollution complaints
        - Waste management
        - Environmental hazards
        - Sanitation issues
        """)
    
    with col2:
        st.markdown("""
        **Transport**
        - Traffic issues
        - Pothole reports
        - Public transit complaints
        - Parking violations
        """)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        **Social & Health Services**
        - Healthcare access
        - Social services
        - Welfare concerns
        - Public health issues
        """)
    
    with col4:
        st.markdown("""
        **Non-Complaint**
        - General inquiries
        - Information requests
        - Positive feedback
        - Suggestions
        """)
    
    st.markdown("---")
    
    # Priority Levels
    st.subheader("🎯 Priority Tiers & SLAs")
    
    priority_data = {
        "Tier": ["P1", "P2", "P3", "P4"],
        "Level": ["🔴 Critical", "🟠 High", "🔵 Medium", "🟢 Low"],
        "Urgency Range": ["80-100", "60-79", "40-59", "0-39"],
        "SLA": ["2 hours", "24 hours", "3 days", "7 days"],
        "Action": [
            "Senior officer review, immediate contact",
            "Escalated review, urgent processing",
            "Scheduled review within SLA",
            "Standard processing"
        ]
    }
    
    import pandas as pd
    df_priority = pd.DataFrame(priority_data)
    st.dataframe(df_priority, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Dataset Information
    st.subheader("📊 Dataset & Training")
    
    st.markdown("""
    ### NYC 311 Service Requests Dataset
    
    - **Source**: NYC 311 public data
    - **Records**: ~24k+ historical complaints
    - **Features**: Complaint text, timestamps, locations, outcomes
    
    ### Model Training
    
    - **Preprocessing**: Lemmatization, stopword removal, custom NLP pipeline
    - **Feature Engineering**: TF-IDF vectorization (unigrams & bigrams)
    - **Model Selection**: 5-Fold Stratified Cross-Validation
    - **Best Model**: Logistic Regression (99.8% accuracy)
    """)
    
    st.markdown("---")
    
    # Contact & Support
    st.subheader("📞 Support & Contact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **For Grievance Submission**
        - Use the "Submit Grievance" page
        - Be specific and detailed
        - Include location details
        
        **For Technical Support**
        - Check API status on Home page
        - View analytics for system health
        - Contact system administrator
        """)
    
    with col2:
        st.info("""
        **System Information**
        - Version: 1.0.0
        - Last Updated: 2026
        - API Endpoint: localhost:8000
        - Frontend Port: 8501
        """)
    
    st.markdown("---")
    
    # Footer
    st.markdown(f"""
    <div style='text-align: center; color: gray; margin-top: 2rem;'>
        <p>AI-Driven Citizen Grievance Management System</p>
        <p>© 2026 All Rights Reserved | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)
