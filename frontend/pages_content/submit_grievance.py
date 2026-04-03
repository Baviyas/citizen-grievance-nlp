"""
Submit Grievance Page - Single Complaint Form
"""

import streamlit as st
from utils.api_client import get_api_client
from utils.ui_components import render_grievance_card
import json

def show():
    """Display submit grievance page"""
    st.title("📝 Submit a Grievance")
    st.markdown("---")
    
    st.markdown("""
    Please provide details about your complaint. Our AI system will
    analyze it and route it to the appropriate department with
    priority assessment.
    """)
    
    # Form layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Grievance description - main field
        description = st.text_area(
            "Describe your grievance in detail",
            height=150,
            placeholder="Please provide a detailed description of your complaint...",
            help="Be specific about what happened, where, and any relevant details"
        )
        
        st.write("")
        
        # Optional fields
        st.subheader("Additional Information (Optional)")
        
        col1_opt, col2_opt = st.columns(2)
        
        with col1_opt:
            location = st.text_input(
                "Location/Borough",
                placeholder="e.g., Manhattan, Brooklyn, Queens, etc.",
                help="Where did the incident occur?"
            )
            
            contact_email = st.text_input(
                "Email Address",
                placeholder="your.email@example.com",
                help="For follow-up communication"
            )
        
        with col2_opt:
            category = st.selectbox(
                "Preferred Category (Optional)",
                options=["", "Environment", "Transport", "Social & Health Services", "Other"],
                help="If you know which department, select it"
            )
            
            contact_phone = st.text_input(
                "Phone Number",
                placeholder="+1 (555) 123-4567",
                help="For urgent follow-up"
            )
    
    with col2:
        st.markdown("""
        ### 📋 What Happens Next?
        
        1. **Analysis** - Our AI analyzes your complaint
        2. **Routing** - Routes to appropriate department
        3. **Priority** - Assigns priority level
        4. **Tracking** - You get a reference number
        
        ### ⏱️ Response Times
        
        - **P1** (Critical): 2 hours
        - **P2** (High): 24 hours
        - **P3** (Medium): 3 days
        - **P4** (Low): 7 days
        """)
    
    st.markdown("---")
    
    # Submit button
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        submit_btn = st.button(
            "🚀 Analyze & Submit",
            key="submit_btn",
            use_container_width=True,
            type="primary"
        )
    
    # Process submission
    if submit_btn:
        if not description.strip():
            st.error("❌ Please provide a grievance description")
            return
        
        with st.spinner("🔄 Analyzing your grievance..."):
            api_client = get_api_client()
            
            result = api_client.analyze_grievance(
                description=description,
                location=location if location else None,
                category=category if category else None,
                contact_email=contact_email if contact_email else None,
                contact_phone=contact_phone if contact_phone else None
            )
        
        if result:
            st.success("✅ Grievance analyzed successfully!")
            st.markdown("---")
            
            # Display results
            st.subheader("📊 Analysis Results")
            
            render_grievance_card(result)
            
            # Additional details
            with st.expander("📋 Full Details", expanded=True):
                st.json(result)
            
            # Download option
            col1, col2 = st.columns(2)
            
            with col1:
                json_str = json.dumps(result, indent=2)
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_str,
                    file_name=f"grievance_{result['grievance_id']}.json",
                    mime="application/json"
                )
            
            # Clear form
            if st.button("📝 Submit Another Grievance"):
                st.rerun()
        else:
            st.error("❌ Failed to analyze grievance. Please try again.")
