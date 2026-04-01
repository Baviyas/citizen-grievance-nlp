"""
Batch Upload Page - Process Multiple Grievances
"""

import streamlit as st
from utils.api_client import get_api_client
from utils.ui_components import render_batch_results, render_grievance_card
import tempfile
import os

def show():
    """Display batch upload page"""
    st.title("📤 Batch Upload Grievances")
    st.markdown("---")
    
    st.markdown("""
    Upload a CSV file containing multiple grievances for bulk analysis.
    The system will process each one and provide routing and priority recommendations.
    """)
    
    # Instructions
    with st.expander("📖 CSV Format Instructions", expanded=True):
        st.markdown("""
        Your CSV file should have the following columns:
        
        **Required:**
        - `description` - The grievance complaint text
        
        **Optional:**
        - `location` - Where the incident occurred
        - `category` - Preferred department
        - `contact_email` - Complainant email
        - `contact_phone` - Complainant phone
        
        **Example:**
        ```
        description,location,category,contact_email,contact_phone
        "Potholes on Main Street",Manhattan,,john@example.com,+1-555-0123
        "Noise complaint at night",Brooklyn,,jane@example.com,+1-555-0456
        ```
        
        ⚠️ **Requirements:**
        - File must be in CSV format (not Excel)
        - Minimum 1 row of data
        - Maximum 1000 rows per upload
        - UTF-8 encoding
        """)
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV file with grievance data"
    )
    
    if uploaded_file:
        # Display file info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
        with col3:
            st.metric("File Type", uploaded_file.type)
        
        # Preview
        import pandas as pd
        
        try:
            df_preview = pd.read_csv(uploaded_file)
            
            st.subheader("📋 Preview (First 5 rows)")
            st.dataframe(df_preview.head(), use_container_width=True)
            
            st.metric("Total Rows", len(df_preview))
            
            # Validation
            if "description" not in df_preview.columns:
                st.error("❌ CSV must contain a 'description' column")
                return
            
            if len(df_preview) == 0:
                st.error("❌ CSV file is empty")
                return
            
            if len(df_preview) > 1000:
                st.error("❌ File exceeds maximum 1000 rows")
                return
            
            st.success("✅ File validation passed")
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return
        
        st.markdown("---")
        
        # Upload button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            process_btn = st.button(
                "🚀 Process Batch",
                key="process_batch",
                use_container_width=True,
                type="primary"
            )
        
        # Process batch
        if process_btn:
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
            
            try:
                with st.spinner(f"🔄 Processing {len(df_preview)} grievances..."):
                    api_client = get_api_client()
                    results = api_client.batch_analyze(tmp_path)
                
                if results:
                    st.success("✅ Batch processing completed!")
                    st.markdown("---")
                    
                    # Display summary
                    st.subheader("📊 Processing Summary")
                    render_batch_results(results)
                    
                    # Display individual results
                    if results["results"]:
                        st.subheader("📋 Detailed Results")
                        
                        tabs = st.tabs([
                            f"All ({len(results['results'])})",
                            "P1 (Critical)",
                            "P2 (High)",
                            "P3 (Medium)",
                            "P4 (Low)"
                        ])
                        
                        with tabs[0]:  # All
                            for grievance in results["results"]:
                                render_grievance_card(grievance)
                        
                        with tabs[1]:  # P1
                            p1_results = [g for g in results["results"] if g["priority_tier"] == "P1"]
                            if p1_results:
                                for grievance in p1_results:
                                    render_grievance_card(grievance)
                            else:
                                st.info("No P1 grievances")
                        
                        with tabs[2]:  # P2
                            p2_results = [g for g in results["results"] if g["priority_tier"] == "P2"]
                            if p2_results:
                                for grievance in p2_results:
                                    render_grievance_card(grievance)
                            else:
                                st.info("No P2 grievances")
                        
                        with tabs[3]:  # P3
                            p3_results = [g for g in results["results"] if g["priority_tier"] == "P3"]
                            if p3_results:
                                for grievance in p3_results:
                                    render_grievance_card(grievance)
                            else:
                                st.info("No P3 grievances")
                        
                        with tabs[4]:  # P4
                            p4_results = [g for g in results["results"] if g["priority_tier"] == "P4"]
                            if p4_results:
                                for grievance in p4_results:
                                    render_grievance_card(grievance)
                            else:
                                st.info("No P4 grievances")
                    
                    st.markdown("---")
                    
                    # Export results
                    import json
                    results_json = json.dumps(results, indent=2)
                    st.download_button(
                        label="📥 Download Full Results as JSON",
                        data=results_json,
                        file_name="batch_results.json",
                        mime="application/json"
                    )
                
                else:
                    st.error("❌ Failed to process batch. Please try again.")
            
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
