import streamlit as st
import pandas as pd
import shap
import plotly.graph_objects as go
import plotly.express as px
from pycaret.classification import load_model, predict_model

# Page configuration
st.set_page_config(
    page_title="Lead Scoring Platform",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        height: 3rem;
        border-radius: 5px;
    }
    .stProgress > div > div > div > div {
        background-color: #0066cc;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the pre-trained model
@st.cache_resource
def load_cached_model():
    return load_model('lead_scoring_modelo')

model = load_cached_model()

# Fixed threshold
PROBABILITY_THRESHOLD = 0.5

def main():
    # Title in main area
    st.title("Lead Scoring Platform")
    
    # Navigation in sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Select View",
            ["Upload Data", "Lead Analysis", "High-Value Leads", "Overview"]
        )
    
    # File upload - always visible at the top
    if page == "Upload Data":
        st.header("Upload Lead Data")
        uploaded_file = st.file_uploader(
            "Upload your CSV file with lead information",
            type=["csv"],
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully. Please proceed to Lead Analysis.")
                
                # Store the dataframe in session state
                st.session_state['df'] = df
                st.session_state['file_uploaded'] = True
                
            except Exception as e:
                st.error(f"Error uploading file: {str(e)}")
    
    elif page == "Lead Analysis" and 'file_uploaded' in st.session_state:
        st.header("Lead Analysis")
        df = st.session_state['df']
        
        if st.button('Generate Predictions'):
            with st.spinner('Processing leads...'):
                # Make predictions
                predictions = predict_model(model, data=df, raw_score=True)
                probabilities = predictions['prediction_score_1']
                df['Purchase Probability'] = probabilities
                st.session_state['df_with_predictions'] = df
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Lead Scoring Results")
                    styled_df = df[['Customer ID', 'Purchase Probability']].style.format({
                        'Purchase Probability': '{:.1%}'
                    })
                    st.dataframe(styled_df, use_container_width=True)
                
                with col2:
                    st.subheader("Probability Distribution")
                    fig = px.histogram(
                        df,
                        x='Purchase Probability',
                        nbins=20,
                        title='Distribution of Purchase Probabilities'
                    )
                    fig.update_layout(
                        xaxis_title="Probability of Purchase",
                        yaxis_title="Number of Leads"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    elif page == "High-Value Leads" and 'df_with_predictions' in st.session_state:
        st.header("High-Value Leads")
        df = st.session_state['df_with_predictions']
        
        # Filter high-value leads
        high_value_df = df[df['Purchase Probability'] >= PROBABILITY_THRESHOLD].sort_values(
            by='Purchase Probability', 
            ascending=False
        )
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total High-Value Leads", len(high_value_df))
        with col2:
            if len(high_value_df) > 0:
                st.metric("Average Probability", f"{high_value_df['Purchase Probability'].mean():.1%}")
        
        # Display high-value leads table
        st.subheader("Priority Leads for Contact")
        if len(high_value_df) > 0:
            styled_df = high_value_df[['Customer ID', 'Purchase Probability']].style.format({
                'Purchase Probability': '{:.1%}'
            })
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No high-value leads found above the 50% probability threshold.")
    
    elif page == "Overview":
        st.header("Platform Overview")
        st.write("""
        This lead scoring platform helps identify and prioritize potential customers based on their likelihood to convert. 
        
        How to use:
        1. Start by uploading your lead data CSV file in the Upload Data section
        2. Generate predictions in the Lead Analysis section
        3. Access your high-priority leads in the High-Value Leads section
        
        The system automatically identifies high-value leads with a conversion probability above 50%.
        """)
    
    else:
        if 'file_uploaded' not in st.session_state:
            st.info("Please upload your data file first.")

if __name__ == "__main__":
    main()
