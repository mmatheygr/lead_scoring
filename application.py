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

# Modern UI styling
st.markdown("""
    <style>
    /* Modern color scheme */
    :root {
        --primary-color: #0073ea;
        --bg-color: #f6f7fb;
        --card-bg: #ffffff;
        --text-color: #323338;
        --secondary-text: #676879;
    }
    
    /* Global styles */
    .main {
        background-color: var(--bg-color);
        padding: 2rem;
    }
    
    /* Header styling */
    .stTitle {
        color: var(--text-color) !important;
        font-weight: 600 !important;
    }
    
    /* Card styling */
    .css-12oz5g7 {
        background-color: var(--card-bg);
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        border: none;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #0066cc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--card-bg);
    }
    
    /* Metric containers */
    .css-1r6slb0 {
        background-color: var(--card-bg);
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Table styling */
    .stDataFrame {
        background-color: var(--card-bg);
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .dataframe {
        font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen-Sans, Ubuntu, Cantarell, "Helvetica Neue", sans-serif;
    }
    
    /* Upload box styling */
    .uploadedFile {
        background-color: var(--card-bg);
        border-radius: 8px;
        border: 2px dashed #ccc;
        padding: 2rem;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: var(--primary-color);
    }
    
    /* Radio buttons */
    .stRadio > div {
        padding: 0.5rem;
        background-color: transparent;
    }
    
    /* Success/Info messages */
    .stSuccess, .stInfo {
        background-color: var(--card-bg);
        border: none;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
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
    # Sidebar navigation with modern styling
    with st.sidebar:
        st.markdown("""
            <div style='padding: 1rem 0; text-align: center;'>
                <h2 style='color: #323338; font-weight: 600; font-size: 1.5rem;'>Lead Scoring</h2>
            </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "",  # Empty label for cleaner look
            ["Upload Data", "Lead Analysis", "High-Value Leads", "Overview"],
            index=0
        )
    
    if page == "Upload Data":
        st.markdown("<h1 style='color: #323338;'>Upload Lead Data</h1>", unsafe_allow_html=True)
        
        # Modern upload container
        with st.container():
            uploaded_file = st.file_uploader(
                "Drop your CSV file here or click to upload",
                type=["csv"],
                help="Upload your lead data in CSV format"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success("✓ File uploaded successfully")
                    st.session_state['df'] = df
                    st.session_state['file_uploaded'] = True
                    
                    # Preview in a modern card
                    st.markdown("### Data Preview")
                    st.dataframe(df.head(), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    elif page == "Lead Analysis" and 'file_uploaded' in st.session_state:
        st.markdown("<h1 style='color: #323338;'>Lead Analysis</h1>", unsafe_allow_html=True)
        df = st.session_state['df']
        
        if st.button('Generate Predictions', key='analyze'):
            with st.spinner('Processing your data...'):
                predictions = predict_model(model, data=df, raw_score=True)
                probabilities = predictions['prediction_score_1']
                df['Purchase Probability'] = probabilities
                st.session_state['df_with_predictions'] = df
                
                # Modern grid layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Scoring Results")
                    styled_df = df[['Customer ID', 'Purchase Probability']].style.format({
                        'Purchase Probability': '{:.1%}'
                    }).background_gradient(cmap='Blues', subset=['Purchase Probability'])
                    st.dataframe(styled_df, use_container_width=True)
                
                with col2:
                    st.markdown("### Distribution")
                    fig = px.histogram(
                        df,
                        x='Purchase Probability',
                        nbins=20,
                        color_discrete_sequence=['#0073ea']
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=30, l=0, r=0, b=0),
                        xaxis_title="Conversion Probability",
                        yaxis_title="Number of Leads"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    elif page == "High-Value Leads" and 'df_with_predictions' in st.session_state:
        st.markdown("<h1 style='color: #323338;'>High-Value Leads</h1>", unsafe_allow_html=True)
        df = st.session_state['df_with_predictions']
        
        high_value_df = df[df['Purchase Probability'] >= PROBABILITY_THRESHOLD].sort_values(
            by='Purchase Probability', 
            ascending=False
        )
        
        # Modern metrics layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""
                <div style='background-color: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                    <h3 style='color: #676879; font-size: 0.9rem; margin: 0;'>HIGH-VALUE LEADS</h3>
                    <p style='color: #323338; font-size: 2rem; font-weight: 600; margin: 0;'>{len(high_value_df)}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            if len(high_value_df) > 0:
                avg_prob = high_value_df['Purchase Probability'].mean()
                st.markdown(
                    f"""
                    <div style='background-color: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                        <h3 style='color: #676879; font-size: 0.9rem; margin: 0;'>AVG PROBABILITY</h3>
                        <p style='color: #323338; font-size: 2rem; font-weight: 600; margin: 0;'>{avg_prob:.1%}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Priority leads table
        st.markdown("### Priority Leads")
        if len(high_value_df) > 0:
            styled_df = high_value_df[['Customer ID', 'Purchase Probability']].style.format({
                'Purchase Probability': '{:.1%}'
            }).background_gradient(cmap='Blues', subset=['Purchase Probability'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No high-value leads found above the 50% threshold.")
    
    elif page == "Overview":
        st.markdown("<h1 style='color: #323338;'>Platform Overview</h1>", unsafe_allow_html=True)
        
        # Modern cards layout
        st.markdown("""
            <div style='background-color: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 1rem;'>
                <h3 style='color: #323338; margin-top: 0;'>About the Platform</h3>
                <p style='color: #676879;'>This lead scoring platform helps identify and prioritize potential customers based on their likelihood to convert.</p>
            </div>
            
            <div style='background-color: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                <h3 style='color: #323338; margin-top: 0;'>How to Use</h3>
                <ol style='color: #676879;'>
                    <li>Upload your lead data CSV file in the Upload Data section</li>
                    <li>Generate predictions in the Lead Analysis section</li>
                    <li>Access your high-priority leads in the High-Value Leads section</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
    
    else:
        if 'file_uploaded' not in st.session_state:
            st.info("Please upload your data file to begin.")

if __name__ == "__main__":
    main()
