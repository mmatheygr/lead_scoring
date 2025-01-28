import streamlit as st
import pandas as pd
import shap
import plotly.graph_objects as go
import plotly.express as px
from pycaret.classification import load_model, predict_model

# Page configuration
st.set_page_config(
    page_title="Construction Lead Scoring",
    page_icon="🏗️",
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

# Header section with company logo placeholder
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("🏗️ Construction Lead Scoring Platform")
    st.markdown("Transform your leads into valuable opportunities with AI-powered insights")

# Main content area
def main():
    # Sidebar for filters and controls
    with st.sidebar:
        st.header("📊 Dashboard Controls")
        st.markdown("---")
        probability_threshold = st.slider(
            "Probability Threshold for High-Value Leads",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="Adjust this value to filter high-potential leads"
        )

    # File upload section
    st.header("📁 Upload Lead Data")
    uploaded_file = st.file_uploader(
        "Upload your CSV file with lead information",
        type=["csv"],
        help="Make sure your CSV contains all required features for the model"
    )

    if uploaded_file is not None:
        try:
            # Load and process data
            df = pd.read_csv(uploaded_file)
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📈 Lead Analysis", "🎯 Predictions", "📊 Insights"])
            
            with tab1:
                st.subheader("Data Overview")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 📋 Lead Data Preview")
                    st.dataframe(df.head(), use_container_width=True)
                with col2:
                    st.markdown("### 📊 Data Statistics")
                    st.write(df.describe())

            with tab2:
                if st.button('🚀 Generate Predictions', key='predict'):
                    with st.spinner('Processing your leads...'):
                        # Make predictions
                        predictions = predict_model(model, data=df, raw_score=True)
                        probabilities = predictions['prediction_score_1']
                        df['Purchase Probability'] = probabilities
                        
                        # Create visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🎯 Lead Scoring Results")
                            # Format probability as percentage and color-code
                            def highlight_high_value(val):
                                color = 'green' if val >= probability_threshold else 'black'
                                return f'color: {color}'
                            
                            styled_df = df[['Customer ID', 'Purchase Probability']].style.format({
                                'Purchase Probability': '{:.1%}'
                            }).applymap(highlight_high_value, subset=['Purchase Probability'])
                            
                            st.dataframe(styled_df, use_container_width=True)
                        
                        with col2:
                            st.markdown("### 📊 Probability Distribution")
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
                        
                        # Summary metrics
                        st.markdown("### 📈 Key Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            high_value_leads = len(df[df['Purchase Probability'] >= probability_threshold])
                            st.metric("High-Value Leads", high_value_leads)
                        with col2:
                            avg_prob = df['Purchase Probability'].mean()
                            st.metric("Average Probability", f"{avg_prob:.1%}")
                        with col3:
                            total_leads = len(df)
                            st.metric("Total Leads", total_leads)
                        with col4:
                            conversion_potential = high_value_leads / total_leads
                            st.metric("Potential Conversion Rate", f"{conversion_potential:.1%}")

            with tab3:
                st.markdown("### 🔍 Advanced Insights")
                if 'Purchase Probability' in df.columns:
                    # Top features analysis (placeholder - customize based on your model's features)
                    st.markdown("#### Most Influential Factors")
                    # Add your feature importance visualization here
                    st.info("Feature importance analysis can be customized based on your specific model and requirements")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
            st.info("Please ensure your CSV file contains all required features for the model")

    else:
        # Landing page content when no file is uploaded
        st.info("👆 Upload your CSV file to get started with lead scoring")
        
        # Feature showcase
        st.markdown("### 🌟 Key Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🎯 Accurate Predictions")
            st.write("Our AI model analyzes multiple factors to predict lead conversion probability")
            
        with col2:
            st.markdown("#### 📊 Visual Insights")
            st.write("Interactive visualizations help you understand your lead data better")
            
        with col3:
            st.markdown("#### 💡 Smart Filtering")
            st.write("Focus on high-potential leads with customizable probability thresholds")

if __name__ == "__main__":
    main()
