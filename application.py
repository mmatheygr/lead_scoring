import streamlit as st
import pandas as pd
#import shap
from pycaret.classification import load_model, predict_model
import plotly.graph_objects as go

# Load the pre-trained model
model = load_model('lead_scoring_modelo')

# Streamlit UI
st.title("Lead Scoring Application")

# File uploader
st.header("Subir Leads")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display the uploaded file as a table
    st.write("Uploaded data preview:")
    st.write(df.head())

    # Process the file when the "Process" button is clicked
    if st.button('Process'):
        # Make predictions on the uploaded data
        predictions = predict_model(model, data=df, raw_score=True)

         # Extract probabilities for class 1 (assuming binary classification)
        probabilities = predictions['prediction_label_1']  # Change to the column name corresponding to the positive class
        df['Purchase Probability'] = probabilities
                
        # Display the table with customer ids and purchase probabilities
        st.subheader("Customer Probability of Purchase")
        
        # Display a scrollable table of customer id and probability of purchase
        st.dataframe(df[['Customer ID', 'Purchase Probability']].set_index('Customer ID'))

        # Create a gauge chart for each customer to display the probability of purchase
        st.subheader("Purchase Probability Gauge")
        for index, row in df.iterrows():
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = row['Purchase Probability'],
                title = {'text': f"Customer {row['Customer ID']}"},
                gauge = {'axis': {'range': [0, 1]},
                         'bar': {'color': "darkblue"},
                         'steps': [
                             {'range': [0, 0.2], 'color': "red"},
                             {'range': [0.2, 0.7], 'color': "yellow"},
                             {'range': [0.7, 1], 'color': "green"}]
                        }
            ))
            st.plotly_chart(fig)

        # Shapley values display
        #st.subheader("Shapley Values for Selected Customer")
        #customer_id = st.selectbox("Select Customer ID", df['Customer ID'].unique())

        # Get the row for the selected customer
        #customer_row = df[df['Customer ID'] == customer_id].drop('Customer ID', axis=1)
        
        # Create a Shap explainer
        #explainer = shap.TreeExplainer(model)
        #shap_values = explainer.shap_values(customer_row)
        
        # Plot the Shapley values for the selected customer
        #shap.initjs()
        #st_shap = shap.force_plot(explainer.expected_value[1], shap_values[1], customer_row)
        #st.write(st_shap)

