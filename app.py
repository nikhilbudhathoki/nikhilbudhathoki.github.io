import streamlit as st
import numpy as np
import pickle
import os

# Load the model (Make sure the model is in the same directory as your app.py or adjust the path accordingly)
MODEL_PATH = 'train_model.sav'

# Function to load the model
def load_model(model_path):
    if not os.path.isfile(model_path):
        st.error(f"Model file '{model_path}' not found.")
        st.stop()
    try:
        model = pickle.load(open(model_path, 'rb'))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model
loaded_model = load_model(MODEL_PATH)

# Function to handle prediction
def predict_loan(input_data):
    # Combine ApplicantIncome and CoapplicantIncome
    input_data.append(input_data[5] + input_data[6])
    input_data.pop(5)
    input_data.pop(6)
    
    # Convert list to numpy array and reshape for prediction
    input_data = np.asarray(input_data).reshape(1, -1)
    
    # Make prediction
    result = loaded_model.predict(input_data)
    
    # Interpret result
    status = "✅ Loan Accepted" if result == 1 else "❌ Loan Not Accepted"
    
    return status

# Streamlit main function
def main():
    # Sidebar
    st.sidebar.title("Bank Loan Predictor")
    st.sidebar.write("---")  # A line separator for better design

    # Create a stylish button-like navigation using markdown
    page = st.sidebar.radio("", ["Main Model", "Model Info", "About"], index=0, format_func=lambda x: f'**{x}**')

    if page == "Main Model":
        st.title("🏦 Loan Approval Prediction Webpage")
        st.write("Fill out the form below to see if your loan application will be approved.")
        
        with st.form(key='loan_form'):
            Gender = st.selectbox("Gender:", ["Male", "Female"], help="Select your gender.")
            Married = st.selectbox("Marital Status:", ["Yes", "No"], help="Are you married?")
            Dependents = st.selectbox("Dependents:", ["0", "1", "2", "3+"], help="Number of dependents.")
            Education = st.selectbox("Education Level:", ["Graduate", "Not Graduate"], help="Highest level of education.")
            Self_Employed = st.selectbox("Self Employed:", ["Yes", "No"], help="Are you self-employed?")
            
            ApplicantIncome = st.number_input("Applicant's Income:", min_value=0, help="Enter your income in USD.", step=100)
            CoapplicantIncome = st.number_input("Coapplicant's Income:", min_value=0, help="Enter the income of the coapplicant in USD.", step=100)
            LoanAmount = st.number_input("Loan Amount Requested:", min_value=0, help="Enter the loan amount you are requesting in USD.", step=1000)
            Loan_Amount_Term = st.number_input("Loan Amount Term (months):", min_value=0, max_value=480, value=360, help="Enter the loan repayment term in months.")
            Credit_History = st.selectbox("Credit History:", [1.0, 0.0], help="Select your credit history status. 1.0 for good history, 0.0 for bad.")
            Property_Area = st.selectbox("Property Area:", ["Urban", "Semiurban", "Rural"], help="Select the area where the property is located.")
            
            submit_button = st.form_submit_button(label='Check Loan Status')
        
        if submit_button:
            input_data = [
                1 if Gender == "Male" else 0,
                1 if Married == "Yes" else 0,
                int(Dependents.replace("3+", "3")),
                1 if Education == "Graduate" else 0,
                1 if Self_Employed == "Yes" else 0,
                ApplicantIncome,
                CoapplicantIncome,
                LoanAmount,
                Loan_Amount_Term,
                Credit_History,
                2 if Property_Area == "Urban" else 1 if Property_Area == "Semiurban" else 0
            ]

            # Call the prediction function
            result = predict_loan(input_data)
            
            # Display the prediction result with style
            st.markdown(f"## {result}")

    elif page == "Model Info":
        st.title("Model Information")
        
        st.write("---")  # A line separator for better design
        
        st.markdown("""
        This application predicts the approval status of a bank loan based on the provided inputs. 
        It uses a pre-trained machine learning model to make the predictions. 
        Fill out the form on the main page to check your loan status.

        **How to use this application:**
        1. Navigate to the "Main Model" page.
        2. Fill out the form with your details.
        3. Click the "Check Loan Status" button to see the prediction.

        **Explanation of Features:**
        - **Gender:** Select your gender.
        - **Marital Status:** Indicate if you are married.
        - **Dependents:** Number of dependents you have (e.g., children).
        - **Education Level:** Your highest level of education.
        - **Self Employed:** Indicate if you are self-employed.
        - **Applicant's Income:** Your income in USD.
        - **Coapplicant's Income:** Income of the coapplicant in USD.
        - **Loan Amount Requested:** The loan amount you are requesting in USD.
        - **Loan Amount Term:** The loan repayment term in months.
        - **Credit History:** Your credit history status (1.0 for good, 0.0 for bad).
        - **Property Area:** The area where the property is located (Urban, Semiurban, Rural).

        **Additional Information:**
        - **Loan Amount Term:** The duration over which you will repay the loan, in months.
        - **Loan Amount:** The total amount of money you are requesting as a loan.
        - **Dependents:** The number of people who rely on you financially.
        - **Self Employed:** Whether you work for yourself or not.
        - **Coapplicant:** A person who applies for the loan with you, sharing the responsibility of repayment.
        - **Applicant:** The primary person applying for the loan.
        - **Credit History:** A record of your ability to repay debts and demonstrated responsibility in repaying loans.
        """)

    elif page == "About":
        st.title("About This Application")
        
        st.write("---")  # A line separator for better design
        
        st.markdown("""
        **Developer:** Nikhil Budhathoki           
        **Email:** bcnikhil844@gmail.com  
        

        **Disclaimer:**
        This project is part of an internship supported by Mindrisers Institute of Technology, Kathmandu.  
        For more information, visit Mindrisers Institute of Technology.
        """)

if __name__ == '__main__':
    main()
