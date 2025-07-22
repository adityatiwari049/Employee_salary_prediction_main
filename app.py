import streamlit as st
import pandas as pd
import joblib
import time
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_lottie import st_lottie

# Load model
model = joblib.load('models/salary_model.pkl')

# Load dataset for EDA charts
data = pd.read_csv('data/employee_data.csv')

# Lottie animation loader function
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie animation
lottie_coding = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_x62chJ.json")

# Page configuration
st.set_page_config(page_title="Salary Prediction", page_icon="💼", layout="centered")

# Main App
st.title("💼 Employee Salary Prediction")

# Animation at top
st_lottie(lottie_coding, speed=1, height=200, key="initial")

st.markdown("---")
st.subheader("📊 Enter Employee Details")

with st.form("salary_form"):
    experience = st.slider("Experience (Years)", 0, 20, 2)
    education = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])
    job_title = st.selectbox("Job Title", ["Software Engineer", "Senior Software Engineer", "Tech Lead", "Manager"])
    location = st.selectbox("Location", ["Hyderabad", "Bangalore", "Pune", "Chennai"])
    submitted = st.form_submit_button("Predict Salary 💰")

if submitted:
    st.info("Predicting Salary... Please wait")
    time.sleep(1)

    # Preprocess Input
    input_df = pd.DataFrame([[experience, education, job_title, location]], columns=["Experience", "Education_Level", "Job_Title", "Location"])
    
    # Predict directly using pipeline
    predicted_salary = model.predict(input_df)[0]

    # Display result
    st.success(f"🎉 Predicted Annual Salary: ₹ {int(predicted_salary):,}")
    st.balloons()

    st.markdown("---")
    st.caption("✅ Powered by RandomForest ML Model")

    st.markdown("## 📈 Salary Data Insights")

    # Salary Distribution Plot
    st.markdown("### 💵 Salary Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(data['Salary'], kde=True, color='skyblue', ax=ax1)
    ax1.set_title('Salary Distribution')
    st.pyplot(fig1)

    # Experience vs Salary
    st.markdown("### 🧑‍💻 Experience vs Salary")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x='Experience', y='Salary', data=data, hue='Education_Level', ax=ax2)
    ax2.set_title('Experience vs Salary by Education Level')
    st.pyplot(fig2)

    # Salary by Job Title
    st.markdown("### 🏷️ Salary by Job Title")
    fig3, ax3 = plt.subplots()
    sns.barplot(x='Job_Title', y='Salary', data=data, ax=ax3)
    ax3.set_title('Average Salary by Job Title')
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    # Salary by Location
    st.markdown("### 📍 Salary by Location")
    fig4, ax4 = plt.subplots()
    sns.boxplot(x='Location', y='Salary', data=data, ax=ax4)
    ax4.set_title('Salary by Location')
    st.pyplot(fig4)

st.markdown("<div class='footer'>Built with ❤️ by  Aditya Tiwari </div>", unsafe_allow_html=True)

