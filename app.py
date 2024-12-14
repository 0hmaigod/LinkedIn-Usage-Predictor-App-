import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import streamlit as st


st.markdown("### Welcome to LinkedIn Usage Predictor!")

st.markdown("## To find out if this individual will use LinkedIn or not, please enter their information below!")

def clean_sm(x):
   return np.where(x == 1, 1, 0)

def train_model():
    s = pd.read_csv("social_media_usage.csv")
    ss = pd.DataFrame(s)
    ss["sm_li"] = ss["web1h"].apply(clean_sm)
    ss["income"] = ss["income"].apply(lambda x: np.nan if x > 9 else x)
    ss["education"] = ss["educ2"].apply(lambda x: np.nan if x > 8 else x)
    ss["parent"] = np.where(ss["par"], 1, 0)
    ss["married"] = np.where(ss["marital"], 1, 0)
    ss["female"] = np.where(ss["gender"] == 2, 1, 0)
    ss["age"] = ss["age"].apply(lambda x: np.nan if x > 98 else x)
    ss_clean = ss.dropna()

    y = ss_clean["sm_li"]
    X = ss_clean[["income", "education", "age", "parent", "married", "female"]]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=450)

    # Train the Logistic Regression model
    lr = LogisticRegression(class_weight="balanced")
    lr.fit(X_train, y_train)
    return lr

# Initialize the model
lr = train_model()



def linkedin_model():
    st.title("LinkedIn User Predictor")
    income = st.selectbox("Income:", options=[1, 2, 3, 4, 5, 6, 7, 8, 9], help="1: Less than $10K, 2: $10K to $20K, 3: $20K to $30K, 4: $30K to $40K, 5: $40K to $50K, 6: $50K to $75K, 7: $75K to $100K, 8: $100K to $150K, 9: $150K or more")
    education = st.selectbox("Education Level:", options=[1, 2, 3, 4, 5, 6, 7, 8], help="1: Less than high school, 8: Postgraduate degree")
    age = st.number_input("Age:", min_value=0, max_value=120, help="Enter age in years")
    parent = st.selectbox("Parent (Is a parent of a child under 18?):", options=[0, 1], help="0: No, 1: Yes")
    married = st.selectbox("Marital Status:", options=[0, 1], help="0: Not Married, 1: Married")
    female = st.selectbox("Female (Is the person female?):", options=[0, 1], help="0: No, 1: Yes")

    if st.button("Predict"):
        user_data = [[income, education, age, parent, married, female]]

        prediction = lr.predict(user_data)
        prob = lr.predict_proba(user_data)[0][1]

        result = "LinkedIn User" if prediction[0] == 1 else "Not a LinkedIn User"
        st.write(f"There is a {prob * 100:.2f}% chance this person will be a LinkedIn User.")
        st.write(f"There is a {100 - prob * 100:.2f}% chance this person will **not** be a LinkedIn User.")

        st.bar_chart([prob, 1 - prob], width=500, height=300, use_container_width=True)
        chart_data = pd.DataFrame([{"Label": "LinkedIn User", "Probability": prob}, 
                                   {"Label": "Not a LinkedIn User", "Probability": 1 - prob}])
        
        st.write(chart_data)

        fig, ax = plt.subplots()
        ax.bar(chart_data["Label"], chart_data["Probability"], color=['blue', 'grey'])
        ax.set_ylabel("Probability")
        ax.set_title("LinkedIn User Prediction")
        st.pyplot(fig)

linkedin_model()







