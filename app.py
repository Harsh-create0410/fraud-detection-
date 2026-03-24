import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Fraud Detection AI", layout="wide")

# =========================
# SIMPLE LOGIN SYSTEM
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("Login successful ✅")
        else:
            st.error("Invalid credentials ❌")

if not st.session_state.logged_in:
    login()
    st.stop()

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.main {background-color: #0e1117; color: white;}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #1c1f26;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

st.title("💳 AI Fraud Detection PRO")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")
    return df.sample(8000)

data = load_data()

# =========================
# PREPROCESS
# =========================
@st.cache_data
def preprocess(df):
    X = df.drop('Class', axis=1)
    y = df['Class']

    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

X_train, X_test, y_train, y_test, scaler = preprocess(data)

# =========================
# MODEL
# =========================
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X, y)
    return model

model = train_model(X_train, y_train)

# =========================
# HISTORY STORAGE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# SIDEBAR
# =========================
menu = st.sidebar.radio("📌 Navigation", ["Dashboard", "Predict", "Upload CSV", "History", "Logout"])

# =========================
# DASHBOARD
# =========================
if menu == "Dashboard":

    st.subheader("📊 Overview")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"<div class='card'>Total<br><h2>{len(data)}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'>Fraud<br><h2>{int(data['Class'].sum())}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'>Model<br><h2>Random Forest</h2></div>", unsafe_allow_html=True)

    # Graph
    st.subheader("Fraud vs Normal")
    fig, ax = plt.subplots()
    data['Class'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    # Accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Accuracy: {acc:.4f}")

# =========================
# PREDICTION
# =========================
elif menu == "Predict":

    st.subheader("🧾 Enter Transaction")

    features = data.drop('Class', axis=1).columns
    user_input = []

    for feature in features:
        val = st.number_input(feature, value=0.0)
        user_input.append(val)

    if st.button("Predict"):

        user_input = np.array(user_input).reshape(1, -1)
        user_input = scaler.transform(user_input)

        pred = model.predict(user_input)[0]
        prob = model.predict_proba(user_input)[0][1]

        result = "Fraud" if pred == 1 else "Normal"

        # Save history
        st.session_state.history.append({
            "Result": result,
            "Probability": prob
        })

        if pred == 1:
            st.error(f"🚨 Fraud Detected ({prob*100:.2f}%)")
            st.warning("⚠️ Alert: Suspicious Transaction!")
        else:
            st.success(f"✅ Normal ({prob*100:.2f}%)")

# =========================
# CSV UPLOAD
# =========================
elif menu == "Upload CSV":

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        df_scaled = scaler.transform(df)
        preds = model.predict(df_scaled)
        probs = model.predict_proba(df_scaled)[:, 1]

        df['Prediction'] = preds
        df['Fraud Probability'] = probs

        st.dataframe(df)
        st.success(f"🚨 Fraud Cases: {sum(preds)}")

# =========================
# HISTORY PAGE
# =========================
elif menu == "History":

    st.subheader("📜 Prediction History")

    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history))
    else:
        st.info("No history yet")

# =========================
# LOGOUT
# =========================
elif menu == "Logout":
    st.session_state.logged_in = False
    st.success("Logged out")
    st.stop()