import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE

# =========================
# DARK UI STYLE
# =========================
st.set_page_config(page_title="Fraud Detection PRO", layout="wide")

st.markdown("""
<style>
body {background-color: #0e1117; color: white;}
.metric-box {background: #1c1f26; padding: 15px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;color:#00C9A7;'>💳 Fraud Detection PRO</h1>", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")

data = load_data()

X = data.drop('Class', axis=1)
y = data['Class']

# =========================
# SMOTE
# =========================
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# =========================
# SPLIT + SCALE
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# TRAIN MODELS
# =========================
@st.cache_resource
def train_models():
    models = {
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier()
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = acc
        trained_models[name] = model

    best_model_name = max(results, key=results.get)
    return trained_models, results, best_model_name

trained_models, results, best_model_name = train_models()
best_model = trained_models[best_model_name]

# =========================
# SIDEBAR
# =========================
menu = st.sidebar.selectbox("📌 Navigation", ["Dashboard", "Manual", "Upload", "Insights"])
threshold = st.sidebar.slider("Fraud Threshold", 0.0, 1.0, 0.5)

# =========================
# DASHBOARD
# =========================
if menu == "Dashboard":

    st.subheader("📊 Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", len(data))
    col2.metric("Fraud", int(data['Class'].sum()))
    col3.metric("Best Model", best_model_name)

    # Pie Chart (Fintech style)
    st.subheader("💰 Fraud Distribution")
    fig, ax = plt.subplots()
    data['Class'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)

    # Accuracy Chart
    st.subheader("📈 Model Performance")
    fig2, ax2 = plt.subplots()
    ax2.plot(list(results.keys()), list(results.values()), marker='o')
    st.pyplot(fig2)

# =========================
# MANUAL
# =========================
elif menu == "Manual":

    st.subheader("🧾 Enter Transaction")

    feature_names = data.drop('Class', axis=1).columns
    user_input = []

    cols = st.columns(3)

    for i, feature in enumerate(feature_names):
        val = cols[i % 3].number_input(feature, value=0.0)
        user_input.append(val)

    if st.button("🎲 Sample"):
        sample = data.sample(1).drop('Class', axis=1).values[0]
        user_input = sample
        st.success("Sample Loaded")

    if st.button("🔍 Predict"):

        user_input = np.array(user_input).reshape(1, -1)
        user_input = scaler.transform(user_input)

        pred = best_model.predict(user_input)[0]
        prob = best_model.predict_proba(user_input)[0][1]

        # 🚨 ALERT SYSTEM
        if prob > 0.8:
            st.error(f"🚨 HIGH RISK FRAUD ({prob*100:.2f}%)")
        elif prob > threshold:
            st.warning(f"⚠️ Suspicious ({prob*100:.2f}%)")
        else:
            st.success(f"✅ Safe ({prob*100:.2f}%)")

# =========================
# UPLOAD CSV
# =========================
elif menu == "Upload":

    st.subheader("📁 Upload CSV")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        df_scaled = scaler.transform(df)
        preds = best_model.predict(df_scaled)
        probs = best_model.predict_proba(df_scaled)[:, 1]

        df['Prediction'] = preds
        df['Fraud Probability'] = probs

        st.dataframe(df)

        # 📥 DOWNLOAD BUTTON
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "fraud_results.csv")

        st.error(f"🚨 Total Frauds: {sum(preds)}")

# =========================
# INSIGHTS
# =========================
elif menu == "Insights":

    st.subheader("📊 Confusion Matrix")

    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
    st.pyplot(fig)

    st.subheader("📄 Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())