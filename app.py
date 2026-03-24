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
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.markdown("<h1 style='text-align: center;'>💳 AI Fraud Detection Dashboard</h1>", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
data = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
X = data.drop('Class', axis=1)
y = data['Class']

# =========================
# HANDLE IMBALANCE
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
# TRAIN MULTIPLE MODELS
# =========================
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

results = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    trained_models[name] = model

best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

# =========================
# SIDEBAR NAVIGATION
# =========================
menu = st.sidebar.selectbox("📌 Navigation", ["Dashboard", "Manual Prediction", "Upload CSV"])

# =========================
# DASHBOARD
# =========================
if menu == "Dashboard":
    st.subheader("📊 Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(data))
    col2.metric("Fraud Cases", int(data['Class'].sum()))
    col3.metric("Best Model", best_model_name)

    # Fraud vs Normal Graph
    st.subheader("Fraud vs Normal Transactions")
    fig1, ax1 = plt.subplots()
    data['Class'].value_counts().plot(kind='bar', ax=ax1, color=['green', 'red'])
    ax1.set_xticklabels(['Normal', 'Fraud'], rotation=0)
    st.pyplot(fig1)

    # Model Accuracy Comparison
    st.subheader("📈 Model Accuracy Comparison")
    fig2, ax2 = plt.subplots()
    ax2.bar(results.keys(), results.values(), color='skyblue')
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Model Comparison")
    st.pyplot(fig2)

    # Confusion Matrix + Classification Report
    st.subheader(f"🧾 {best_model_name} Performance Metrics")
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    st.write("**Confusion Matrix:**")
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal','Fraud'], yticklabels=['Normal','Fraud'])
    ax3.set_ylabel('Actual')
    ax3.set_xlabel('Predicted')
    st.pyplot(fig3)

    st.write("**Classification Report:**")
    report = classification_report(y_test, y_pred_best, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

# =========================
# MANUAL PREDICTION
# =========================
elif menu == "Manual Prediction":
    st.subheader("🧾 Enter Transaction Details")
    feature_names = data.drop('Class', axis=1).columns
    user_input = []
    for feature in feature_names:
        val = st.number_input(f"{feature}", value=0.0)
        user_input.append(val)

    if st.button("Check Fraud"):
        user_input = np.array(user_input).reshape(1, -1)
        user_input = scaler.transform(user_input)
        prediction = best_model.predict(user_input)[0]
        probability = best_model.predict_proba(user_input)[0][1]

        if prediction == 1:
            st.error("🚨 Fraud Detected!")
        else:
            st.success("✅ Normal Transaction")
        st.write(f"Fraud Probability: {probability * 100:.2f}%")

# =========================
# CSV UPLOAD
# =========================
elif menu == "Upload CSV":
    st.subheader("📁 Upload CSV File")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.write("Preview:")
        st.dataframe(df.head())

        df_scaled = scaler.transform(df)
        predictions = best_model.predict(df_scaled)
        probabilities = best_model.predict_proba(df_scaled)[:, 1]

        df['Prediction'] = predictions
        df['Fraud Probability'] = probabilities
        st.write("Results:")
        st.dataframe(df)
        st.success(f"🚨 Fraud Cases Detected: {sum(predictions)}")