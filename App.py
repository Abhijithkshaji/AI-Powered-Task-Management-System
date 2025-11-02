import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import spacy
from scipy.sparse import hstack, csr_matrix

# -------------------------------------------------
# 1️⃣ Load saved models and encoders
# -------------------------------------------------
@st.cache_resource
def load_models():
    models = {
        "tfidf": joblib.load("tfidf_vectorizer.pkl"),
        "task_model": joblib.load("task_type_model.pkl"),
        "priority_model": joblib.load("priority_model.pkl"),
        "user_model": joblib.load("user_assignment_model.pkl"),
        "task_le": joblib.load("task_le.pkl"),
        "priority_le": joblib.load("priority_le.pkl"),
        "user_le": joblib.load("user_le.pkl")
    }

    # Load department dummy columns if available
    try:
        models["dept_dummies"] = joblib.load("dept_dummies.pkl")
    except:
        models["dept_dummies"] = []  # fallback if not saved
    return models


models = load_models()

# -------------------------------------------------
# 2️⃣ Load spaCy model
# -------------------------------------------------
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()


# -------------------------------------------------
# 3️⃣ Text Cleaning and Lemmatization
# -------------------------------------------------
def clean_and_lemmatize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and len(token.lemma_) > 2
    ]
    return " ".join(tokens)


# -------------------------------------------------
# 4️⃣ Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="AI Task Management Predictor", page_icon="🤖", layout="wide")
st.title(" AI Task Management Prediction Dashboard 🤖")
st.markdown("Predict **Task Type**, **Priority**, and **User Assignment** using trained ML models.")

with st.sidebar:
    st.header("📝 Enter Task Details")
    task_name = st.text_input("Task Name", "")
    task_tags = st.text_input("Tags", "")
    department = st.text_input("Department", "")
    estimated_hours = st.number_input("Estimated Hours", min_value=0, value=0)
    actual_hours = st.number_input("Actual Hours", min_value=0.0, value=0.0)
    days_until_due = st.number_input("Days Until Due", min_value=0, value=0)
    submit = st.button("🔮 Predict")


# -------------------------------------------------
# 5️⃣ Initialize Prediction History
# -------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []


# -------------------------------------------------
# 6️⃣ Prediction Logic
# -------------------------------------------------
if submit:
    with st.spinner("Analyzing and predicting..."):
        # Step 1 — Clean & vectorize text
        combined_text = clean_and_lemmatize(task_name + " " + task_tags)
        X_text = models["tfidf"].transform([combined_text])

        # Step 2 — Predict Task Type (SVM)
        task_pred_label = models["task_model"].predict(X_text)[0]
        task_pred = models["task_le"].inverse_transform([task_pred_label])[0]

        # Step 3 — Predict Priority (Random Forest)
        X_priority = hstack([
            X_text,
            csr_matrix(np.array([[estimated_hours, actual_hours, days_until_due]]))
        ])
        priority_label = models["priority_model"].predict(X_priority)[0]
        priority_pred = models["priority_le"].inverse_transform([priority_label])[0]

        # Step 4 — Department one-hot vector (align with training)
        dept_cols = models.get("dept_dummies", [])
        if dept_cols:
            dept_vector = pd.DataFrame(columns=dept_cols, data=[[0]*len(dept_cols)])
            if department in dept_vector.columns:
                dept_vector.loc[0, department] = 1
        else:
            dept_vector = pd.DataFrame([[0]], columns=[department])

        # Step 5 — Build full feature vector for user assignment
        X_user = hstack([
            X_text,
            csr_matrix(np.array([[estimated_hours, actual_hours, days_until_due]])),
            csr_matrix(dept_vector.values.astype(float)),
            csr_matrix(np.array([[task_pred_label, priority_label]]))
        ])

        # Step 6 — Align feature count with model expectations
        expected_features = models["user_model"].n_features_in_
        current_features = X_user.shape[1]
        if current_features < expected_features:
            diff = expected_features - current_features
            X_user = hstack([X_user, csr_matrix(np.zeros((1, diff)))])
        elif current_features > expected_features:
            X_user = X_user[:, :expected_features]

        # Step 7 — Predict assigned user
        user_label = models["user_model"].predict(X_user)[0]
        assigned_user = models["user_le"].inverse_transform([user_label])[0]

        # Step 8 — Top keywords by TF-IDF
        feature_array = np.array(models["tfidf"].get_feature_names_out())
        tfidf_sorting = np.argsort(X_text.toarray()).flatten()[::-1]
        top_keywords = feature_array[tfidf_sorting][:10]

        # Step 9 — Store result in session history
        st.session_state["history"].append({
            "Task Name": task_name,
            "Tags": task_tags,
            "Department": department,
            "Estimated Hours": estimated_hours,
            "Actual Hours": actual_hours,
            "Days Until Due": days_until_due,
            "Predicted Task Type": task_pred,
            "Priority": priority_pred,
            "Assigned User": assigned_user
        })

        # Step 10 — Display results
        st.success("✅ Prediction Complete!")
        col1, col2, col3 = st.columns(3)
        col1.metric("🧩 Task Type", task_pred)
        col2.metric("⚡ Priority", priority_pred)
        col3.metric("👤 Assigned User", assigned_user)

        st.markdown("---")
        st.subheader("🔑 Top Keywords (by TF-IDF weight)")
        st.write(", ".join(top_keywords))

        st.markdown("---")
        st.markdown(f"**🧼 Cleaned Input Text:** `{combined_text}`")


# -------------------------------------------------
# 7️⃣ Show Prediction History + Download
# -------------------------------------------------
if len(st.session_state["history"]) > 0:
    st.markdown("### 📜 Prediction History")
    df_hist = pd.DataFrame(st.session_state["history"])
    st.dataframe(df_hist, use_container_width=True)

    # Download button
    csv = df_hist.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Prediction History (CSV)",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )
else:
    st.info("No predictions yet — make one to see the history table.")


# -------------------------------------------------
# 8️⃣ Notes for Setup
# -------------------------------------------------
# st.markdown("""
# ---
# ### ⚙️ **Usage Instructions**
# 1. Place these files in the same folder:
#    - `task_type_model.pkl`
#    - `priority_model.pkl`
#    - `user_assignment_model.pkl`
#    - `tfidf_vectorizer.pkl`
#    - `task_le.pkl`
#    - `priority_le.pkl`
#    - `user_le.pkl`
#    - *(Optional)* `dept_dummies.pkl`

# 2. Run:
#    ```bash
#    streamlit run app.py
