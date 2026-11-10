
import pickle
import streamlit as st

# Load saved model and vectorizer
with open("models/spam_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/tfidf_vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)


# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

st.title("📩 SMS Spam Detector")
st.write("Enter an SMS message to check whether it is spam or legitimate.")

# User input
message = st.text_area(
    "Enter your SMS message",
    placeholder="Example: Congratulations! You have won a free prize...",
    height=150
)

if st.button("Check Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        # Transform message using the saved TF-IDF vectorizer
        message_tfidf = tfidf.transform([message])

        # Predict class and probability
        prediction = model.predict(message_tfidf)[0]

        spam_index = list(model.classes_).index("spam")
        spam_probability = model.predict_proba(message_tfidf)[0][spam_index]

        if prediction == "spam":
            st.error("🚨 This message is predicted to be SPAM.")
        else:
            st.success("✅ This message is predicted to be HAM (legitimate).")

        st.write(f"Estimated spam probability: {spam_probability:.2%}")

st.caption("Machine learning predictions can be incorrect. Check suspicious messages carefully.")
