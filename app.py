import streamlit as st
import joblib
import os

st.set_page_config(page_title="Review Analysis by Pratyush Patel", layout="wide")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "sentiment_model.pkl")
    return joblib.load(model_path)

model = load_model()

# ── Sample reviews in sidebar ─────────────────────────────────────────────────
sample_reviews = [
    ("🟢 Great product!", "Absolutely love this product. The quality is outstanding and it arrived two days early. Will definitely buy again!"),
    ("🟢 Excellent service", "The customer support team was incredibly helpful. They resolved my issue within minutes. Highly recommend this company."),
    ("🔴 Very disappointed", "The product stopped working after just three days. Terrible build quality and the return process was a nightmare."),
    ("🔴 Waste of money", "Nothing like the pictures. Cheap material, wrong size, and the seller ignored all my messages. Avoid at all costs."),
    ("🟡 It's okay", "Does what it says, nothing more. Packaging was fine and delivery was on time. Not great, not bad."),
    ("🟡 Average experience", "The product is decent for the price. I've seen better but also much worse. Probably wouldn't buy again but no complaints really."),
]

if "selected_review" not in st.session_state:
    st.session_state.selected_review = ""

with st.sidebar:
    st.markdown("### 📋 Sample Reviews")
    st.markdown("Click any review to load it into the text box.")
    st.markdown("---")
    for title, text in sample_reviews:
        if st.button(title, key=title, use_container_width=True):
            st.session_state.selected_review = text

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Review Analysis by Pratyush Patel")
st.markdown("Paste a product or service review below and click **Analyse** to find out if it is Positive, Negative, or Neutral.")

review = st.text_area(
    "Enter your review here:",
    height=160,
    placeholder="e.g. The product quality was great and shipping was fast!",
    value=st.session_state.selected_review,
)

if st.button("Analyse"):
    if not review.strip():
        st.warning("Please enter a review before clicking Analyse.")
    else:
        prediction = model.predict([review])[0]
        proba = model.predict_proba([review])[0]
        classes = model.classes_
        scores = dict(zip(classes, (proba * 100).round(1)))

        label_map = {
            "positive": ("Positive", "🟢", "#d4edda", "#155724"),
            "negative": ("Negative", "🔴", "#f8d7da", "#721c24"),
            "neutral":  ("Neutral",  "🟡", "#fff3cd", "#856404"),
        }

        label, icon, bg, fg = label_map[prediction]

        st.markdown(
            f"""
            <div style="background-color:{bg}; padding:20px; border-radius:10px; text-align:center; margin-top:20px;">
                <h2 style="color:{fg}; margin:0;">{icon} {label}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Confidence Scores")
        col1, col2, col3 = st.columns(3)
        col1.metric("Positive", f"{scores['positive']}%")
        col2.metric("Neutral",  f"{scores['neutral']}%")
        col3.metric("Negative", f"{scores['negative']}%")
