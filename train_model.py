import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Training data: (review text, label)
training_data = [
    # Positive
    ("This product is absolutely amazing, I love it!", "positive"),
    ("Great quality, fast shipping, very happy with my purchase.", "positive"),
    ("Exceeded my expectations, highly recommend to everyone.", "positive"),
    ("Fantastic experience, the staff was super helpful and kind.", "positive"),
    ("Best purchase I've made this year, works perfectly.", "positive"),
    ("Wonderful product, easy to use and great results.", "positive"),
    ("Really impressed with the quality, will buy again.", "positive"),
    ("Outstanding service and the product is top notch.", "positive"),
    ("Loved every bit of it, totally worth the money.", "positive"),
    ("Five stars! Delivery was quick and packaging was excellent.", "positive"),
    ("I'm so happy with this, it works exactly as described.", "positive"),
    ("Perfect gift, the recipient was thrilled with it.", "positive"),
    ("Superb quality, beautiful design, very satisfied.", "positive"),
    ("Brilliant product, easy setup and works flawlessly.", "positive"),
    ("Amazing value for the price, very pleased.", "positive"),

    # Negative
    ("This is the worst product I have ever bought, total waste of money.", "negative"),
    ("Terrible quality, broke after just two days of use.", "negative"),
    ("Very disappointed, it does not work as advertised at all.", "negative"),
    ("Poor customer service, they refused to help me with the issue.", "negative"),
    ("Absolute garbage, do not buy this under any circumstances.", "negative"),
    ("The item arrived damaged and the company ignored my complaint.", "negative"),
    ("Completely useless product, nothing like the description.", "negative"),
    ("Worst experience ever, I want a full refund immediately.", "negative"),
    ("Horrible product, it stopped working within a week.", "negative"),
    ("Very bad quality, the material feels cheap and flimsy.", "negative"),
    ("Awful smell when opened, clearly defective item.", "negative"),
    ("Disappointed with the purchase, not worth the price at all.", "negative"),
    ("Does not work, returned it immediately.", "negative"),
    ("Terrible packaging, item was broken on arrival.", "negative"),
    ("Waste of money, would not recommend to anyone.", "negative"),

    # Neutral
    ("The product is okay, nothing special about it.", "neutral"),
    ("It works as expected, not great but not bad either.", "neutral"),
    ("Average quality for the price, gets the job done.", "neutral"),
    ("Decent product, does what it says on the box.", "neutral"),
    ("It is fine, met my basic needs without any issues.", "neutral"),
    ("Neither impressive nor disappointing, just an average item.", "neutral"),
    ("Shipping was on time and the product is functional.", "neutral"),
    ("Works fine, nothing to complain about but nothing exciting either.", "neutral"),
    ("It is acceptable, I have seen better and I have seen worse.", "neutral"),
    ("Reasonable product, does its job adequately.", "neutral"),
    ("Product is standard, no surprises good or bad.", "neutral"),
    ("Fairly typical product, serves the purpose.", "neutral"),
    ("It does the job, would be hard to say I love or hate it.", "neutral"),
    ("Mediocre but functional, not something I would rave about.", "neutral"),
    ("Average experience overall, nothing stood out to me.", "neutral"),
]

texts = [item[0] for item in training_data]
labels = [item[1] for item in training_data]

# Build and train pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])

pipeline.fit(texts, labels)

print("=== Training Complete ===\n")
print("Classification Report on Training Data:")
preds = pipeline.predict(texts)
print(classification_report(labels, preds, target_names=["negative", "neutral", "positive"]))

# Save model
joblib.dump(pipeline, "sentiment_model.pkl")
print("Model saved to sentiment_model.pkl\n")

# Example predictions
examples = [
    "I absolutely love this product, it changed my life!",
    "This is terrible, I want my money back.",
    "It is an okay product, nothing special.",
    "Horrible experience, never buying again.",
    "Pretty good overall, met my expectations.",
]

print("=== Example Predictions ===\n")
for text in examples:
    pred = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    confidence = dict(zip(classes, (proba * 100).round(1)))
    print(f"Review  : {text}")
    print(f"Result  : {pred.upper()}")
    print(f"Scores  : {confidence}\n")
