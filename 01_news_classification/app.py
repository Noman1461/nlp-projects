import os
from pathlib import Path
# Import NLTK and set data path to pre-downloaded directory
import nltk
NLTK_DATA_DIR = "/opt/render/nltk_data"
Path(NLTK_DATA_DIR).mkdir(parents=True, exist_ok=True)

# Download punkt BEFORE Flask app initialization
nltk.download('punkt', download_dir=NLTK_DATA_DIR)
nltk.download('stopwords', download_dir=NLTK_DATA_DIR)
nltk.download('wordnet', download_dir=NLTK_DATA_DIR)

nltk.data.path.append(NLTK_DATA_DIR)

from flask import Flask, render_template, request, jsonify
import joblib


# Load the trained pipeline
pipeline = joblib.load("models/news_classifier_pipeline.pkl")

# Label mapping
label_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Technology"
}

# Classification thresholds
CONFIDENCE_THRESHOLD = 0.60
MIN_WORDS = 5
MIN_CHARS = 30

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = (data.get("text", "") or "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Basic length checks
    if len(text.split()) < MIN_WORDS or len(text) < MIN_CHARS:
        return jsonify({
            "error": f"Please enter a longer news-style article (more than {MIN_WORDS} words)."
        }), 422

    try:
        probs = pipeline.predict_proba([text])[0]
        pred_num = int(pipeline.predict([text])[0])
    except AttributeError:
        pred_num = int(pipeline.predict([text])[0])
        return jsonify({"category": label_map.get(pred_num, "Unknown"), "probability": None})

    max_prob = float(probs.max())

    if max_prob < CONFIDENCE_THRESHOLD:
        return jsonify({
            "error": "Failed to classify the article with enough confidence. Please enter a clearer news article."
        }), 422

    return jsonify({
        "category": label_map.get(pred_num, "Unknown"),
        "probability": round(max_prob, 3)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
