# from flask import Flask, render_template, request, jsonify
# import joblib

# # Load pipeline (tfidf transformer, logistic model)
# tfidf, model = joblib.load("logistic_pipeline.pkl")

# # numeric label -> display name
# label_map = {
#     1: "World",
#     2: "Sports",
#     3: "Business",
#     4: "Sci/Tech"
# }

# # Confidence threshold: tune this using validation set. 0.60 is a reasonable starting point.
# CONFIDENCE_THRESHOLD = 0.60
# MIN_WORDS = 5        # client + server side minimal "article-like" heuristic
# MIN_CHARS = 30

# app = Flask(__name__)

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()
#     text = (data.get("text", "") or "").strip()

#     if not text:
#         return jsonify({"error": "No text provided"}), 400

#     # simple heuristics to catch very short / nonsense inputs
#     word_count = len(text.split())
#     if word_count < MIN_WORDS or len(text) < MIN_CHARS:
#         return jsonify({"error": "Please enter a longer news-style article (more than {} words).".format(MIN_WORDS)}), 422

#     # transform, predict and compute confidences
#     X = tfidf.transform([text])
#     try:
#         probs = model.predict_proba(X)[0]
#     except AttributeError:
#         # model has no predict_proba (unlikely for logistic regression)
#         pred_num = int(model.predict(X)[0])
#         category = label_map.get(pred_num, "Unknown")
#         return jsonify({"category": category, "probability": None})

#     # predicted numeric label
#     pred_num = int(model.predict(X)[0])
#     max_prob = float(probs.max())

#     if max_prob < CONFIDENCE_THRESHOLD:
#         return jsonify({
#             "error": "Failed to classify the article with enough confidence. Please enter a clearer news article."
#         }), 422

#     category = label_map.get(pred_num, "Unknown")
#     return jsonify({"category": category, "probability": round(max_prob, 3)})

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import joblib

# Load full pipeline (preprocessor + tfidf + logistic model)
pipeline = joblib.load("models/news_classifier_pipeline.pkl")

# numeric label -> display name
label_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Technology"
}

# Confidence threshold: tune this using validation set. 0.60 is a reasonable starting point.
CONFIDENCE_THRESHOLD = 0.60
MIN_WORDS = 5        # client + server side minimal "article-like" heuristic
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

    # Simple heuristics to catch very short / nonsense inputs
    word_count = len(text.split())
    if word_count < MIN_WORDS or len(text) < MIN_CHARS:
        return jsonify({
            "error": f"Please enter a longer news-style article (more than {MIN_WORDS} words)."
        }), 422

    try:
        # Predict probabilities and label
        probs = pipeline.predict_proba([text])[0]
        pred_num = int(pipeline.predict([text])[0])
    except AttributeError:
        # Model has no predict_proba (unlikely for logistic regression)
        pred_num = int(pipeline.predict([text])[0])
        category = label_map.get(pred_num, "Unknown")
        return jsonify({"category": category, "probability": None})

    max_prob = float(probs.max())

    if max_prob < CONFIDENCE_THRESHOLD:
        return jsonify({
            "error": "Failed to classify the article with enough confidence. Please enter a clearer news article."
        }), 422

    category = label_map.get(pred_num, "Unknown")
    return jsonify({"category": category, "probability": round(max_prob, 3)})

if __name__ == "__main__":
    app.run(debug=True)
