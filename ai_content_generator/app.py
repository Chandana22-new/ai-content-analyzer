from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)
classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    result = classifier(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
