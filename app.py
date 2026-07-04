from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.json["message"]
    vector = vectorizer.transform([message])
    prediction = model.predict(vector)[0]
    return jsonify({"result": prediction})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))