from flask import Flask, request, jsonify
from model import train_and_predict

app = Flask(__name__)

@app.route("/")
def home():
    return "ML Flask App Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    result = train_and_predict(data["input"])
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
