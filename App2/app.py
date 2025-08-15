from flask import Flask, render_template, request
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import joblib
import numpy as np

app = Flask(__name__)

# Load DistilBERT model and tokenizer
model_path = "models/distilbert"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
bert_model = DistilBertForSequenceClassification.from_pretrained(model_path)
bert_model.eval()

# Load label encoder
label_encoder = joblib.load("models/label_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        notes = request.form["notes"]

        # Tokenize input
        inputs = tokenizer(notes, truncation=True, padding=True, max_length=256, return_tensors="pt")

        # Make prediction
        with torch.no_grad():
            outputs = bert_model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).item()

        # Decode label
        bert_pred = label_encoder.inverse_transform([preds])[0]

        return render_template("result.html", notes=notes, bert_pred=bert_pred)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
