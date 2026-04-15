from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

app = Flask(__name__)
MODEL_PATH = "model/skin_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1bqwEkTMcJODBNfsUpAsdDDM3PFYaszYY"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

classes = ["acne","dry","normal","oily"]

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]

        img = Image.open(file).convert("RGB").resize((128,128))
        img = np.array(img)/255.0
        img = img.reshape(1,128,128,3)

        prediction = model.predict(img)
        result = classes[np.argmax(prediction)]

        if result == "dry":
            suggestion = "Your skin is dry. Use moisturizer and drink water."

        elif result == "oily":
            suggestion = "Your skin is oily. Use oil-free cleanser."

        elif result == "acne":
            suggestion = "You have acne. Use gentle cleanser and avoid touching face."

        else:
            suggestion = "Your skin is normal. Maintain simple skincare routine."

        return f"Skin: {result} | Suggestion: {suggestion}"

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)