from flask import Flask, request, render_template
#import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model
#model = tf.keras.models.load_model("model/skin_model.h5")

classes = ["acne","dry","normal","oily"]

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]

        img = Image.open(file).resize((128,128))
        img = np.array(img)/255.0
        img = img.reshape(1,128,128,3)

        prediction = model.predict(img)
        result = classes[np.argmax(prediction)]

        # Cosmetic suggestion
        if result == "dry":
         suggestion = (
        "Your skin is dry. It lacks moisture and may feel rough or flaky. "
        "Use a good moisturizer twice daily to keep your skin hydrated. "
        "Drink plenty of water and avoid using harsh soaps or hot water."
    )

        elif result == "oily":
         suggestion = (
        "Your skin is oily. It produces excess sebum which can make your face look shiny. "
        "Use an oil-free face wash twice daily to control oil. "
        "Avoid heavy creams and prefer lightweight, non-comedogenic products."
    )

        elif result == "acne":
         suggestion = (
        "Acne is detected on your skin. This may be caused by clogged pores or excess oil. "
        "Use a gentle cleanser and avoid touching your face frequently. "
        "Consider using products with salicylic acid or benzoyl peroxide."
    )

        else:
         suggestion = (
        "Your skin is normal. It is well-balanced and healthy. "
        "Maintain a simple skincare routine with regular cleansing and moisturizing. "
        "Protect your skin from sun exposure by using sunscreen daily."
    )
        return f"Skin: {result} | Suggestion: {suggestion}"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)