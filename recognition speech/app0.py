app khdama from flask import Flask, render_template, request
import os
import tensorflow as tf
from model_loader import model as model_en, encode_single_sample, decode_batch_predictions
from model_darija import transcribe_darija

app = Flask(_name_)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="Aucun fichier reçu.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="Aucun fichier sélectionné.")

        langue = request.form.get("langue")  # ⬅️ Langue choisie
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        try:
            if langue == "en":
                # Modèle anglais
                spectrogram = encode_single_sample(file_path)
                spectrogram = tf.cast(spectrogram, tf.float32)
                X = tf.expand_dims(spectrogram, axis=0)
                pred = model_en.predict(X)
                prediction = decode_batch_predictions(pred)[0]
            elif langue == "darija":
                # Modèle darija
                prediction = transcribe_darija(file_path)
            elif langue == "ar":
                # Modèle darija
                prediction = transcribe_arabe(file_path)
            else:
                prediction = "Langue inconnue."

        except Exception as e:
            prediction = f"Erreur : {str(e)}"

        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
            tf.keras.backend.clear_session()

    return render_template("index.html", prediction=prediction)

if _name_ == "_main_":
    app.run(debug=True)