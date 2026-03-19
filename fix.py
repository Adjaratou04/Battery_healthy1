import tensorflow as tf

model = tf.keras.models.load_model("models/lstm_soh_model.h5", compile=False)
model.save("models/lstm_soh_model.h5", save_format="h5")
print("Modele re-sauvegarde avec succes !")