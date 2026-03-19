import tensorflow as tf

model = tf.keras.models.load_model("lstm_soh_model.h5", compile=False)
model.save("lstm_soh_model_fixed.h5", save_format="h5")

print("OK - fichier créé : lstm_soh_model_fixed.h5")