import tensorflow as tf

# Charger ancien modèle
model = tf.keras.models.load_model("models/lstm_soh_model.h5", compile=False)

# Sauvegarder nouveau format
model.save("models/model.keras")

print("✅ Conversion terminée")