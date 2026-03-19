import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Désactive Keras 3 pour forcer Keras 2
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf

# Charge ton modèle (remplace le nom si besoin)
model = tf.keras.models.load_model("models/lstm_soh_model.h5", compile=False)

# Re-sauvegarde en .h5 legacy
model.save("models/lstm_soh_model_v2.h5", save_format="h5")
print("OK - fichier créé : models/lstm_soh_model_v2.h5")