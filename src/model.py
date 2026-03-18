# src/model.py

import tensorflow as tf

# Raccourcis pour les couches
Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense


def build_lstm_model(input_shape):
    """
    Crée et compile le modèle LSTM
    """

    model = Sequential()

    # Couche LSTM (apprentissage des séquences)
    model.add(LSTM(50, input_shape=input_shape))

    # Sortie → prédiction du SoH
    model.add(Dense(1))

    # Compilation
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model