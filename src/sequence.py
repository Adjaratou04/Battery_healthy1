

import numpy as np

def create_sequences(data, target, window_size=5):
    """
    Transforme les données en séquences pour LSTM
    """

    X_seq = []
    y_seq = []

    for i in range(len(data) - window_size):
        X_seq.append(data[i:i+window_size])
        y_seq.append(target[i+window_size])

    return np.array(X_seq), np.array(y_seq)