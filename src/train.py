# src/train.py

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def split_data(X, y, test_size=0.2):
    """
    Sépare les données en train et test
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)


def train_model(model, X_train, y_train):
    """
    Entraîne le modèle
    """
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )
    return history



def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle
    """

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return mae, rmse, r2