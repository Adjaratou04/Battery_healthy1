

import pandas as pd

def load_data(path):
    """
    Charge le dataset depuis un fichier CSV
    """
    df = pd.read_csv(path)
    return df