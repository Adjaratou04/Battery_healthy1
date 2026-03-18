
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    """
    Sélection des variables + normalisation
    """

    features = [
        "Voltage_measured",
        "Current_measured",
        "Temperature_measured",
        "SoC",
        "cycle_number"
    ]

    target = "SoH"

    X = df[features]
    y = df[target]

    # Normalisation
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, scaler