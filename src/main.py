
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from data_loader import load_data
from preprocessing import preprocess_data
from sequence import create_sequences
from train import split_data, train_model
from model import build_lstm_model
import matplotlib.pyplot as plt

# 1. Charger les données
df = load_data("data/battery_health_dataset.csv")

# 2. Prétraitement
X_scaled, y, scaler = preprocess_data(df)

# 3. Séquences
X_seq, y_seq = create_sequences(X_scaled, y, window_size=5)

# 4. Split
X_train, X_test, y_train, y_test = split_data(X_seq, y_seq)

# 5. Modèle
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_lstm_model(input_shape)

model.summary()

# 6. Entraînement
history = train_model(model, X_train, y_train)

import seaborn as sns

# 📊 1. Distribution du SoH
plt.figure()
plt.hist(df["SoH"], bins=30)
plt.title("Distribution du SoH")
plt.xlabel("SoH")
plt.ylabel("Fréquence")
plt.savefig("results/soh_distribution.png")
plt.show()


# 📊 2. Matrice de corrélation
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Matrice de corrélation")
plt.savefig("results/correlation_matrix.png")
plt.show()


# 📊 3. Courbe d’apprentissage
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Courbe d'apprentissage")
plt.savefig("results/learning_curve.png")
plt.show()


# 📊 4. Réel vs Prédit
y_pred = model.predict(X_test)

plt.figure()
plt.plot(y_test[:100], label="Réel")
plt.plot(y_pred[:100], label="Prédit")
plt.legend()
plt.title("Comparaison SoH réel vs prédit")
plt.savefig("results/pred_vs_real.png")
plt.show()
# Sauvegarde du modèle
model.save("models/lstm_soh_model.h5")

print("Modèle sauvegardé dans models/")

from train import evaluate_model

# Évaluation
mae, rmse, r2 = evaluate_model(model, X_test, y_test)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)


y_pred = model.predict(X_test)

plt.figure(figsize=(10,5))

plt.plot(y_test[:100], label="Réel")
plt.plot(y_pred[:100], label="Prédit")

# Ajout des métriques sur le graphique
plt.text(0, max(y_test[:100]),
         f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR2: {r2:.2f}",
         fontsize=10,
         bbox=dict(facecolor='white', alpha=0.7))

plt.legend()
plt.title("Comparaison SoH réel vs prédit")

plt.savefig("results/prediction_plot.png")
plt.show()

# Sauvegarder les métriques dans un fichier texte
with open("results/metrics.txt", "w") as f:
    f.write(f"MAE: {mae}\n")
    f.write(f"RMSE: {rmse}\n")
    f.write(f"R2: {r2}\n")

print("Résultats sauvegardés dans results/")