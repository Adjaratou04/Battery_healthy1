# Battery State of Health (SoH) Prediction using LSTM

## 1. Introduction

Ce projet vise à prédire l’état de santé (State of Health - SoH) d’une batterie à partir de données électriques et thermiques, en utilisant un modèle de Deep Learning basé sur les réseaux LSTM.

Le SoH est un indicateur essentiel permettant d’évaluer la dégradation d’une batterie au cours du temps et d’anticiper sa fin de vie.

---

## 2. Contexte industriel

Les batteries sont au cœur de nombreux systèmes modernes :

- Véhicules électriques  
- Systèmes de stockage d’énergie  
- IoT et lampadaires solaires  

Le SoH permet de :

- Anticiper les pannes  
- Optimiser la maintenance  
- Réduire les coûts opérationnels  
- Prolonger la durée de vie des batteries  

---

## 3. Description du dataset

Le dataset contient des mesures issues de plusieurs batteries sur différents cycles de charge/décharge.

### Variables :

| Variable | Description |
|---------|------------|
| Voltage_measured | Tension mesurée |
| Current_measured | Courant mesuré |
| Temperature_measured | Température |
| SoC | State of Charge |
| cycle_number | Numéro du cycle |
| SoH | State of Health (variable cible) |

Les données sont séquentielles et représentent l’évolution des batteries dans le temps.

---

## 4. Problématique

Peut-on estimer le SoH d’une batterie à partir des mesures électriques, thermiques et du SoC ?

Type de problème :

- Régression  
- Données temporelles  
- Apprentissage supervisé  

---

## 5. Variables du modèle

### Variables d’entrée :

- Voltage_measured  
- Current_measured  
- Temperature_measured  
- SoC  
- cycle_number  

### Variable cible :

- SoH  

Ces variables sont choisies car elles influencent directement la dégradation des batteries.

---

## 6. Sliding Window

Les données sont transformées en séquences de taille 5 (fenêtres glissantes).

Objectifs :

- Capturer la dynamique temporelle  
- Augmenter le nombre d’échantillons  
- Permettre au modèle d’apprendre des motifs  

Chaque fenêtre correspond à un échantillon d’entrée pour le modèle.

---

## 7. Modèle utilisé : LSTM

Le modèle utilisé est un réseau de neurones LSTM (Long Short-Term Memory).

### Pourquoi LSTM ?

- Prend en compte l’ordre temporel  
- Capture les dépendances dans les séquences  
- Modélise des relations non linéaires  
- Adapté aux séries temporelles  

---

## 8. Pipeline du projet

1. Chargement des données  
2. Vérification et nettoyage  
3. Normalisation (MinMaxScaler)  
4. Création des séquences (sliding window)  
5. Séparation train/test  
6. Entraînement du modèle LSTM  
7. Prédiction  
8. Évaluation et visualisation  

---

## 9. Résultats

Les performances du modèle sont les suivantes :

- MAE ≈ 3.18  
- RMSE ≈ 4.26  
- R² ≈ 0.63  

Interprétation :

- Le modèle est globalement précis  
- Il capture correctement la tendance de dégradation  
- Les erreurs restent faibles  

---

## 10. Visualisation et interprétation

L’application génère plusieurs graphiques :

- Distribution du SoH : permet de visualiser l’état global des batteries  
- Matrice de corrélation : identifie les variables influentes  
- Évolution du SoH : montre la dégradation au fil des cycles  
- Réel vs Prédit : évalue la performance du modèle  

Chaque graphique est accompagné d’une interpration automatique.

---

## 11. Application web

Une application interactive a été développée avec Streamlit.

Fonctionnalités :

- Upload de fichier CSV ou Excel  
- Prédiction automatique du SoH  
- Visualisation des résultats  
- Interprétation automatique  
- Téléchargement des résultats  

---

## 12. Déploiement

L’application est déployée sur Render et accessible en ligne.

Lien de l’application :

https://battery-healthy1.onrender.com/

---

## 13. Utilisation de l’application

Étapes :

1. Accéder au lien de l’application  
2. Importer un fichier CSV ou Excel  
3. Vérifier les données affichées  
4. Observer les graphiques  
5. Analyser les résultats et interprétations  

---

## 14. Format du fichier attendu

Colonnes obligatoires :

- Voltage_measured  
- Current_measured  
- Temperature_measured  
- SoC  
- cycle_number  

Colonne optionnelle :

- SoH  

Si SoH est présent, une comparaison réel vs prédit est affichée.

---

## 15. Questions de réflexion

### 1. Pourquoi le SoC est-il une variable clé ?

Le SoC représente le niveau de charge de la batterie.  
Il influence directement son comportement électrique et permet de mieux interpréter les mesures.

---

### 2. Quel est l’intérêt des fenêtres glissantes ?

- Augmenter le nombre d’échantillons  
- Capturer les motifs locaux  
- Améliorer l’apprentissage du modèle  

---

### 3. Que se passe-t-il si la fenêtre est mal choisie ?

- Trop courte : perte d’information  
- Trop longue : bruit et complexité  

---

### 4. Quels sont les risques de biais ?

Si les mêmes batteries apparaissent dans les données d’entraînement et de test :

- Le modèle peut mémoriser  
- Les performances peuvent être surestimées  

---

### 5. Dans quels cas industriels ce modèle est pertinent ?

- Maintenance prédictive  
- Gestion des batteries dans les véhicules électriques  
- Optimisation des systèmes de stockage  

---

## 16. Conclusion

Ce projet démontre qu’un modèle LSTM peut prédire efficacement le SoH des batteries à partir de données électriques et thermiques.

Il combine :

- Deep Learning  
- Analyse de données  
- Visualisation  
- Déploiement web  

Des améliorations possibles incluent :

- Meilleure séparation train/test  
- Optimisation des hyperparamètres  
- Utilisation de modèles plus avancés  

---

## 17. Exécution en local (optionnel)

```bash
pip install -r requirements.txt
streamlit run app.py
