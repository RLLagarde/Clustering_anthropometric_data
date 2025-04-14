import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree



data = pd.read_csv("bodym.csv")   
data = data.drop(columns=[ "photo_names", "subject_id"])
data["gender"] = data["gender"].replace({"male": 0, "female": 1})
# Ensuite, vous pouvez appliquer la méthode pd.cut() avec des pas de 9 de 80 à 170 pour obtenir 10 catégories
data['hip'] = pd.cut(data['hip'], bins=range(80, 171, 15), labels=range(1, 7))


# Calcul des plages de valeurs pour chaque catégorie
def calculate_range(value):
    lower_bound = 80 + (value - 1) * 9
    upper_bound = 80 + value * 9
    return f"Sa taille est comprise entre {lower_bound} et {upper_bound}"

# Remplacement des valeurs dans la colonne "hip" par les plages de valeurs
data['hip'] = data['hip'].apply(lambda x: calculate_range(x))


# Séparation des caractéristiques (X) et des étiquettes (y)
X = data.drop(columns=['hip'])  # Caractéristiques
y = data['hip']  # Étiquettes


# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialisation et entraînement du modèle d'arbre de décision
tree_classifier = DecisionTreeClassifier(max_depth=3)
tree_classifier.fit(X_train, y_train)

# Affichage de l'arbre de décision
plt.figure(figsize=(20,10))  # Définition de la taille de la figure
plot_tree(tree_classifier, feature_names=X.columns, class_names=y.unique(), filled=True,  fontsize=7)

plt.show()
