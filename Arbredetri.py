
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # index de la caractéristique à tester
        self.threshold = threshold  # seuil pour la séparation
        self.left = left  # sous-arbre gauche
        self.right = right  # sous-arbre droit
        self.value = value  # valeur de la feuille si c'est une feuille


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(n_samples_per_class)

        # Arrêt : tous les échantillons appartiennent à la même classe ou profondeur maximale atteinte
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            return Node(value=predicted_class)

        # Sélection de la meilleure caractéristique et du meilleur seuil
        best_feature, best_threshold = self._best_criteria(X, y)

        # Divise les données en fonction de la meilleure caractéristique et du seuil
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(feature_index=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_criteria(self, X, y):
        best_gini = np.inf
        best_feature, best_threshold = None, None

        for feature_index in range(self.n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_idxs, right_idxs = self._split(X[:, feature_index], threshold)
                gini = self._gini_impurity(y[left_idxs], y[right_idxs])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_threshold

    def _split(self, feature, threshold):
        left_idxs = np.where(feature <= threshold)[0]
        right_idxs = np.where(feature > threshold)[0]
        return left_idxs, right_idxs

    def _gini_impurity(self, left_y, right_y):
        p_left = len(left_y) / (len(left_y) + len(right_y))
        p_right = len(right_y) / (len(left_y) + len(right_y))
        gini = 1 - (p_left ** 2 + p_right ** 2)
        return gini

    def predict(self, X):
        return [self._traverse_tree(x, self.tree) for x in X]

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

df_train = pd.read_csv("bodym.csv")

# Extraction des noms de colonnes (étiquettes) et suppression de la première ligne
column_names = df_train.iloc[0]
df_train = df_train[1:]

# Séparation des données en X_train (caractéristiques) et y_train (étiquettes)
X_train = df_train.drop(columns=[ "photo_names", "subject_id", "gender"]).values
labels_row = df_train.iloc[0]
y_train = labels_row.values

# Suppression de la première ligne pour obtenir les données d'entraînement
df_train = df_train.drop(index=0)

# Lecture des données de test à partir du fichier CSV
df_test = pd.read_csv("test.csv")

# Extraction des noms de colonnes (étiquettes) et suppression de la première ligne
column_names_test = df_test.iloc[0]
df_test = df_test[1:]

# Extraction des données de test (sans les étiquettes)
X_test = df_test.values

# Entraînement de l'arbre de décision
tree_classifier = DecisionTreeClassifier(max_depth=2)
tree_classifier.fit(X_train, y_train)
# Prédiction pour les données de test
predictions = tree_classifier.predict(X_test)
print("Prédictions pour les données de test :", predictions)

