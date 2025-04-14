import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Charger les données depuis le fichier CSV
data = pd.read_csv('nomo32_metadata.csv')

# Supprimer la colonne "photo_names" si elle existe
if 'photo_names' in data.columns:
    data.drop(columns=['photo_names'], inplace=True)
if 'subject_id' in data.columns:
    data.drop(columns=['subject_id'], inplace=True)
# Supprimer la colonne "gender"
if 'gender' in data.columns:
    data.drop(columns=['gender'], inplace=True)

# Supprimer toutes les colonnes vides
data.dropna(axis=1, how='all', inplace=True)

# Remplacer toutes les valeurs NaN par les valeurs moyennes de la colonne respective
data.fillna(data.mean(), inplace=True)

# Supprimer la colonne "gender"
if 'gender' in data.columns:
    data.drop(columns=['gender'], inplace=True)


# Séparer les fonctionnalités des étiquettes
X = data

# Mise à l'échelle des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Appliquer PCA
pca = PCA(n_components=2)  # Réduire la dimension à 2 pour la visualisation
X_pca = pca.fit_transform(X_scaled)

# Visualiser les données dans l'espace des composantes principales
plt.scatter(X_pca[:, 0], X_pca[:, 1], cmap='viridis')
plt.xlabel('Composante Principale 1')
plt.ylabel('Composante Principale 2')
plt.title('Visualisation des données avec PCA')
plt.colorbar()
plt.show()
