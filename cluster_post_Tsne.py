import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Charger les données depuis le fichier CSV
data = pd.read_csv('bodym.csv')

# Supprimer la colonne "photo_names" si elle existe
if 'photo_names' in data.columns:
    data.drop(columns=['photo_names'], inplace=True)

# Supprimer la colonne "subject_id"
if 'subject_id' in data.columns:
    data.drop(columns=['subject_id'], inplace=True)


# Supprimer la colonne "gender"
if 'gender' in data.columns:
    data.drop(columns=['gender'], inplace=True)

# Supprimer toutes les colonnes vides
data.dropna(axis=1, how='all', inplace=True)

# Remplacer toutes les valeurs NaN par les valeurs moyennes de la colonne respective
data.fillna(data.mean(), inplace=True)

# Séparer les fonctionnalités des étiquettes
X = data

# Appliquer t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Appliquer K-means sur les données t-SNE
kmeans = KMeans(n_clusters=4, random_state=42)  # Choisir le nombre de clusters en fonction de votre contexte
kmeans.fit(X_tsne)
labels = kmeans.labels_

# Visualiser les clusters dans l'espace des composantes t-SNE
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
plt.xlabel('Composante t-SNE 1')
plt.ylabel('Composante t-SNE 2')
plt.title('Clusters dans l\'espace des composantes t-SNE')
plt.colorbar()
plt.show()
