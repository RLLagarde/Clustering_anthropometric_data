import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Charger les données depuis le fichier CSV
data = pd.read_csv('bodym.csv')

# Supprimer les colonnes indésirables
columns_to_exclude = ['subject_id', 'gender', 'photo_names']
data = data.drop(columns=columns_to_exclude, errors='ignore')  # Ignorer les colonnes qui pourraient ne pas être présentes

# Remplacer toutes les valeurs NaN par les valeurs moyennes de la colonne respective
data.fillna(data.mean(), inplace=True)

# Séparer les fonctionnalités des étiquettes
X = data.drop(columns=['gender'])  # Assurez-vous de ne pas inclure la variable cible dans les fonctionnalités

# Appliquer PCA
pca = PCA()
X_pca = pca.fit_transform(X)

#charge factorielle
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])], index=X.columns)

# Visualiser les charges factorielles (vous pouvez ajuster le nombre de composantes selon vos besoins)
plt.figure(figsize=(12, 6))
loadings.abs().sort_values(by='PC1', ascending=False).head(10).plot(kind='bar')
plt.title('Top 10 des charges factorielles pour la première composante principale (PC1)')
plt.xlabel('Variable')
plt.ylabel('Charge factorielle (valeur absolue)')
plt.show()
