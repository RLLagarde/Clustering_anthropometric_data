import pandas as pd
import numpy as np

# Charger le fichier CSV
df = pd.read_csv('bodym.csv')

# Filtrer les colonnes pour ne garder que celles qui contiennent des valeurs numériques
df = df.select_dtypes(include=[np.number])

# Remplacer les valeurs NaN par la moyenne de chaque colonne
df = df.apply(lambda x: x.fillna(x.mean()), axis=0)

# Initialiser les listes pour stocker les résultats
close_columns = []
high_correlation_columns = []

# Comparer les colonnes pour trouver celles qui ont 95% des valeurs avec un écart inférieur à 5%
for i in range(len(df.columns)):
    for j in range(i + 1, len(df.columns)):
        col1 = df.iloc[:, i]
        col2 = df.iloc[:, j]
        
        # Calculer l'écart relatif
        relative_diff = np.abs(col1 - col2) / col1
        
        # Vérifier si 95% des valeurs ont un écart inférieur à 5%
        if (relative_diff < 0.05).mean() >= 0.95:
            close_columns.append((df.columns[i], df.columns[j]))

# Calculer la corrélation de Pearson pour chaque paire de colonnes
correlation_matrix = df.corr()
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if correlation_matrix.iloc[i, j] > 0.9:
            high_correlation_columns.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

# Afficher les résultats
print("Paires de colonnes avec 95% des valeurs ayant un écart inférieur à 5% :")
for col_pair in close_columns:
    print(col_pair)

print("\nPaires de colonnes avec une corrélation de Pearson supérieure à 0.9 :")
for col_pair in high_correlation_columns:
    print(col_pair)
