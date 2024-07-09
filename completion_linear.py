import csv
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def lire_fichier_csv(chemin_fichier):
    donnees = {}

    with open(chemin_fichier, newline='') as fichier_csv:
        lecteur_csv = csv.reader(fichier_csv)
        categories = next(lecteur_csv, None)

        for categorie in categories:
            donnees[categorie] = []

        for ligne in lecteur_csv:
            for i, valeur in enumerate(ligne):
                if valeur == '':
                    donnees[categories[i]].append(None)
                else:
                    try:
                        donnees[categories[i]].append(float(valeur))
                    except:
                        donnees[categories[i]].append(valeur)
    return donnees

data = lire_fichier_csv("nomo32_metadata.csv")
df = pd.DataFrame(data)

numeric_columns = df.select_dtypes(include=['number']).columns
df_numeric = df[numeric_columns].dropna(axis=1, how='all')

imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

cov_matrix = np.cov(df_imputed, rowvar=False)
mean_vector = df_imputed.mean()
print(mean_vector)

def deduce_missing_components(vector_with_missing, covariance_matrix, mean_vector):
    # Récupérer les indices des valeurs manquantes dans le vecteur
    missing_indices = np.isnan(vector_with_missing)

    # Extraire les parties connues et inconnues du vecteur
    known_values = vector_with_missing[~missing_indices]
    unknown_indices = np.where(missing_indices)[0]

    # Sélectionner les parties correspondantes de la matrice de covariance
    cov_known = covariance_matrix[~missing_indices][:, ~missing_indices]
    cov_mixed = covariance_matrix[~missing_indices][:, missing_indices]

    # Calculer les valeurs manquantes à partir des valeurs connues et de la covariance
    print(cov_mixed.shape,cov_known.shape)
    deduced_values = np.dot(np.dot(cov_mixed.T, np.linalg.inv(cov_known)), known_values - mean_vector[~missing_indices]) + mean_vector[unknown_indices]

    # Mettre à jour le vecteur avec les valeurs manquantes déduites
    vector_with_missing[missing_indices] = deduced_values

    return vector_with_missing


# Exemple d'utilisation avec le vecteur df_imputed.iloc[-1] (dernière ligne de df_imputed)
vector_row = df_imputed.iloc[-1].to_numpy()
print("Vecteur original :", vector_row)

# Mettre une composante manquante (par exemple, la deuxième composante)
vector_row[1] = np.nan
vector_row[0] = np.nan

# Déduction des composantes manquantes
deduced_vector = deduce_missing_components(vector_row, cov_matrix, mean_vector)
print("Vecteur avec les composantes manquantes déduites :", deduced_vector)
