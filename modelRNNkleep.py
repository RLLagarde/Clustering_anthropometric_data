import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv('bodym.csv')
noms = df.columns.tolist()
# Supprimer les lignes contenant des valeurs NaN
df.dropna(axis=0, inplace=True)
df.drop(columns=['subject_id', 'photo_names'], inplace=True)
# Remplacer 'female' par 0 et 'male' par 1 dans la colonne 'gender'
df['gender'] = df['gender'].map({'female': 0, 'male': 1})
df.drop_duplicates(inplace=True)

X = df.drop(columns=[ 'shoulder-to-crotch', 'shoulder-breadth', 'weight_kg', 'calf', 'forearm', 'thigh', ])   # Toutes les colonnes sauf la 15ème
y = df['weight_kg']   # La 15ème colonne

# Créer un transformateur de colonnes pour mettre à l'échelle les caractéristiques entre 0 et 1
ct = make_column_transformer(
    (MinMaxScaler(), X.columns)  # Mettre à l'échelle toutes les colonnes de X
)
# Appliquer les transformations
X = ct.fit_transform(X)



y = (y - y.min()) / (y.max() - y.min())


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
X_train = tf.expand_dims(X_train, axis=-1)


# Replicate model_1 and add an extra layer
model_2 =  tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu', input_shape=(11,)),  # Couche avec 64 neurones et input_shape=(16,)
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),  # Couche avec 32 neurones
    tf.keras.layers.Dense(1)  # add a second layer
])

# Compile the model
model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

# Fit the model
model_2.fit(X_train, y_train, epochs=100, verbose=0) # set verbose to 0 for less output
# Make and plot predictions for model_2
y_preds_2 = model_2.predict(X_test)
results = pd.DataFrame({'y_true': y_test, 'y_pred': y_preds_2.flatten()})
from sklearn.metrics import mean_squared_error

# Calculer l'erreur MSE
mse = mean_squared_error(y_test, y_preds_2)

# Afficher l'erreur MSE
print("Mean Squared Error (MSE) :", mse)


from sklearn.metrics import r2_score

# Calculer le coefficient de détermination (R^2)
r2 = r2_score(y_test, y_preds_2)

# Afficher le coefficient de détermination
print("Coefficient de détermination (R^2) :", r2)

# Calculer le nombre de fois où y_preds_2 est différent de y_test
counter = np.sum(y_preds_2.flatten() != y_test.to_numpy())




plt.figure(figsize=(10, 6))
plt.scatter(results['y_true'], results['y_pred'], alpha=0.5, s=1, label='Prédictions')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Ligne Idéale')
plt.xlabel('Valeurs Réelles (y_test)')
plt.ylabel('Valeurs Prédites (y_preds_2)')
plt.title('Valeurs Réelles vs Valeurs Prédites')
plt.legend()
plt.grid(True)
plt.show()
