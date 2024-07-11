import numpy as np
import pandas as pd

df = pd.read_csv("bodym.csv")
df.head()

len(df)

df["height_cm"].mean()


# +
#On est systémaitquement un centimètre en dessous de la taille moyenne en France

# 
#Ajout de l'IMC

df["IMC"] = df["weight_kg"] / (0.01*df["height_cm"] * 0.01*df["height_cm"])
df.head()
# -

df["IMC"].mean()

df[df["gender"] == "male"]["height_cm"].mean()

df[df["gender"] == "female"]["height_cm"].mean()

df[df["gender"] == "female"]["IMC"].mean()

df[df["gender"] == "male"]["IMC"].mean()

df["ankle"].mean()
