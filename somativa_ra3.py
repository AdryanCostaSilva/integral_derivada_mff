import pandas as pd;

with open("C:\\Users\\adrya\Desktop\\3° semestre\\Modelagem de Fenômenos Fisicos\\somativa_ra3\\integral_derivada_mff\\1045.txt", "r", encoding="utf-8") as f:
    linhas = f.readlines()  

df = pd.read_csv("C:\\Users\\adrya\\Desktop\\3° semestre\\Modelagem de Fenômenos Fisicos\\somativa_ra3\\integral_derivada_mff\\1045.txt", sep="\t", skiprows=6)

print(df)