import pandas as pd;

#Abrindo arquivo
with open("C:\\Users\\adrya\Desktop\\3° semestre\\Modelagem de Fenômenos Fisicos\\somativa_ra3\\integral_derivada_mff\\1045.txt", "r", encoding="utf-8") as f:
    linhas = f.readlines()  

#Lendo arquivo
df = pd.read_csv("C:\\Users\\adrya\\Desktop\\3° semestre\\Modelagem de Fenômenos Fisicos\\somativa_ra3\\integral_derivada_mff\\1045.txt", sep='\t', decimal=',',skiprows=8, names=["Travessa Horizontal", "Carga", "Tempo"])

#Substituindo "," por "."
df = df.replace(",", ".", regex=True)

#Transformando kN em N
try:
    df["Carga"] = df["Carga"].astype(float) * 1000
except:
    pass

#Tranformando os dados em float
for col in df.columns:
    try:
        df[col] = df[col].astype(float)
    except ValueError:
        pass

#Entradas pedidas no pdf
area_secao_transversal = 67.5 #mm²
comprimento_inicial = 46.3 #mm

#Calculando tensão e transformando em MPa
df['Tensao'] = df['Carga'] / area_secao_transversal #Mpa(N/mm²)

#Calculando deformação
df['Deformação'] = df['Travessa Horizontal'] / comprimento_inicial #mm

print(df)
