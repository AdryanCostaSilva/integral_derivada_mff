import pandas as pd;
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#Parte 1 - Leitura e tratamento de dados 

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

#Parte 2 - Geração de gráfico tensão x deformação

#Encontrando ponto de escoamento com MÉTODO OFFSET 0.2%
#Selecionando trecho elástico
#Identificando o melhor valor de n que maximiza o coeficiente R² da regressão linear

melhor_r2 = 0
melhor_n = 0
for n in range(10, 100):
    x = df["Deformação"][:n].values.reshape(-1, 1)
    y = df["Tensao"][:n].values
    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)
    if r2 > melhor_r2:
        melhor_r2 = r2
        melhor_n = n
epsilon_elastico = df["Deformação"][:melhor_n]
tensao_elastico = df["Tensao"][:melhor_n]

#Fazendo a regressão linear para encontrar o Módulo de Elasticidade E
coef = np.polyfit(epsilon_elastico, tensao_elastico, 1)
E = coef[0]

#Criando linha offset (E * (ε - 0.002))
offset = 0.002
linha_offset = E * (df["Deformação"] - offset)

#Encontrando a interseção da linha offset com a curva real
dif = df["Tensao"] - linha_offset
idx = np.where(np.diff(np.sign(dif)))[0][0]  # onde cruza

#Interpolando para maior precisão
eps1, eps2 = df["Deformação"].iloc[idx], df["Deformação"].iloc[idx+1]
sig1, sig2 = df["Tensao"].iloc[idx], df["Tensao"].iloc[idx+1]
dif1, dif2 = dif.iloc[idx], dif.iloc[idx+1]

intersecao_eps = np.interp(0, [dif1, dif2], [eps1, eps2])
intersecao_sig = np.interp(intersecao_eps, df["Deformação"], df["Tensao"])

#plotando gáfico (Tensão) x (Deformação) com ponto de escoamento

plt.figure(figsize=(10,6))
plt.plot(df["Deformação"], df["Tensao"], label="Curva Tensão x Deformação", color='blue')
plt.plot(df["Deformação"], linha_offset, '--', label="Linha Offset 0.2%", color='red')
plt.plot(intersecao_eps, intersecao_sig, 'ro', label="Ponto de Escoamento")

plt.xlabel("Deformação")
plt.ylabel("Tensão (MPa)")
plt.title("Gráfico Tensão x Deformação com Escoamento")
plt.xlim(0, 0.140)
plt.ylim(0, 950)   
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Exibindo o ponto de escoamento
print(f"Ponto aproximado de escoamento: Deformação = {intersecao_eps:.6f}, Tensão = {intersecao_sig:.2f} MPa") 

"""O ponto de escoamento, determinado pelo método do offset de 0,2%, foi encontrado em deformação de 0.017022 e tensão de 559.60 MPa. Esse ponto marca o início da deformação plástica no material, ou seja, a partir desse momento a deformação passa a ser permanente e o material não retorna mais à sua forma original mesmo após a retirada da carga."""


"""Justificativa
    O ponto de escoamento corresponde ao início da deformação plástica, momento em que a curva tensão x deformação deixa de ser linear e começa a se curvar, indicando que o material não retorna mais à sua forma original.

    Visualmente, essa transição pode ser percebida no gráfico pela mudança na inclinação da curva, que passa do regime elástico linear para o regime plástico não-linear.

    Matematicamente, essa mudança de comportamento pode ser detectada analisando a derivada da curva tensão em relação à deformação (dσ/dε):

    No regime elástico, a derivada é aproximadamente constante, igual ao módulo de elasticidade (E).

    No ponto de escoamento, essa derivada começa a diminuir, indicando perda da rigidez do material.

    Para maior precisão, pode-se usar o método do escoamento por offset de 0,2%, onde uma linha paralela ao trecho linear inicial é traçada com um deslocamento na deformação, e o ponto de escoamento é a interseção dessa linha com a curva.

    Assim, combinando a observação visual da mudança de curvatura da curva com a análise da derivada e/ou aplicação do método do offset, podemos identificar o ponto de escoamento com maior precisão.
"""

#Parte 3 - Região Elástica e Módulo de Elasticidade
#Selecionando manualmente os pontos da região elástica antes do escoamento (ex: até deformação 0.014)
limite_elastico = 0.016
regiao_elastica = df[df["Deformação"] <= limite_elastico]

x = regiao_elastica["Deformação"]
y = regiao_elastica["Tensao"]

#O módulo de elasticidade E é o coeficiente angular da reta na região elástica, que corresponde à derivada da tensão em relação à deformação  (dσ/dε) nesta região linear da curva.
#Achando Regressão linear (reta da Lei de Hooke)
coef = np.polyfit(x, y, 1)
E = coef[0]  # derivada (pendente) da curva tensão x deformação

#Criando reta da regressão
reta = np.polyval(coef, x)

#Plotando a curva com a reta da região elástica
plt.figure(figsize=(10, 6))
plt.plot(df["Deformação"], df["Tensao"], label="Curva Tensão x Deformação", color='blue')
plt.plot(x, reta, label="Regressão Linear (Região Elástica)", color='red', linestyle='--')
plt.xlabel("Deformação")
plt.ylabel("Tensão (MPa)")
plt.title("Região Elástica - Módulo de Elasticidade")
plt.grid(True)
plt.legend()
plt.show()

print(f"Módulo de Elasticidade (E) = {E:.2f} MPa") #Módulo de Elasticidade (E) = 33557.17 MPa

#Parte 4 - Identificação da região de deformação plástica

#Considerando como região plástica os pontos após o limite elástico (ex: deformação > 0.016)
limite_elastico = 0.016
regiao_plastica = df[df["Deformação"] > limite_elastico]

x_plast = regiao_plastica["Deformação"]
y_plast = regiao_plastica["Tensao"]

#Ajustando a função polinomial de grau 3 à região plástica
coef_poli = np.polyfit(x_plast, y_plast, 3)
polinomio = np.poly1d(coef_poli)

#Gerando valores ajustados
x_plot = np.linspace(x_plast.min(), x_plast.max(), 500)
y_plot = polinomio(x_plot)

#Plotando curva original e o ajuste na região plástica
plt.figure(figsize=(10, 6))
plt.plot(df["Deformação"], df["Tensao"], label="Curva Tensão x Deformação", color='blue')
plt.plot(x_plot, y_plot, label="Ajuste Polinomial Grau 3 (Região Plástica)", color='orange', linestyle='--')
plt.axvline(limite_elastico, color='gray', linestyle=':', label='Limite Elástico')
plt.xlabel("Deformação")
plt.ylabel("Tensão (MPa)")
plt.title("Região Plástica com Ajuste Polinomial")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""
A região de deformação plástica corresponde ao trecho da curva onde o material se deforma permanentemente, ou seja, não retorna à sua forma original após a remoção da carga.

Para representar matematicamente esse comportamento, foi ajustada uma função polinomial de grau 3 aos dados experimentais após o limite elástico (ε > 0.016). Essa aproximação permite capturar a curvatura e a não linearidade típicas da resposta plástica dos materiais.

Embora o modelo polinomial não represente perfeitamente todos os fenômenos físicos envolvidos, ele é útil para análises matemáticas e computacionais da curva tensão x deformação na região plástica.
"""



"""
    Referências Bibliográficas
        Livro-texto (ABNT):
        CALLISTER, William D.; RETHWISCH, David G. Ciência e Engenharia de Materiais: uma introdução. 10. ed. Porto Alegre: Bookman, 2018.

        Norma de ensaio de tração (ASTM):
        ASTM INTERNATIONAL. ASTM E8/E8M‑16a: Standard Test Methods for Tension Testing of Metallic Materials. West Conshohocken, PA: ASTM International, 2016.

        Norma internacional equivalente (ISO):
        ISO. ISO 6892‑1:2019 – Metallic materials — Tensile testing — Part 1: Method of test at room temperature. Genebra: International Organization for Standardization, 2019.
"""
