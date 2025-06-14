import pandas as pd;
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#Parte 1 - Leitura e tratamento de dados 

#Lendo arquivo
df = pd.read_csv("1045.txt", sep='\t', decimal=',',skiprows=8, names=["Travessa Horizontal", "Carga", "Tempo"])

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

"""
    Nesta etapa, realizamos a leitura dos dados experimentais provenientes do ensaio de tração, que inclui deslocamento da travessa horizontal e carga aplicada.

    A transformação dos dados em unidades consistentes (como converter carga de kN para N) e o cálculo da tensão (σ = força / área) e da deformação (ε = deslocamento / comprimento inicial) são fundamentais para a análise mecânica.

    Esses cálculos permitem construir a curva tensão x deformação, que é a base para avaliação do comportamento mecânico do material.

    Referências:
    - CALLISTER, William D.; RETHWISCH, David G. Ciência e Engenharia de Materiais: uma introdução. 10. ed. Porto Alegre: Bookman, 2018.
    - ASTM INTERNATIONAL. ASTM E8/E8M‑16a: Standard Test Methods for Tension Testing of Metallic Materials. West Conshohocken, PA: ASTM International, 2016.

    Esses procedimentos seguem normas e práticas padrão de ensaio para garantir a confiabilidade dos dados experimentais.
"""


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

    Referências:
    - CALLISTER, William D.; RETHWISCH, David G. Ciência e Engenharia de Materiais: uma introdução. 10. ed. Porto Alegre: Bookman, 2018.
    - ASTM INTERNATIONAL. ASTM E8/E8M‑16a: Standard Test Methods for Tension Testing of Metallic Materials. West Conshohocken, PA: ASTM International, 2016.

    A precisão da determinação do ponto de escoamento é fundamental para dimensionamento estrutural e avaliação da ductilidade do material.
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

"""
    O módulo de elasticidade (E) foi determinado a partir da região elástica da curva tensão x deformação, que corresponde ao trecho inicial onde o material apresenta comportamento linear e reversível.

    Selecionamos os pontos até o limite elástico (ε ≤ 0.016) para garantir que o ajuste fosse feito apenas nessa parte linear da curva.

    Aplicamos uma regressão linear (polinomial de grau 1) aos dados da região elástica, obtendo uma reta que representa a Lei de Hooke. O coeficiente angular dessa reta corresponde ao módulo de elasticidade, ou seja, à derivada da tensão em relação à deformação (dσ/dε) nessa região.

    Esse módulo é uma propriedade mecânica fundamental que indica a rigidez do material e a sua resistência à deformação elástica.

    O gráfico mostra a curva tensão x deformação completa em azul, e a reta ajustada na região elástica em vermelho tracejado, evidenciando a linearidade do comportamento elástico.

    Referências:
    - CALLISTER, William D.; RETHWISCH, David G. Ciência e Engenharia de Materiais: uma introdução. 10. ed. Porto Alegre: Bookman, 2018.
    - ASTM INTERNATIONAL. ASTM E8/E8M‑16a: Standard Test Methods for Tension Testing of Metallic Materials. West Conshohocken, PA: ASTM International, 2016.

    A compreensão do módulo de elasticidade auxilia no projeto e análise estrutural para garantir que deformações permaneçam dentro do limite elástico.
"""

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

    Referências:
    - CALLISTER, William D.; RETHWISCH, David G. Ciência e Engenharia de Materiais: uma introdução. 10. ed. Porto Alegre: Bookman, 2018.
    - ASTM INTERNATIONAL. ASTM E8/E8M‑16a: Standard Test Methods for Tension Testing of Metallic Materials. West Conshohocken, PA: ASTM International, 2016.

    O estudo da região plástica é importante para entender a ductilidade e resistência à fratura do material.
"""

#Parte 5 - Cálculo de Resiliência e Tenacidade

#Definindo o limite elástico (mesmo usado antes)
limite_elastico = 0.016

#Selecionando a região elástica
regiao_elastica = df[df["Deformação"] <= limite_elastico]
x_elast = regiao_elastica["Deformação"]
y_elast = regiao_elastica["Tensao"]

#Calculando a resiliência (área sob a curva na região elástica)
resiliencia = simpson(y_elast, x_elast) 

#Selecionando toda a curva até a fratura
x_total = df["Deformação"]
y_total = df["Tensao"]

#Calculando a tenacidade (área total sob a curva)
tenacidade = simpson(y_total, x_total)

#Plotando a curva com áreas sombreadas para resiliência e tenacidade
plt.figure(figsize=(10, 6))
plt.plot(df["Deformação"], df["Tensao"], label="Curva Tensão x Deformação", color='blue')

#Área da resiliência (região elástica)
plt.fill_between(df[df["Deformação"] <= 0.016]["Deformação"], 
                 df[df["Deformação"] <= 0.016]["Tensao"], 
                 color='green', alpha=0.4, label="Resiliência")

#Área da tenacidade (toda a curva)
plt.fill_between(df["Deformação"], df["Tensao"], color='orange', alpha=0.2, label="Tenacidade")

plt.xlabel("Deformação")
plt.ylabel("Tensão (MPa)")
plt.title("Resiliência e Tenacidade (Áreas sob a Curva)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Exibindo os resultados
print(f"Resiliência (energia elástica): {resiliencia:.2f} MJ/m³")
print(f"Tenacidade (energia total absorvida): {tenacidade:.2f} MJ/m³")

"""
    A resiliência foi calculada como a área sob a curva tensão x deformação na região elástica, ou seja, desde o início da deformação até o limite elástico (ε ≤ 0.016). Essa área representa a energia armazenada elasticamente pelo material durante o carregamento, que pode ser liberada ao remover a carga.

    Para obter essa área, utilizamos a integração numérica pelo método de Simpson (scipy.integrate.simpson), aplicado aos dados experimentais na região elástica.

    A tenacidade corresponde à energia total absorvida pelo material até a fratura, representada pela área total sob a curva tensão x deformação, desde a deformação zero até a máxima deformação antes da ruptura. Essa área foi também obtida pela integração pelo método de Simpson, considerando toda a curva.

    Essas integrações fornecem uma aproximação prática da energia envolvida nos processos de deformação do material, sendo úteis para análise comparativa do comportamento mecânico e desempenho dos materiais.

    Os gráficos mostram as regiões correspondentes à resiliência (área verde) e à tenacidade (área laranja), destacando visualmente a contribuição da região plástica para a absorção total de energia.

    Referências:
    - CALLISTER, William D.; RETHWISCH, David G. Ciência e Engenharia de Materiais: uma introdução. 10. ed. Porto Alegre: Bookman, 2018.
    - ASTM INTERNATIONAL. ASTM E8/E8M‑16a: Standard Test Methods for Tension Testing of Metallic Materials. West Conshohocken, PA: ASTM International, 2016.

    Esses parâmetros são essenciais para a seleção de materiais em aplicações sujeitas a cargas dinâmicas ou choques.
"""

#Plotando Gráfico completo, com todas as informações adquiridas (Escoamento, Região Elástica e Plástica, Resiliência e Tenacidade)
plt.figure(figsize=(10, 6))

#Curva tensão x deformação original
plt.plot(df["Deformação"], df["Tensao"], label="Curva Tensão x Deformação", color='blue')

#Linha offset 0.2%
plt.plot(df["Deformação"], linha_offset, '--', label="Linha Offset 0.2%", color='red')

#Ponto de escoamento
plt.plot(intersecao_eps, intersecao_sig, 'ro', label="Ponto de Escoamento")

#Regressão linear região elástica
plt.plot(x, reta, label="Regressão Linear (Região Elástica)", color='black', linestyle='--')

#Ajuste polinomial na região plástica
plt.plot(x_plot, y_plot, label="Ajuste Polinomial Grau 3 (Região Plástica)", color='orange', linestyle='--')

#Áreas preenchidas (resiliência e tenacidade)
plt.fill_between(df[df["Deformação"] <= limite_elastico]["Deformação"],
                 df[df["Deformação"] <= limite_elastico]["Tensao"],
                 color='green', alpha=0.15, label="Resiliência")

plt.fill_between(df["Deformação"], df["Tensao"], color='orange', alpha=0.07, label="Tenacidade")

#Limites do gráfico
plt.xlim(0, 0.14)
plt.ylim(0, 950)

#Labels e título
plt.xlabel("Deformação")
plt.ylabel("Tensão (MPa)")
plt.title("Análise Comp.: Curva Tensão x Deformação com Escoamento, Região Elástica e Plástica, Resiliência e Tenacidade")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


"""
    Referências Bibliográficas
    Livro-texto (ABNT):
    CALLISTER, William D.; RETHWISCH, David G. Ciência e Engenharia de Materiais: uma introdução. 10. ed. Porto Alegre: Bookman, 2018.

    Norma de ensaio de tração (ASTM):
    ASTM INTERNATIONAL. ASTM E8/E8M‑16a: Standard Test Methods for Tension Testing of Metallic Materials. West Conshohocken, PA: ASTM International, 2016.
"""
