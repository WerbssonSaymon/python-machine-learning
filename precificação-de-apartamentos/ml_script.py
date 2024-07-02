from sklearn import linear_model
import numpy as np
import plotly.graph_objects as go

#scikit learn: Linear Models - Ordinary Least Squares

"""
    Projeto de predição de valor de apartamentos

    Proporção R$ 550 por m² e 1 quarto
    Proporção R$ 1100 por m² e 2 quarto
    Proporção R$ 1650 por m² e 3 quarto
"""

# Dados de treino: x => [área (m²), número de quartos] e y => preços dos apartamentos em reais
x_1_treino = np.array([
    [1, 1], [10, 1], [15, 1], [20, 1], [25, 1], [30, 1], [40, 1], [50, 1]
])
y_1_treino = np.array([
    550, 5500, 8250, 11000, 13750, 16500, 22000, 27500  
])

x_2_treino = np.array([
    [1, 2], [10, 2], [15, 2], [20, 2], [25, 2], [30, 2], [40, 2], [50, 2]
])
y_2_treino = np.array([
    1100, 11000, 16500, 22000, 27500, 33000, 44000, 55000,
])

x_3_treino = np.array([
    [1, 3], [10, 3], [15, 3], [20, 3], [25, 3], [30, 3], [40, 3], [50, 3]
])
y_3_treino = np.array([
    1650, 16500, 24750, 33000, 41250, 49500, 66000, 82500 
])

modelo_1 = linear_model.LinearRegression()
modelo_2 = linear_model.LinearRegression()
modelo_3 = linear_model.LinearRegression()

modelo_1.fit(x_1_treino, y_1_treino)
modelo_2.fit(x_2_treino, y_2_treino)
modelo_3.fit(x_3_treino, y_3_treino)

def info_modelo_1():
    print("Coeficientes modelo 1:", modelo_1.coef_)
    print("Intercepto modelo 1:", modelo_1.intercept_)

def info_modelo_2():
    print("Coeficientes modelo 2:", modelo_2.coef_)
    print("Intercepto modelo 2:", modelo_2.intercept_)

def info_modelo_3():
    print("Coeficientes modelo 3:", modelo_3.coef_)
    print("Intercepto modelo 3:", modelo_3.intercept_)

# Testes para os modelo
teste_modelo_1 = np.array([
    [60, 1], [22, 1], [48, 1]  
])
teste_modelo_2 = np.array([
    [60, 2], [22, 2], [48, 2] 
])
teste_modelo_3 = np.array([
    [60, 3], [22, 3], [48, 3]   
])

previsoes_1 = modelo_1.predict(teste_modelo_1)
previsoes_2 = modelo_2.predict(teste_modelo_2)
previsoes_3 = modelo_3.predict(teste_modelo_3)

info_modelo_1()
for i, area in enumerate(teste_modelo_1):
    print(f"Previsão de preço para apartamento com {area[0]}m² e 1 quarto: R$ {previsoes_1[i]:,.2f}")

info_modelo_2()
for i, area in enumerate(teste_modelo_2):
    print(f"Previsão de preço para apartamento com {area[0]}m² e 2 quartos: R$ {previsoes_2[i]:,.2f}")

info_modelo_3()
for i, area in enumerate(teste_modelo_3):
    print(f"Previsão de preço para apartamento com {area[0]}m² e 3 quartos: R$ {previsoes_3[i]:,.2f}")


def plot_model(x_treino, y_treino, modelo, titulo):
    
    # Plotar os dados de treino
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_treino[:, 0], y=y_treino, mode='markers', name='Dados de Treino'))

    # Gerar uma linha de previsão
    x_values = np.linspace(x_treino[:, 0].min(), x_treino[:, 0].max(), 100).reshape(-1, 1)
    x_values_com_quartos = np.hstack([x_values, np.full((x_values.shape[0], 1), x_treino[0, 1])])
    y_values = modelo.predict(x_values_com_quartos)
    
    fig.add_trace(go.Scatter(x=x_values.flatten(), y=y_values, mode='lines', name='Modelo de Regressão'))
    
    fig.update_layout(title=titulo, xaxis_title='Área (m²)', yaxis_title='Preço (R$)')
    fig.show()

plot_model(x_1_treino, y_1_treino, modelo_1, 'Modelo para Apartamentos com 1 Quarto')
plot_model(x_2_treino, y_2_treino, modelo_2, 'Modelo para Apartamentos com 2 Quartos')
plot_model(x_3_treino, y_3_treino, modelo_3, 'Modelo para Apartamentos com 3 Quartos')