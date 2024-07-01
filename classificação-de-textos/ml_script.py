import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Dados de treino
dados = pd.read_csv('arquivo.csv')

textos_treino = dados['texto'].tolist()
categorias_treino = dados['categoria'].tolist()

modelo = make_pipeline(CountVectorizer(), MultinomialNB())
modelo.fit(textos_treino, categorias_treino)

# Textos para testes
textos_teste = [
    'Eu adoro programação em Python',
    'Big data e análise de dados são fascinantes',
    'Minha profissão é desenvolvedor de software',
    'O aprendizado de máquina é muito interessante',
    'A inteligência artificial domina minha empresa'
]

categorias_previstas = modelo.predict(textos_teste)

for texto, categoria in zip(textos_teste, categorias_previstas):
    print(f'Texto: {texto}\nCategoria prevista: {categoria}\n')
