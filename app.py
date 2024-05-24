from flask import Flask, request, render_template
import joblib
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from googletrans import Translator
from deep_translator import GoogleTranslator
import random


app = Flask(__name__)




filmes = ["Avatar","Vingadores: Ultimato","Titanic","Star Wars","Vingadores: Guerra Infinita","Jurassic World","O Rei Leão","Frozen","Velozes & Furiosos 7"]

# Carregar o modelo e o vetor
modelo_carregado = joblib.load('modelo_nb.pkl')
vectorizer_carregado = joblib.load('vectorizer.pkl')

# Função para prever sentimento de um novo texto
def prever_sentimento(novo_texto):
    # Vetorizar o novo texto
    novo_texto_tfidf = vectorizer_carregado.transform([novo_texto])
    # Fazer a previsão
    predicao = modelo_carregado.predict(novo_texto_tfidf)
    # Converter a previsão para rótulo de categoria
    return 'positivo' if predicao[0] == 1 else 'negativo'

def translate_to_english(text):
    translator = Translator()
    translated_text = translator.translate(text, src='pt', dest='en').text
    return translated_text

def traduzir(text):
    tradutor = GoogleTranslator(source="pt", target="en")
    traducao = tradutor.translate(text)
    return traducao

@app.route('/')
def home():
    filme_aleatorio = random.choice(filmes)
    return render_template('index.html',  filme=filme_aleatorio)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['review']
        input_text = traduzir(input_text)
        filme_aleatorio = request.form['filme']
        resultado = prever_sentimento(input_text)
        return render_template('index.html', prediction=resultado, review=input_text, filme=filme_aleatorio)

if __name__ == '__main__':
    app.run(debug=True)




#------------------------------------------------------------------------------------------------------------------

