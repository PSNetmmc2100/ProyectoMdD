import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# cargar los datos desde un archivo csv
df = pd.read_csv('Genshin charac rev (by banner).csv')

# tokenizar el texto utilizando NLTK
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(nltk.PorterStemmer().stem(item))
    return stems

# vectorizar el texto utilizando TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
X = vectorizer.fit_transform(df['Personaje'])

# ajustar el modelo de Latent Dirichlet Allocation (LDA)
model = LatentDirichletAllocation(n_components=10, random_state=42)
model.fit(X)

# imprimir los 10 Personajes principales y sus palabras clave
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Personajes %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

print_topics(model, vectorizer)
