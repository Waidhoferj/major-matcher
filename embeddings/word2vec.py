
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from glob import iglob
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from functools import cache
    



class Word2VecEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100):
        self.model = None
        self.stop_words = get_stopwords()
        self.vector_size = vector_size

    def _preprocess(self, text:str) -> List[str]:
       words = word_tokenize(text)
       only_keywords = [word for word in words 
                        if word not in self.stop_words
                        and word.isalpha()]
       return only_keywords

    def fit(self, sentences:List[str]):
        sentences = [self._preprocess(t) for t in sentences]
        self.model = Word2Vec(sentences, vector_size=self.vector_size, window=5, min_count=1, workers=4)


    def transform(self, X:List[str]) -> List[List[np.ndarray]]:
        if self.model is None:
            raise Exception("fit model before transforming")
        sents = map(self._preprocess,X)
        def get_embedding(word):
            try:
                return self.model.wv[word] 
            except:
                return np.zeros((self.vector_size,))
        
        return [[get_embedding(word) for word in sent] for sent in sents]

    def latent_distance(self,text1:str, text2:str) -> float:
        first_tokens, second_tokens = self.transform([text1])[0], self.transform([text2])[0]
        sum_dist = 0.0
        for t1 in first_tokens:
            for t2 in second_tokens:
                sum_dist += np.sum((t2-t1)**2)**0.5
        return sum_dist / float(len(first_tokens) * len(second_tokens))
        
@cache
def get_stopwords():
    words = set()
    for stop_file in iglob("stopwords/*.txt"):
        with open(stop_file, "r") as f:
            words.update(l.lower() for l in f.readlines())
    return set(stopwords.words('english')) | words



def test_latent_dist():
    df = pd.read_csv("course_sentences.csv")
    embedder = Word2VecEmbedder()
    sentences = list(df["sentence"])
    embedder.fit(sentences)

    show_dist = lambda s1, s2: print(s1 + "\n", s2 + "\n", embedder.latent_distance(s1,s2))
    show_dist(*(["This is the same sentence."] * 2))
    show_dist("artificial intelligence is my passion", "I really enjoy computer science")
    show_dist("artificial intelligence is my passion", "I really enjoy archeology")

    



def test_pipeline():
    df = pd.read_csv("course_sentences.csv")
    embedder = Word2VecEmbedder()
    sentences = list(df["sentence"])
    embedder.fit(sentences)
    embeddings = embedder.transform(sentences)
    labels = df["program"]
    classifier = KNeighborsClassifier(n_neighbors=10)
    classifier.fit(embeddings, labels)
    num_suggestions = 10

    prompt = "Covers methods currently available to address complexity, including systems thinking, model based systems engineering and life cycle governance."
    embedding = embedder.transform([prompt])
    probs = classifier.predict_proba(embedding)[0]
    idx = np.argsort(-probs)[:num_suggestions]
    label_map = np.array(sorted(set(labels)))
    print(prompt, label_map[idx], probs[idx])


if __name__ == "__main__":
    test_latent_dist()