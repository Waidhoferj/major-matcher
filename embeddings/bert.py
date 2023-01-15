import torch
from transformers import BertModel, BertTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
class BertSentenceEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, device="cpu",padding_length=50):
        """
        Args:
            `device`: pytorch device for inference. Either 'cpu' or a specific type of GPU.
            `padding_length`: The max sentence token length. Shorter sentences are padded to this length.
        """
        self._device = device
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self._model = model.to(device)
        self._model.eval()
        self._padding_length = padding_length

    def transform(self, X:list) -> np.ndarray:
        """
        Transforms sentences into embeddings

        Args:
            `X`: a dataset of sentences of shape (n_sentences,)
        Returns:
            Embeddings of the provided sentences of shape (n_sentences, embedding_dims)
        """
        tokens = self._tokenizer(
            X, 
            return_token_type_ids=False, 
            return_attention_mask=False,
            padding=True,
            truncation=True,
            max_length=self._padding_length,
            return_tensors="pt"
            )
        tokens = tokens["input_ids"].to(self._device)
        with torch.no_grad():
           hidden_states = self._model(
                            input_ids=tokens, 
                            output_hidden_states=True
                            )["hidden_states"]
        embeddings = torch.cat(hidden_states[-4:], dim=-1)
        embeddings = torch.mean(embeddings, dim=1)
        return embeddings.cpu().numpy()



if __name__ == "__main__":
    df = pd.read_csv("course_sentences.csv")
    embedder = BertSentenceEmbedder("mps", padding_length=1000)
    embeddings = embedder.transform(list(df["sentence"]))
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