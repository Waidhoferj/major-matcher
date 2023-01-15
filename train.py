from classifiers.bert import BertClassifier
from classifiers.mlp import MajorMlpClassifier
from embeddings.bert import BertSentenceEmbedder
import pickle
from helper import load_data


def train_bert_classifier(
    device="cpu",
    n_classes=40,
    include_majors=[],
    epochs=25
):
    sentences, labels = load_data(num_majors=n_classes, include_majors=include_majors)
    bert_classifier = BertClassifier(device=device, epochs=epochs)
    bert_classifier.fit(sentences, labels)


def train_major_classifier(
    device="cpu",
    n_classes=40,
    include_majors=[],
    epochs=200
):
    sentences, labels = load_data(num_majors=n_classes, include_majors=include_majors)
    embedder = BertSentenceEmbedder(device, padding_length=1000)
    embeddings = embedder.transform(sentences)
    mlp = MajorMlpClassifier(device, epochs=epochs)
    mlp.fit(embeddings,labels)
    mlp.save_weights("weights/major_classifier")


if __name__ == "__main__":
    train_major_classifier(device="mps", include_majors=["Computer Science", "Computer Engineering"])