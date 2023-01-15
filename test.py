from classifiers.mlp import MajorMlpClassifier
from embeddings.bert import BertSentenceEmbedder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from classifiers.bert import BertClassifier
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from helper import load_data, get_recommendations, plot_confusion_matrix
import matplotlib.pyplot as plt
import os

device = "mps"


def evaluate(load_weights=False):
    """
    Performs basic train/test split evaluation.
    """
    os.makedirs("figures", exist_ok=True)
    sentences, labels = load_data(num_majors=40)
    embedder = BertSentenceEmbedder(device, padding_length=1000)

    seed = 2
    x_train, x_test, y_train, y_test = train_test_split(
        sentences, labels, random_state=seed, shuffle=True, train_size=0.8
    )
    train_embeddings = embedder.transform(x_train)
    test_embeddings = embedder.transform(x_test)
    knn = KNeighborsClassifier()
    mlp = MajorMlpClassifier(device)
    bert_classifier = BertClassifier(
        device=device,
        epochs=25,
    )

    if load_weights:
        mlp.load_weights("weights/major_classifier")
        bert_classifier.load_weights("weights/bert_classifier_deployment_weights")
    else:
        bert_classifier.fit(x_train, y_train)
        mlp.fit(train_embeddings, y_train)
    knn.fit(train_embeddings, y_train)
    class_labels = np.array(bert_classifier.labels)

    def report(name, classifier, x, y, n=3):
        probs = classifier.predict_proba(x)
        ordered_choices = class_labels[(-probs).argsort(-1)[:, :n]]
        preds = ordered_choices[:, 0]
        print(name)
        print(
            f"Top {n} accuracy",
            np.mean([label in choices for label, choices in zip(y, ordered_choices)]),
        )
        print(classification_report(y, preds))
        plot_confusion_matrix(y, preds, class_labels)
        plt.savefig(f"figures/{name}_cm.png")
        plt.clf()

    report("bert_classifier", bert_classifier, x_test, y_test)
    report("KNN", knn, test_embeddings, y_test)
    report("major_mlp", mlp, test_embeddings, y_test)


def demo():
    """
    Interact with a model on the command line.
    """
    bert_classifier = BertClassifier(device="mps")
    weights_path = os.path.join("weights", "bert_classifier_deployment_weights")
    bert_classifier.load_weights(weights_path)
    while True:
        command = input("Describe your ideal major: ")
        if command.lower() == "q" or command.lower() == "quit":
            break
        probs = bert_classifier.predict_proba(command)
        labels = bert_classifier.labels
        print(get_recommendations(probs, labels, n=3)[0])


if __name__ == "__main__":
    evaluate()
