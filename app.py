import gradio as gr
from classifiers.bert import BertClassifier
import os
import numpy as np
from functools import cache
from preprocessing.helper import get_recommendations

CONFIG_FILE = os.path.join("weights", "bert_classifier_deployment_weights")
N_SUGGESTIONS = 3


@cache
def get_model(config_path: str) -> BertClassifier:
    bert_classifier = BertClassifier(device="mps")
    bert_classifier.load_weights(config_path)
    return bert_classifier


def predict(interests: str) -> list[str]:
    bert_classifier = get_model(CONFIG_FILE)
    probs = bert_classifier.predict_proba(interests)
    labels = np.array(bert_classifier.labels)
    results_mask = (-probs).argsort(-1)[:,:N_SUGGESTIONS]
    suggested_majors = labels[results_mask][0].tolist()
    confidences = probs[0][results_mask[0]]
    confidences /= confidences.sum()
    confidences = confidences.tolist()
    return dict(zip(suggested_majors, confidences))


def demo():
    title = "Major Matcher"
    description = "Describe your interests and the model will suggest a compatible college major."
    example_interests = [
        "I really enjoy spending time with animals.",
        "I like playing music and dancing.",
        "A good book makes me happy."
    ]
    
    app = gr.Interface(
        title=title,
        description=description,
        inputs=gr.TextArea(
            label="Describe your interests",
            placeholder="I really enjoy..."
        ),
        fn=predict,
        outputs=gr.Label(label="Suggested Majors"),
        examples=example_interests
    )
    return app


if __name__ == "__main__":
    demo().launch()
