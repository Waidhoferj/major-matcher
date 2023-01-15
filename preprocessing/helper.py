import re
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.metrics import confusion_matrix
import seaborn as sns

PROGRAM = "Program"


def clean_text(text):
    text_input = re.sub('[^a-zA-Z1-9]+', ' ', str(text))
    output = re.sub(r'\d+', '', text_input)
    return output.lower().strip()


def get_num_courses_per_program():
    df = pd.read_csv('program_courses.csv')
    return df.groupby([PROGRAM])[PROGRAM].count()


def load_data(num_majors=20, include_majors=[]) -> Tuple[List[str], np.ndarray]:
    """
    Loads and preprocesses `course_sentences` data.
    """
    courses = pd.read_csv("course_sentences.csv").drop(["course"], axis=1).dropna()
    descriptions = pd.read_csv("program_descriptions.csv").rename(columns={"description": "sentence"}).dropna()
    df = pd.concat([courses, descriptions], axis=0, ignore_index=True)
    majors = list(df.groupby("program").count().sort_values(by=["sentence"], ascending=False).index)
    majors = include_majors + majors
    majors = majors[:num_majors]
    df = df[df["program"].isin(majors)]
    sentences = list(df["sentence"])
    labels = np.array(df["program"])

    return sentences, labels

def plot_confusion_matrix(y_true:List[str], y_pred:List[str], classes:List[str]):
    """Plots a confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df=pd.DataFrame(data=cm, index=classes, columns=classes)
    sns.heatmap(cm_df, annot=True)
    


def get_recommendations(probs:np.ndarray, labels:List[str], n=5) -> List[List[str]]:
    """
    Args:
        `probs`: predictions array of shape (n_inputs,n_classes)
        `labels`: class labels of shape (n_classes,)
        `n`: number of recommendations
    Returns:
        Top labels based on a probability distribution
    """
    np_labels = np.array(labels)
    return np_labels[(-probs).argsort(-1)[:,:n]]

