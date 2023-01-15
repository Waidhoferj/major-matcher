import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json
import os

class MajorMlpClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, device="cpu", seed=42, epochs=200, patience:int=None):
        super().__init__()
        self.device = device
        self.seed = seed
        self.model = None
        self.epochs = epochs
        self.patience = patience if patience is not None else epochs
        self.class_labels = None


    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        return torch.from_numpy(X).to(self.device)

    def _preprocess_labels(self, y: List[str]) -> np.ndarray:
        unique_labels = np.array(self._get_classes(y))
        one_hot = np.array([
            unique_labels == label
            for label in y
        ], dtype="float32")
        
        return torch.from_numpy(one_hot).to(self.device)

    def _get_classes(self, y: List[str]) -> List[str]:
        return sorted(set(y))
    
    def fit(self, X:np.ndarray, y:List[str]):
        """
        Args:
            X: embeddings of shape (n_sentences, embedding_size)
            y: program labels that match with each sentence
        """
        self.class_labels = np.array(self._get_classes(y))
        class_weights = compute_class_weight("balanced", classes=self.class_labels, y=y).astype("float32")
        class_weights = torch.from_numpy(class_weights).to(self.device)
        X, y = self._preprocess_features(X), self._preprocess_labels(y)
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=self.seed, shuffle=True)
        should_stop = EarlyStopping(self.patience)
        val_loss = np.inf
        model = ProgramClassifierNetwork(x_train.shape[1], y_train.shape[1])
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        epoch = 0
        while not should_stop.step(val_loss) and epoch < self.epochs:
            preds = model(x_train)
            loss = criterion(preds, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                val_preds = model(x_val)
                val_loss = criterion(val_preds, y_val).item()
            epoch += 1
        model.eval()
        self.model = model

    def predict_proba(self, X:np.ndarray) -> np.ndarray:
        X = self._preprocess_features(X)
        if self.model is None:
            raise Exception("Train model with fit() before predicting.")
        with torch.no_grad():
            logits = self.model(X)
            return F.softmax(logits, dim=-1).cpu().numpy()
    
    def predict(self, X:np.ndarray) -> List[str]:
        """
        Args:
            X: embeddings of shape (n_sentences, embedding_size)
        Returns:
            predicted classes for each embedding
        """
        pred_i = self.predict_proba(X).argmax(-1)
        return self.class_labels[pred_i]

    def save_weights(self,path:str):
        os.makedirs(path, exist_ok=True)
        weights_path = os.path.join(path, "weights.pt")
        config_path = os.path.join(path,"config.json")
        torch.save(self.model.state_dict(), weights_path)
        state = {
            "device": self.device,
            "seed": self.seed,
            "epochs": self.epochs,
            "patience": self.patience,
            "class_labels": list(self.class_labels)
        }
        with open(config_path, "w") as f:
            json.dump(state, f)


    def load_weights(self, path:str):
        weights_path = os.path.join(path, "weights.pt")
        config_path = os.path.join(path,"config.json")
        state_dict = torch.load(weights_path)
        input_size = int(state_dict["input_size"].item())
        n_classes = int(state_dict["n_classes"].item())
        model = ProgramClassifierNetwork(input_size,n_classes).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model
        with open(config_path, "r") as f:
            config = json.load(f)
        config["class_labels"] = np.array(config["class_labels"]) if config["class_labels"] is not None else None
        self.__dict__.update(config)


        


class ProgramClassifierNetwork(nn.Module):
    def __init__(self, input_size:int, n_classes:int) -> None:
        super().__init__()
        self.input_size = nn.Parameter(torch.Tensor([input_size]), requires_grad=False)
        self.n_classes = nn.Parameter(torch.Tensor([n_classes]), requires_grad=False)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )


    def forward(self,x):
        return self.classifier(x)

class EarlyStopping:
    def __init__(self, patience=0):
        self.patience = patience
        self.last_measure = np.inf
        self.consecutive_increase = 0
    
    def step(self, val) -> bool:
        if self.last_measure <= val:
            self.consecutive_increase +=1
        else:
            self.consecutive_increase = 0
        self.last_measure = val

        return self.patience < self.consecutive_increase
