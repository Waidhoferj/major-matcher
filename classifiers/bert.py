import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
from pathlib import Path
import json
from numpy.typing import NDArray

class BertClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, seed=42, epochs=5, device="cpu"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.seed = seed
        self.epochs = epochs
        self.model = None
        self.labels = None
        self.device=device

    def _get_classes(self, y: List[str]) -> Tuple[NDArray, List[str]]:
        labels = sorted(set(y))
        ids = [i for i in range(len(labels))]
        return ids, labels

    def _compute_metrics(self,eval_pairs):
            logits, labels = eval_pairs
            n = 3
            ordered_choices = (-logits).argsort(-1)[:,:n]
            metrics = {}
            metrics["top_n_accuracy"] = np.mean([label in choices for label, choices in zip(labels, ordered_choices)])
            metrics["accuracy"] = np.mean(labels == ordered_choices[:,0])
            return metrics



    def load_weights(self, path:str):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path).to(self.device)
        self.labels = list(self.model.config.label2id.keys())
        
    def _tokenize(self, texts:List[str]) -> torch.Tensor:
        return self.tokenizer(texts, padding=True,
            truncation=True,
            max_length=100,
            return_tensors="pt").to(self.device)


    
    def fit(self, X:List[str], y:List[str]):
        ids, labels = self._get_classes(y)
        self.labels = labels
        id2label = dict(zip(ids,labels))
        label2id = dict(zip(labels,ids))
        X = self._tokenize(X)
        dataset = [{"input_ids": text, "label": label2id[label]} for text, label in zip(X["input_ids"],y)]
        train_ds, test_ds = train_test_split(dataset, shuffle=True, random_state=self.seed, train_size=0.85)
        batch_size = 64

        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=len(labels), id2label=id2label, label2id=label2id
        ).to(self.device)
        weights_path="weights/bert_classifier"
        training_args = TrainingArguments(
            output_dir=weights_path,
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            use_mps_device=self.device=="mps"
        )

        class_weights = torch.Tensor()

        trainer = WeightedTrainer(
            class_ids=ids,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics
        )

        trainer.train()
        model.eval()
        self.model = model

        
    
    def predict_proba(self, X:List[str]) -> NDArray:
        if self.model is None:
            raise Exception("Fit the model before inference.")
        tokens = self._tokenize(X)
        with torch.no_grad():
            logits = self.model(**tokens).logits
            return F.softmax(logits, -1).cpu().numpy()
        

    def predict(self, X:List[str])-> List[str]:
        preds = self.predict_proba(X)
        return [self.labels[i] for i in preds.argmax(-1)]




class WeightedTrainer(Trainer):

    def __init__(self,class_ids, train_dataset, *args, **kwargs):
        super().__init__(train_dataset=train_dataset, *args,**kwargs)
        y_train = [y["label"] for y in train_dataset]
        class_weights = compute_class_weight("balanced", classes=class_ids, y=y_train).astype("float32")
        class_weights = torch.from_numpy(class_weights).to(self.args.device.type)
        self.criteria = nn.CrossEntropyLoss(weight=class_weights)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.criteria(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss



