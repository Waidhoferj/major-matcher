---
title: Major Matcher
emoji: 🎓
colorFrom: green
colorTo: yellow
sdk: gradio
python_version: 3.10.8
sdk_version: 3.15.0
app_file: app.py
pinned: false
---

# Major Matcher

A tool for matching student interests to areas of study.

## Getting Started

1. Set up python environment:

```
conda env create --file environment.yml
conda activate major-matcher
```

## Project Layout

- `embeddings`: Sklearn-style transformers that encode natural language into latent embedding vectors.
- `classifiers`: Model architectures for classifying college majors.
- `test.py`: Evaluation and demo code for all models.
- `train.py`: Training loops for models.
