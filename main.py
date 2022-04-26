import numpy as np
from transformers import DebertaV2Tokenizer, TFDebertaV2ForSequenceClassification
from fastapi import FastAPI

tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-xsmall")
model = TFDebertaV2ForSequenceClassification.from_pretrained("api/Model/")

app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the movie's reviews",
    version="0.1",
)

@app.get("/predict-review")
def predict_sentiment(review: str):
    """
    A simple function that receive a review content and predict the sentiment of the content.
    :param review:
    :return: prediction
    """
    
    tf_batch = tokenizer([review], max_length=128, padding=True, truncation=True, return_tensors='tf')
    tf_outputs = model(tf_batch)
    labels = ['Negative','Positive']
    label = np.argmax(tf_outputs[0], axis=1)[0]
    return labels[label]