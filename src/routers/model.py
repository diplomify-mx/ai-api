from fastapi import APIRouter
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import warnings
import nltk
import geograpy
from typing import List
import requests

warnings.filterwarnings("ignore")
nltk.downloader.download('maxent_ne_chunker')
nltk.downloader.download('words')
nltk.downloader.download('treebank')
nltk.downloader.download('maxent_treebank_pos_tagger')
nltk.downloader.download('punkt')

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)

MODEL = f"cardiffnlp/twitter-roberta-base-emotion"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.to("cuda:0")
router = APIRouter(prefix="", tags = ["Model"])

@router.post("/predict")
async def predictions(handle: str):
    url = f"https://b54c-148-241-227-110.ngrok-free.app/api/tweets/{handle}/"
    r = requests.get(url).json()
    tweets = []
    countries_emotions = []
    for i in r:
        tweets.append(i['tweet'])
    for text in tweets:
        countries = geograpy.get_place_context(text = text)
        text = preprocess(text)
        encode = tokenizer(text, return_tensors = "pt").to("cuda:0")
        results = model(**encode)
        scores = results[0][0].detach().cpu().numpy()
        scores = softmax(scores)
        ranked = np.argsort(scores)
        ranked = ranked[::-1]
        main_country = countries.countries
        if main_country != []:
            if labels[ranked[0]] != "anger":
                labels[ranked[0]] = "pos"
            else:
                labels[ranked[0]] = "neg"
            countries_emotions.append({"country": main_country[0], "emotion": labels[ranked[0]]})
    return countries_emotions
    #return str(labels[ranked[0]])