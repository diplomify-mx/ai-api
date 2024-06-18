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
router = APIRouter(prefix="", tags = ["Model"])

@router.post("/predict")
async def predictions(tweets: List[str]):
    countries_emotions = []
    for text in tweets:
        countries = geograpy.get_place_context(text = text)
        text = preprocess(text)
        encode = tokenizer(text, return_tensors = "pt")
        results = model(**encode)
        scores = results[0][0].detach().numpy()
        scores = softmax(scores)
        ranked = np.argsort(scores)
        ranked = ranked[::-1]
        main_country = countries.countries
        if main_country != []:
            countries_emotions.append({"country": main_country[0], "emotion": labels[ranked[0]]})
    return countries_emotions
    #return str(labels[ranked[0]])