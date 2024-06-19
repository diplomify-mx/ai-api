from transformers import pipeline
from fastapi import APIRouter
from typing import List
import geograpy
import requests


router = APIRouter(prefix="", tags = ["Classification"])
sentiment_classifier = pipeline('text-classification', model='AdamCodd/tinybert-sentiment-amazon', device = 0)

@router.post("/classify")
async def classification(handle: str):
    countries_emotions = []
    url = f"https://b54c-148-241-227-110.ngrok-free.app/api/tweets/{handle}/"
    r = requests.get(url).json()
    tweets = []
    countries_emotions = []
    for i in r:
        tweets.append(i['tweet'])
    for text in tweets:
        countries = geograpy.get_place_context(text = text)
        ranked = sentiment_classifier(text)[0]['label']
        main_country = countries.countries
        if main_country != []:
            if ranked == "positive":
                ranked = "pos"
            else:
                ranked = "neg"
            countries_emotions.append([main_country[0], ranked])
    return countries_emotions
    #return str(labels[ranked[0]])