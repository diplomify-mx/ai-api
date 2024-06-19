import requests

handle = "realDonaldTrump"
url = f"https://b54c-148-241-227-110.ngrok-free.app/api/tweets/{handle}/"
r = requests.get(url)
tweets = []
for i in r.json():
    tweets.append(i['tweet'])
print(tweets)