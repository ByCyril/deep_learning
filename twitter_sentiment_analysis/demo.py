# Cyril Garcia
# Intro to Data Science
# Twitter Sentiment Analysis


import tweepy
from textblob import TextBlob

consumer_key = "ETgBvPn6zvQ7Rxu4GFV2wNAyx"
consumer_secret = "NR3gU0GKwHBcCPkMt9rRglxJWInG4c4d5TwICoBzRJQX4dHqfh"

access_token = "924159274602790912-5T3WYiKB7wiBLPX9aPeI8ch7ntM1ECo"
access_token_secret = "6efspts1hKcaY92AM0m3VZPli3LIvbMGmFYHurJF2URgX"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

user_input = raw_input("What do you want to search for? ")

public_tweets = api.search(user_input)

for tweet in public_tweets:
	print(tweet.text)
	analysis = TextBlob(tweet.text)
	print(analysis.sentiment)
	print("")

