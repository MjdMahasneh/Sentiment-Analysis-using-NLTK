import nltk
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

result = sentiment_analysis("I hate this product!")
# print(result)

values = list(result.values())
names = list(result.keys())
# print(names, values,sep= '\n')

print(names[np.argmax(values)])