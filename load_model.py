from transformers import pipeline

# Load sentiment analysis pipestreline
sentiment_pipeline = pipeline("sentiment-analysis")
tweets = [
    "I love the new design of your website!",
    "This product is the worst thing I've ever bought.",
    "I'm not sure how I feel about this.",
    "Amazing work! Keep it up.",
    "This is terrible. I'm so disappointed."
]
results = sentiment_pipeline(tweets)

for tweet, result in zip(tweets, results):
    print(f"Tweet: {tweet}")
    print(f"Sentiment: {result['label']} with score {round(result['score'], 2)}\n")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Convert results to DataFrame
df = pd.DataFrame(results)
df['tweet'] = tweets

# Plot sentiment counts
sns.countplot(x='label', data=df)
plt.title("Sentiment Distribution")
plt.show()

