import streamlit as st
from langdetect import detect
from deep_translator import GoogleTranslator
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import datetime

# Download VADER if not already done
nltk.download('vader_lexicon')

# Set page configuration
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

st.title("💬 Sentiment Analysis of Tweets")
st.write("This app analyzes the sentiment of your tweet or feedback using AI.")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Text Input
text = st.text_area("✍️ Enter a tweet or feedback:", height=200)

if st.button("🔍 Analyze"):
    if text.strip() == "":
        st.error("Please enter some text to analyze.")
    else:
        # Detect language and translate
        try:
            lang = detect(text)
            if lang != 'en':
                translated = GoogleTranslator(source='auto', target='en').translate(text)
                st.write(f"🌐 Detected Language: `{lang}`")
                st.info(f"🔤 Translated Text: {translated}")
            else:
                translated = text
                st.write("🌐 Language: English (No translation needed)")
        except Exception as e:
            st.warning(f"Language detection/translation failed: {e}")
            translated = text

        # Sentiment Analysis with VADER
        scores = sia.polarity_scores(translated)
        compound = scores['compound']

        if compound >= 0.05:
            sentiment = "Positive 😀"
        elif compound <= -0.05:
            sentiment = "Negative 😞"
        else:
            sentiment = "Neutral 😐"

        st.subheader("🧠 Sentiment Result")
        st.success(f"Sentiment: **{sentiment}**")
        st.markdown(f"Compound Score: `{compound:.2f}`")

        # Feedback based on positivity
        if compound > 0.5:
            st.write("😍 You're spreading positivity!")
        elif compound < -0.5:
            st.write("😡 Seems pretty negative.")
        elif -0.5 <= compound <= 0.5 and compound != 0:
            st.write("🙂 Slightly opinionated.")
        else:
            st.write("😐 Very neutral or unclear emotion.")

        # Word Cloud
        st.subheader("☁️ Word Cloud")
        try:
            wc = WordCloud(width=600, height=300, background_color="white").generate(translated)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"WordCloud generation failed: {e}")

        # Save to history
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "original_text": text,
            "translated_text": translated,
            "sentiment": sentiment,
            "compound": compound,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

# Sidebar History
st.sidebar.title("📜 Sentiment History")
if "history" in st.session_state and st.session_state.history:
    for item in st.session_state.history[::-1]:
        st.sidebar.markdown(f"**{item['time']}**")
        st.sidebar.markdown(f"📝 Original: {item['original_text']}")
        st.sidebar.markdown(f"🔤 Translated: {item['translated_text']}")
        st.sidebar.markdown(f"💬 Sentiment: *{item['sentiment']}*")
        st.sidebar.markdown(f"📊 Compound Score: `{item['compound']:.2f}`")
        st.sidebar.markdown("---")
else:
    st.sidebar.info("No history yet. Analyze something!")
