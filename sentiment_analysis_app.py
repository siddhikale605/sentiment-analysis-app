import streamlit as st
from textblob import TextBlob
from deep_translator import GoogleTranslator
from langdetect import detect
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import datetime

# Set page configuration
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

st.title("💬 Sentiment Analysis of Tweets")
st.write("This app analyzes the sentiment of your tweet or feedback using AI.")

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
                st.info(f"Translated Text: {translated}")
            else:
                translated = text
                st.write("🌐 Language: English (No translation needed)")
        except Exception as e:
            st.warning(f"Language detection/translation failed: {e}")
            translated = text

        # Sentiment Analysis
        blob = TextBlob(translated)
        polarity = blob.sentiment.polarity

        # Result display
        if polarity > 0:
            sentiment = "Positive 😀"
        elif polarity < 0:
            sentiment = "Negative 😞"
        else:
            sentiment = "Neutral 😐"

        st.subheader("🧠 Sentiment Result")
        st.success(f"Sentiment: **{sentiment}**")
        st.markdown(f"Polarity Score: `{polarity:.2f}`")

        # Emoji Feedback
        if polarity > 0.5:
            st.write("😍 You're spreading positivity!")
        elif polarity < -0.5:
            st.write("😡 Seems pretty negative.")
        elif -0.5 <= polarity <= 0.5 and polarity != 0:
            st.write("🙂 Slightly opinionated.")
        else:
            st.write("😐 Very neutral or unclear emotion.")

        # AI Summary
        st.subheader("🧠 AI Summary")
        summary = blob.noun_phrases
        if summary:
            st.markdown(f"🔍 Key Topics: `{', '.join(summary[:5])}`")
        else:
            st.write("No significant summary detected.")

        # Word Cloud
        st.subheader("☁️ Word Cloud")
        wc = WordCloud(width=600, height=300, background_color="white").generate(translated)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        # Save to history
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "text": text,
            "sentiment": sentiment,
            "polarity": polarity,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

# History
st.sidebar.title("📜 Sentiment History")
if "history" in st.session_state and st.session_state.history:
    for item in st.session_state.history[::-1]:
        st.sidebar.markdown(f"**{item['time']}**  \nSentiment: *{item['sentiment']}*  \nPolarity: `{item['polarity']:.2f}`")
else:
    st.sidebar.info("No history yet.")
