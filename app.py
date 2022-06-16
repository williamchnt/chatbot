
import pandas as pd
# Importing modules
from nltk.corpus import stopwords
import nltk
nltk.download('all')
import numpy as np
import random
import string

import bs4 as bs
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from streamlit_chat import message as st_message

def setIconPage():
    st.set_page_config(
        page_title = "NLP Chat bot",
        layout = 'wide'
    )

def HideStreamlitContent():
    
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)



@st.cache
def read_data():
    df = pd.read_csv('data_next_cleaned.csv', header = 0)
    return df 

def perform_lemmatization(tokens):
    wnlemmatizer = nltk.stem.WordNetLemmatizer()
    return [wnlemmatizer.lemmatize(token) for token in tokens]

def get_processed_text(document):
    return perform_lemmatization(nltk.word_tokenize(document.lower().translate(punctuation_removal)))

def generate_response(user_input):
    chatbot_answer = ''
    article_sentences.append(user_input)
    word_vectorizer = TfidfVectorizer(tokenizer=get_processed_text, stop_words='english')
    all_word_vectors = word_vectorizer.fit_transform(article_sentences)
    similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)
    similar_sentence_number = similar_vector_values.argsort()[0][-2]

    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-1]

    if vector_matched == 0:
        chatbot_answer = chatbot_answer + "I am sorry but I can't understand, but I am still learning !"
        f = open("QuestionDontKnow.txt", "a")
        f.write(user_input)
        f.close()
        unknown_answer.append(user_input)
        return chatbot_answer
    else:
        chatbot_answer = chatbot_answer + article_sentences[similar_sentence_number]
        return chatbot_answer

def generate_greeting_response(greeting):
    for token in greeting.split():
        if token.lower() in greeting_inputs:
            return random.choice(greeting_responses)

# MAIN

setIconPage()
HideStreamlitContent()


df = read_data()
corpus =''.join(str(line).strip() for line in df['responce'])
article_sentences = nltk.sent_tokenize(corpus)
article_words = nltk.word_tokenize(corpus)
punctuation_removal = dict((ord(punctuation), None) for punctuation in string.punctuation)

unknown_answer = []
stopwords = stopwords.words('english')

word_vectorizer = TfidfVectorizer(tokenizer=get_processed_text, stop_words=stopwords)
all_word_vectors = word_vectorizer.fit_transform(article_sentences)


greeting_inputs = ("hey", "good morning", "good evening", "morning", "evening", "hi", "whatsup",'hello')
greeting_responses = ["hey", "hey how are you?", "*nods*", "hello, how you doing", "hello", "Welcome, I am good and you"]


similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)

similar_sentence_number = similar_vector_values.argsort()[0][-2]

byeAnswer = ['bye','goodbye']


if "history" not in st.session_state:
    st.session_state.history = []

st.title("NLP Project Chatbot")

def generate_answer():
    human_text = st.session_state.input_text
    human_text_lowered = human_text.lower()
    if human_text_lowered not in byeAnswer:
        st.session_state.history.append({"message": human_text, "is_user": True,"key":random.randint(1,100000)})
        if human_text_lowered == 'thanks' or human_text_lowered == 'thank you very much' or human_text_lowered == 'thank you':
            st.session_state.history.append({"message": 'Your welcome !', "is_user": False,"key":random.randint(1,100000)})
        else:
            if generate_greeting_response(human_text_lowered) != None:
                st.session_state.history.append({"message": generate_greeting_response(human_text_lowered), "is_user": False,"key":random.randint(1,100000)})
            else:
                st.session_state.history.append({"message": generate_response(human_text_lowered), "is_user": False,"key":random.randint(1,100000)})
                # article_sentences.remove(human_text_lowered)
    else:
        st.session_state.history.append({"message": human_text, "is_user": True,"key":random.randint(1,100000)})
        st.session_state.history.append({"message": "Good bye and take care of yourself... I'm staying here if you need more informations", "is_user": False,"key":random.randint(1,100000)})


st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)

# HelloText = "Hello, I am the American Airlines Chatbot. You can ask me any question regarding our company :"
# st.session_state.history.append({"message": HelloText, "is_user": False,"key":random.randint(1,100000)})

for chat in st.session_state.history:
    st_message(**chat)  # unpacking
