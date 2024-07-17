import streamlit as st
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f'{BASE_DIR}/model.pkl', 'rb') as model_file, open(f'{BASE_DIR}/tfidfVectorizer.pkl', 'rb') as vectorizer_file:
    model = pickle.load(model_file)
    vectorizer = pickle.load(vectorizer_file)

st.title('News Classification')


sms_input = st.text_area('Enters the news below to classify it as fake or real.')

if st.button('Check'):
    if sms_input:
        transformed_input = vectorizer.transform([sms_input])

        prediction = model.predict(transformed_input)[0]
        prediction_proba = model.predict_proba(transformed_input)[0]

        if prediction == 1:
            st.write(f'This news is classified as **fake** with a confidence of {prediction_proba[1]*100:.2f}%.')
        else:
            st.write(f'This news is classified as **real** with a confidence of {prediction_proba[0]*100:.2f}%.')
    else:
        st.write('Please enter news to classify.')
