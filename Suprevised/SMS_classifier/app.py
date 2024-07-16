import streamlit as st
import pickle

# Load the trained model and CountVectorizer
with open('model.pkl', 'rb') as model_file, open('tfidfVectorizer.pkl', 'rb') as vectorizer_file:
    model = pickle.load(model_file)
    vectorizer = pickle.load(vectorizer_file)

# Streamlit app
st.title('SMS Spam Classifier')

st.write('Enter the SMS message below to classify it as spam or not spam.')

# Text input for the SMS message
sms_input = st.text_area('SMS Message', '')

# Button to classify the SMS
if st.button('Classify'):
    if sms_input:
        # Transform the input text using the loaded CountVectorizer
        transformed_input = vectorizer.transform([sms_input])

        # Predict the class and probability
        prediction = model.predict(transformed_input)[0]
        prediction_proba = model.predict_proba(transformed_input)[0]

        # Display the result
        if prediction == 1:
            st.write(f'This message is classified as **spam** with a confidence of {prediction_proba[1]*100:.2f}%.')
        else:
            st.write(f'This message is classified as **not spam** with a confidence of {prediction_proba[0]*100:.2f}%.')
    else:
        st.write('Please enter an SMS message to classify.')
