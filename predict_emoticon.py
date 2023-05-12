import pickle


def load_resources():
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, vectorizer, scaler


def predict_emoticon(text, model, vectorizer, scaler):
    text_processed = [text]  # preprocess the text if necessary
    text_vectorized = vectorizer.transform(text_processed)
    text_scaled = scaler.transform(text_vectorized)
    predicted_emoji = model.predict(text_scaled)
    return predicted_emoji


model, vectorizer, scaler = load_resources()
while True:
    input_text = input("Your input text or word here: ")
    predicted_emoticon = predict_emoticon(input_text, model, vectorizer, scaler)
    print(predicted_emoticon)