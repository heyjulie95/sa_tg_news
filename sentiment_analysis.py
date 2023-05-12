from joblib import load
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

df = pd.read_csv("data/old_lentach_preprocessed_data.csv")
tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = BertModel.from_pretrained("DeepPavlov/rubert-base-cased")
classifier = load('trained_random_forest.joblib')
reference_emoticon_list = ['🤡', '😁', '🤯', '❤', '✍', '🤮', '👏', '\U0001fae1', '🍓', '🤔', '🥰', '🔥', '🤬', '☃', '😢',
                           '🤣', '🖕', '❤\u200d🔥', '🍾', '🌚', '🍌', '💋', '🤩', '🤨', '😐', '🥴', '🐳', '⚡', '🥱', '💯',
                           '💔', '🌭', '🎉', '🏆', '😱', '😈', '🕊', '👌', '😍', '🙏', '💩', '👍', '👎']


# Function to preprocess a single row
def preprocess_row(row):
    text = row[0]
    reactions_str = row[1].replace("'", '"')
    reactions = json.loads(reactions_str)
    reaction_counts = {next(iter(reaction)): reaction[next(iter(reaction))] for reaction in reactions}
    return text, reaction_counts


# Preprocess the entire dataframe
df['preprocessed'] = df.apply(preprocess_row, axis=1)
df = df.dropna(subset=['preprocessed'])
# Select 20 random samples from the dataframe
samples = df.sample(n=200)


def get_embedding(text, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()


def predict_sentiment(text, emoticons):
    # Get the embedding for the text
    text_embedding = get_embedding(text)

    # Create an emoticon vector
    emoticon_vector = np.zeros(len(reference_emoticon_list))
    for emoticon, count in emoticons.items():
        if emoticon in reference_emoticon_list:
            index = reference_emoticon_list.index(emoticon)
            emoticon_vector[index] = count

    # Normalize the emoticons
    normalized_emoticons = emoticon_vector / (np.sum(emoticon_vector) + 1e-10)  # Adding a small constant to avoid division by zero
    normalized_emoticons = normalized_emoticons.reshape(1, -1)  # ensure it's a 2D array

    # Combine the text embeddings and normalized emoticons into a single feature vector
    combined_input = np.concatenate((text_embedding, normalized_emoticons), axis=1)

    # Predict the sentiment
    sentiment = classifier.predict(combined_input)
    return sentiment[0]


def reactions_to_feature_vector(reactions):
    # Initialize a zero vector of the length of reference_emoticon_list
    emoticon_vector = [0] * len(reference_emoticon_list)

    # For each emoticon in the sample's reactions, update the corresponding index in the vector
    for emoticon, count in reactions.items():
        if emoticon in reference_emoticon_list:
            emoticon_vector[reference_emoticon_list.index(emoticon)] = count

    # Normalize the vector
    emoticon_vector = np.array(emoticon_vector)
    normalized_emoticons = emoticon_vector / np.sum(emoticon_vector)

    return normalized_emoticons


results = []
# Make predictions on the 20 samples
for _, row in samples.iterrows():
    text, reactions = row['preprocessed']
    sentiment = predict_sentiment(text, reactions)
    # Create a DataFrame for this result and add it to the list
    result_df = pd.DataFrame({'Text': [text], 'Reactions': [reactions], 'Predicted Sentiment': [sentiment]})
    results.append(result_df)

# Concatenate all the result DataFrames
results_df = pd.concat(results, ignore_index=True)
results_df.to_csv('sentiment_results.csv', index=False)

# Create a bar plot
sentiment_counts = results_df['Predicted Sentiment'].value_counts()
sentiment_counts.plot(kind='bar')
plt.show()