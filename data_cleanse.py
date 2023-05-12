import pandas as pd
import re

# Load the dataset
file_path = "data/svtvnews_messages_cleaned.csv"
data = pd.read_csv(file_path)


# Function to clean the text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = text.replace("ё", "е").replace("Ё", "Е")  # Replace "ё" with "е" and "Ё" with "Е"
    text = re.sub(r'[^а-яА-Яa-zA-Z0-9\s-]', '', text)  # Keep Russian and Latin characters, numbers, and hyphens
    text = re.sub(r'\n|\r|\t', ' ', text)  # Remove newline, carriage return, and tab symbols
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'Задонатить через бота Patreon Boosty Предложить новость', '', text)
    text = text.strip()  # Remove leading and trailing spaces
    return text


# Clean the news text
data['message'] = data['message'].apply(clean_text)


def process_reactions(reactions):
    reactions = reactions[1:-1].split(', ')
    reaction_dict = {}
    for reaction in reactions:
        matches = re.findall(r"\{'(.*?)': (\d+)\}", reaction)
        if len(matches) > 0:
            emoji, count = matches[0]
            reaction_dict[emoji] = int(count)
    return reaction_dict


# Process the reactions and create separate columns for each emoji and its count
for index, row in data.iterrows():
    reactions = process_reactions(row['reactions'])
    for emoji, count in reactions.items():
        if emoji not in data.columns:
            data[emoji] = 0
        data.at[index, emoji] = count

# Save the preprocessed data
preprocessed_data_path = "data/svtvnews_preprocessed_data.csv"
data.to_csv(preprocessed_data_path, index=False)