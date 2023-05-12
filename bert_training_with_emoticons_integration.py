import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from tqdm.auto import tqdm
from collections import Counter
from transformers import BertModel, BertTokenizer, BertForSequenceClassification

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim

from joblib import dump


# Read the labeled dataset
data = pd.read_csv('data/svtvnews_labeled_data.csv')

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Extract the text and labels from the train and test sets
train_texts = train_data['message'].tolist()
train_labels = train_data['sentiment'].tolist()
test_texts = test_data['message'].tolist()
test_labels = test_data['sentiment'].tolist()


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# Load the pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Prepare the dataset
max_length = 128
train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_length)
test_dataset = NewsDataset(test_texts, test_labels, tokenizer, max_length)

# Create DataLoaders
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Set up the device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained mBERT model
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
model.to(device)

# handle the imbalance
class_counts = data['sentiment'].value_counts().sort_index().values
class_weights = 1.0 / class_counts
class_weights_normalized = class_weights / np.sum(class_weights)
weights_tensor = torch.tensor(class_weights_normalized, dtype=torch.float).to(device)

# Set up the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_dataloader) * 3  # Number of epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_function = CrossEntropyLoss(weight=weights_tensor)

# Fine-tune the model
for epoch in range(3):
    model.train()
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss = loss_function(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()

            predictions.extend(logits.argmax(axis=-1).tolist())
            true_labels.extend(labels.tolist())

    return predictions, true_labels


predictions, true_labels = evaluate_model(model, test_dataloader, device)
accuracy = accuracy_score(true_labels, predictions)
report = classification_report(true_labels, predictions, zero_division=1)

print(f"Accuracy: {accuracy}")
print(f"Classification report: \n{report}")

unlabeled_data = pd.read_csv('data/preprocessed_data.csv')
texts = unlabeled_data['message'].tolist()

# Prepare the dataset
max_length = 128
unlabeled_dataset = NewsDataset(texts, [0] * len(texts), tokenizer, max_length)

# Create a DataLoader
batch_size = 16
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False)


# Function for sentiment prediction
def predict_sentiment(model, dataloader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()

            predictions.extend(logits.argmax(axis=-1).tolist())
    return predictions


# Predict sentiment for the entire dataset
sentiment_predictions = predict_sentiment(model, unlabeled_dataloader, device)

# Add sentiment predictions to the data and save as a new CSV file
unlabeled_data['sentiment'] = sentiment_predictions
unlabeled_data.to_csv('news_with_sentiment.csv', index=False)
Counter(sentiment_predictions)

# ----- emoticons_integration -----
df = pd.read_csv('data/news_with_sentiment.csv')
emoticon_columns = df.loc[:, '🤡':'👎']
emoticon_sums = emoticon_columns.sum(axis=1)
normalized_emoticons = emoticon_columns.div(emoticon_sums, axis=0)
normalized_emoticons.fillna(0, inplace=True)
normalized_emoticons.columns = 'normalized_' + normalized_emoticons.columns
df = pd.concat([df, normalized_emoticons], axis=1)

df.to_csv('emoticons_normalized_file.csv', index=False)

# Convert the text messages to numerical representations using mBERT embeddings
tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = BertModel.from_pretrained("DeepPavlov/rubert-base-cased")


def get_embedding(text, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()


# Apply the get_embedding function to the 'message' column to get the embeddings
text_embeddings = np.vstack(df['message'].apply(get_embedding).values)

# Split the dataset into features (X) and target (y)
X_emoticons = df.loc[:, 'normalized_🤡':'normalized_👎'].values
y = df['sentiment'].values

# Combine text embeddings and normalized emoticon counts
X_combined = np.concatenate((text_embeddings, X_emoticons), axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# Save the trained classifier
dump(classifier, 'trained_random_forest.joblib')