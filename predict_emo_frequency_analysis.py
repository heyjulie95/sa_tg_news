import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
import pickle

SEED = 1

df = pd.read_csv('output_test.csv')
print(df.most_freq_reaction.value_counts())
reactions = df.groupby('most_freq_reaction')
reactions = {k for k, v in reactions.groups.items() if len(v) > 1000}
print(reactions)

df = df.loc[df.most_freq_reaction.isin(reactions)]
df.reset_index(inplace=True)
print(df)

texts_train, texts_test, y_train, y_test = train_test_split(
    df.message,
    df.most_freq_reaction,
    test_size=0.1,
    random_state=SEED,
    stratify=df.most_freq_reaction,
)

texts_train[:5], y_train[:5]

cvect = CountVectorizer(
    max_features=30_000,
    ngram_range=(1, 1),
    analyzer='word'
)

X_train = cvect.fit_transform(texts_train)
X_test = cvect.transform(texts_test)
logreg = LogisticRegression(random_state=SEED, solver='saga', max_iter=5000)
scaler = StandardScaler(with_mean=False)  # with_mean=False, since you're using sparse data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)
print(classification_report(y_test, y_pred, zero_division=0))

clf = DummyClassifier(strategy='uniform')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# Save the trained model, vectorizer, and scaler
pickle.dump(logreg, open('model.pkl', 'wb'))
pickle.dump(cvect, open('vectorizer.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))