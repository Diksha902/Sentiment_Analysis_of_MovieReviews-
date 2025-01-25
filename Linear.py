import pandas as pd
import numpy as np
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from collections import Counter
import joblib

# Download NLTK resources
nltk.download('stopwords')
from nltk.corpus import stopwords

# Step 1: Load the Movie Review Dataset
df = pd.read_csv("IMDB Dataset.csv")
print(f"Initial Dataset Shape: {df.shape}")

# Step 2: Lets do some Data Inspection
print('The Shape of the data is as below:')
print(df.shape)
print("\nSample Data:")
print(df.head())

# Sentiment Distribution in Dataset
sns.countplot(x='sentiment', data=df)
plt.title("Sentiment distribution")


# Step 3: Check for the Missing Values in the dataset
print("\nChecking for Missing Values:")
print(df.isnull().sum())

# Step 4: Map Sentiments to Binary Values 
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
print("\nSentiment Mapping Complete.")

# Step 5: Preprocessing the Text Data
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove HTML tags
    text = re.sub('<br />', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"https\S+|www\S+|http\S+", '', text, flags = re.MULTILINE)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    #Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stop words
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

print("\nPreprocessing Text Data...")
df['cleaned_review'] = df['review'].apply(preprocess_text)
print("Text Preprocessing Complete.")

# Step 6: Check for Duplicates and Remove Them
initial_length = len(df)
df = df.drop_duplicates(subset='cleaned_review')
print(f"\nRemoved {initial_length - len(df)} duplicate reviews.")

# Step 7 Generating Word Cloud
print("\nGenerating Word Cloud...")
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(df['cleaned_review']))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Most Frequent Words", fontsize=16)
plt.show()

# Positive Review Word Cloud
pos_reviews =  df[df.sentiment == 1]
text = ' '.join([word for word in pos_reviews['cleaned_review']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in positive reviews', fontsize = 19)
plt.show()

#Negative Review Word Cloud
neg_reviews =  df[df.sentiment == 0]
text = ' '.join([word for word in neg_reviews['cleaned_review']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in negative reviews', fontsize = 19)
plt.show()

# Step 8 Bar Plot for Top Words
print("\nGenerating Bar Plot of Top Words...")
all_tokens = " ".join(df['cleaned_review']).split()
word_freq = Counter(all_tokens)
freq_df = pd.DataFrame(word_freq.most_common(20), columns=["Word", "Frequency"])

plt.figure(figsize=(10, 6))
sns.barplot(
    x="Frequency",
    y="Word",
    data=freq_df,
    hue="Word",           # Assign 'Word' to 'hue' to map colors
    palette="viridis",
    dodge=False,
    legend=False          # Hide the legend since it's redundant
)
plt.title("Top 20 Most Frequent Words", fontsize=16)
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.show()


# Step 9: Apply Stemmer Function
stemmer = PorterStemmer() 
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data
df.review = df['review'].apply(lambda x: stemming(x))


# Step 10: Feature Extraction using Bag of Words (BoW)
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']


# Step 11: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData Split into Training and Testing Sets.")

# Step 12: Define Function to Evaluate Models
def evaluate_model(name, model, X_test, y_test, training_time):
    start = time.time() 
    y_pred = model.predict(X_test)
    evaluation_time = time.time() - start
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\nResults for {name}:")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Evaluation Time: {evaluation_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    return {
        "Model": name,
        "Training Time (s)": training_time,
        "Evaluation Time (s)": evaluation_time,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }

# Step 13: Train and Evaluate Models
results = []

# Logistic Regression
start = time.time()
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
training_time = time.time() - start
results.append(evaluate_model("Logistic Regression", log_reg, X_test, y_test, training_time))

# Naive Bayes
start = time.time()
nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)
training_time = time.time() - start
results.append(evaluate_model("Naive Bayes", nb_clf, X_test, y_test, training_time))

# Linear Support Vector Machine
start = time.time()
linear_svc = LinearSVC(random_state=42, max_iter=1000)
linear_svc.fit(X_train, y_train)
training_time = time.time() - start
results.append(evaluate_model("Linear SVC", linear_svc, X_test, y_test, training_time))

# Random Forest
start = time.time()
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
training_time = time.time() - start
results.append(evaluate_model("Random Forest", rf_clf, X_test, y_test, training_time))

# Step 14: Results DataFrame
results_df = pd.DataFrame(results)

# Step 15: Display Results in Tabular Format
print("\nEvaluation Metrics (Tabular Format):")
print(results_df)

# Step 16: Graphical Representation
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y=metric, data=results_df, palette="viridis", hue="Model", dodge=False)
    plt.title(f'Model Comparison: {metric}', fontsize=16)
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.legend([], [], frameon=False)  # Remove the legend
    plt.show()

# Step 17: Display Best Model
best_model = results_df.loc[results_df['Accuracy'].idxmax()]
print("\nBest Model Based on Accuracy:")
print(best_model)

# Step 18 Save the Model
# Save the trained Logistic Regression model
joblib.dump(log_reg, 'log_reg.pkl')

# Save the CountVectorizer
joblib.dump(vectorizer, 'count_vectorizer.pkl')

print("Model and Vectorizer saved successfully!")
