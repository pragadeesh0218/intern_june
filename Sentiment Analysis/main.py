import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

class TwitterSentimentAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.label_mapping = {}
        self.model = None

    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"@\w+|#\w+", "", text)
        text = re.sub(r"^rt[\s]+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words and len(t) > 2]
        return " ".join(tokens)

    def load_twitter_dataset(self, file_path):
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, header=None)
                    break
                except:
                    continue
        if df is None:
            print("Failed to load dataset.")
            return None

        if len(df.columns) == 6:
            label_col, text_col = df.columns[0], df.columns[5]
        elif len(df.columns) == 2:
            avg_len = df.astype(str).applymap(len).mean()
            text_col = df.columns[0] if avg_len[0] > avg_len[1] else df.columns[1]
            label_col = df.columns[1] if text_col == df.columns[0] else df.columns[0]
        elif len(df.columns) == 3:
            label_col, text_col = df.columns[1], df.columns[2]
        else:
            print("Please specify column names manually.")
            return None

        df = df.dropna(subset=[text_col, label_col])
        df = df.drop_duplicates(subset=[text_col])

        if df[label_col].dtype in ['int64', 'float64']:
            unique_vals = sorted(df[label_col].unique())
            if set(unique_vals) == {0, 4}:
                label_map = {0: 'negative', 4: 'positive'}
            elif set(unique_vals) == {0, 1}:
                label_map = {0: 'negative', 1: 'positive'}
            elif set(unique_vals) == {0, 2, 4}:
                label_map = {0: 'negative', 2: 'neutral', 4: 'positive'}
            else:
                label_map = {val: f'sentiment_{val}' for val in unique_vals}
            df[label_col] = df[label_col].map(label_map)

        df['clean_text'] = df[text_col].apply(self.clean_text)
        df = df[df['clean_text'].str.len() > 0]

        unique_labels = sorted(df[label_col].unique())
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        df['label_numeric'] = df[label_col].map(self.label_mapping)

        print(f"Dataset loaded with shape: {df.shape}")
        print(f"Label distribution:\n{df[label_col].value_counts()}")
        return df

    def analyze_sentiment(self, df):
        X = df['clean_text']
        y = df['label_numeric']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        models = {
            'Naive Bayes': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                ('nb', MultinomialNB())
            ]),
            'Logistic Regression': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                ('lr', LogisticRegression(random_state=42, max_iter=1000))
            ])
        }

        results = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'report': classification_report(y_test, y_pred, zero_division=0)
            }
            print(f"Accuracy: {accuracy:.4f}")
        return results, X_test, y_test

    def visualize_results(self, results, df, y_test):
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        bars = axes[0, 0].bar(model_names, accuracies, color=['#FF6B6B', '#4ECDC4'])
        axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        for bar, acc in zip(bars, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{acc:.3f}', ha='center', fontweight='bold')

        sentiment_counts = df['label_numeric'].value_counts().sort_index()
        labels = [k for k, v in sorted(self.label_mapping.items(), key=lambda x: x[1])]
        axes[0, 1].pie(sentiment_counts.values, labels=labels,
                       autopct='%1.1f%%', colors=['#FF6B6B', '#FFD93D', '#4ECDC4'][:len(labels)])
        axes[0, 1].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')

        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_predictions = results[best_model_name]['predictions']
        cm = confusion_matrix(y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')

        df['text_length'] = df['clean_text'].str.len()
        for label in df['label_numeric'].unique():
            label_data = df[df['label_numeric'] == label]['text_length']
            axes[1, 1].hist(label_data, alpha=0.6, bins=30, label=f"Class {label}")
        axes[1, 1].set_title('Tweet Length Distribution by Sentiment', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Tweet Length')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

        print("\n" + "="*50)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*50)
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print("Classification Report:")
            print(result['report'])

    def predict(self, text):
        if not self.model:
            print("Model not trained.")
            return None
        cleaned = self.clean_text(text)
        pred = self.model.predict([cleaned])[0]
        reverse_map = {v: k for k, v in self.label_mapping.items()}
        return reverse_map[pred]

if __name__ == "__main__":
    file_path = 'C:/Users/admin/Documents/Oasis Infobyte/Sentiment Analysis/Twitter_Data.csv'
    analyzer = TwitterSentimentAnalyzer()
    df = analyzer.load_twitter_dataset(file_path)

    if df is not None:
        results, X_test, y_test = analyzer.analyze_sentiment(df)
        analyzer.visualize_results(results, df, y_test)
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        analyzer.model = results[best_model_name]['model']
        print(f"\nBest Model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")

        sample = "I love this new phone. It's amazing!"
        print(f"\nSample Prediction:\nText: {sample}")
        print("Predicted Sentiment:", analyzer.predict(sample))
