import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import re
import json
from typing import List, Dict, Tuple, Optional
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

try:
    import nltk
    from nltk.util import ngrams
    from nltk.tokenize import word_tokenize
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Using basic tokenization.")

class AutocompleteAutocorrectAnalyzer:
    
    def __init__(self):
        self.word_freq = Counter()
        self.bigram_freq = defaultdict(Counter)
        self.trigram_freq = defaultdict(Counter)
        self.word_to_corrections = {}
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
        self.vocabulary = set()
        self.processed_texts = []
        self.dataset_info = {}
        self.df = None
        self.numerical_features = []
        self.categorical_features = []
        self.text_features = []
    csv_path="C:/Users/admin/Documents/Oasis Infobyte/Autocomplete and Autocorrect Data Analytics/creditcard.csv"    
    def load_dataset_from_csv(self, csv_path: str, text_columns: List[str] = None, 
                             sample_size: int = None, encoding: str = 'utf-8'):
        try:
            print(f" Loading dataset from: {csv_path}")
            encodings = [encoding, 'utf-8', 'latin-1', 'cp1252']
            for enc in encodings:
                try:
                    self.df = pd.read_csv(csv_path, encoding=enc)
                    print(f" Dataset loaded successfully with {enc} encoding! Shape: {self.df.shape}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode file with any standard encoding")
            if sample_size and sample_size < len(self.df):
                self.df = self.df.sample(n=sample_size, random_state=42)
                print(f" Sampled {sample_size} rows from dataset")
            self._analyze_dataset_structure()
            if text_columns is None:
                text_columns = self.text_features
                if text_columns:
                    print(f" Auto-detected text columns: {text_columns}")
                else:
                    print("  No text columns detected. Creating synthetic text from numerical features.")
                    text_columns = self._create_synthetic_text_features()
            else:
                print(f" Using specified text columns: {text_columns}")
            valid_columns = [col for col in text_columns if col in self.df.columns]
            if not valid_columns:
                print("  No valid text columns found. Creating synthetic text features.")
                valid_columns = self._create_synthetic_text_features()
            if valid_columns:
                corpus = self._extract_text_from_columns(valid_columns)
                if corpus:
                    self.train_from_corpus(corpus)
                else:
                    print("  No text data extracted. Using sample corpus.")
                    self._use_sample_corpus()
            else:
                print("  Using sample corpus for demonstration.")
                self._use_sample_corpus()
            return True
        except FileNotFoundError:
            print(f" Error: File not found at {csv_path}")
            return False
        except Exception as e:
            print(f" Error loading dataset: {str(e)}")
            return False
    
    def _analyze_dataset_structure(self):
        if self.df is None:
            return
        self.dataset_info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': dict(self.df.dtypes),
            'null_counts': dict(self.df.isnull().sum()),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        self.numerical_features = []
        self.categorical_features = []
        self.text_features = []
        for column in self.df.columns:
            if self.df[column].dtype in ['int64', 'float64']:
                self.numerical_features.append(column)
            elif self.df[column].dtype == 'object':
                sample_values = self.df[column].dropna().astype(str).head(100)
                avg_length = sample_values.str.len().mean()
                unique_ratio = self.df[column].nunique() / len(self.df)
                if avg_length > 10 and unique_ratio > 0.1:
                    self.text_features.append(column)
                else:
                    self.categorical_features.append(column)
            else:
                self.categorical_features.append(column)
        print(f" Dataset Analysis:")
        print(f"    Numerical features: {len(self.numerical_features)}")
        print(f"    Categorical features: {len(self.categorical_features)}")
        print(f"    Text features: {len(self.text_features)}")
        print(f"    Memory usage: {self.dataset_info['memory_usage']:.2f} MB")
    
    def _create_synthetic_text_features(self) -> List[str]:
        if not self.numerical_features:
            return []
        synthetic_texts = []
        for _, row in self.df.iterrows():
            feature_descriptions = []
            for feature in self.numerical_features[:10]:
                value = row[feature]
                if pd.notna(value):
                    if feature.lower() in ['time', 'amount', 'class']:
                        feature_descriptions.append(f"{feature} is {value}")
                    else:
                        if value > 0:
                            feature_descriptions.append(f"{feature} positive {abs(value):.2f}")
                        else:
                            feature_descriptions.append(f"{feature} negative {abs(value):.2f}")
            if feature_descriptions:
                synthetic_texts.append(" ".join(feature_descriptions))
        self.df['synthetic_text'] = synthetic_texts[:len(self.df)]
        print(f"✨ Created synthetic text feature with {len(synthetic_texts)} entries")
        return ['synthetic_text']
    
    def _detect_text_columns(self) -> List[str]:
        return self.text_features
    
    def _extract_text_from_columns(self, columns: List[str]) -> List[str]:
        corpus = []
        for _, row in self.df.iterrows():
            combined_text = ""
            for col in columns:
                if col in self.df.columns and pd.notna(row[col]):
                    combined_text += str(row[col]) + " "
            if combined_text.strip():
                corpus.append(combined_text.strip())
        print(f" Extracted {len(corpus)} text samples from columns: {columns}")
        return corpus
    
    def _use_sample_corpus(self):
        sample_corpus = [
            "machine learning algorithms analyze patterns in large datasets",
            "credit card fraud detection requires sophisticated classification models",
            "principal component analysis reduces dimensionality while preserving variance",
            "anomaly detection identifies unusual transactions in financial data",
            "supervised learning models require labeled training data",
            "cross validation ensures model generalization to unseen data",
            "feature engineering transforms raw data into meaningful representations",
            "ensemble methods combine multiple models for better performance",
            "precision and recall metrics evaluate classification accuracy",
            "data preprocessing cleans and prepares datasets for analysis",
            "statistical analysis reveals hidden patterns in transaction data",
            "neural networks learn complex nonlinear relationships",
            "random forests aggregate decision trees for robust predictions",
            "gradient boosting iteratively improves model performance",
            "support vector machines find optimal decision boundaries",
            "clustering algorithms group similar data points together",
            "dimensionality reduction simplifies high dimensional data",
            "regularization prevents overfitting in complex models",
            "hyperparameter tuning optimizes model performance",
            "feature selection identifies most relevant predictors"
        ]
        self.train_from_corpus(sample_corpus)
        print(" Using sample corpus for autocomplete and autocorrect training")
    
    def preprocess_text(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s\']', '', text)
        if NLTK_AVAILABLE:
            tokens = word_tokenize(text)
        else:
            tokens = text.split()
        tokens = [token for token in tokens if token.strip()]
        return tokens
    
    def train_from_corpus(self, corpus: List[str]):
        print(" Training models from corpus...")
        all_tokens = []
        for text in corpus:
            tokens = self.preprocess_text(text)
            all_tokens.extend(tokens)
            self.processed_texts.append(' '.join(tokens))
        self.word_freq = Counter(all_tokens)
        self.vocabulary = set(all_tokens)
        self._build_ngram_models(all_tokens)
        if self.processed_texts:
            try:
                self.tfidf_vectorizer.fit(self.processed_texts)
            except Exception as e:
                print(f"  TF-IDF training failed: {e}")
        print(f"  Training complete!")
        print(f"    Vocabulary size: {len(self.vocabulary):,}")
        print(f"    Total tokens processed: {len(all_tokens):,}")
        print(f"    Processed texts: {len(self.processed_texts):,}")
    
    def _build_ngram_models(self, tokens: List[str]):
        if NLTK_AVAILABLE:
            for w1, w2 in ngrams(tokens, 2):
                self.bigram_freq[w1][w2] += 1
            for w1, w2, w3 in ngrams(tokens, 3):
                self.trigram_freq[(w1, w2)][w3] += 1
        else:
            for i in range(len(tokens) - 1):
                self.bigram_freq[tokens[i]][tokens[i + 1]] += 1
            for i in range(len(tokens) - 2):
                self.trigram_freq[(tokens[i], tokens[i + 1])][tokens[i + 2]] += 1
    
    def autocomplete(self, prefix: str, n_suggestions: int = 5) -> List[Tuple[str, float]]:
        prefix = prefix.lower().strip()
        if not prefix:
            return []
        suggestions = []
        for word in self.vocabulary:
            if word.startswith(prefix) and word != prefix:
                score = self.word_freq[word] / sum(self.word_freq.values())
                suggestions.append((word, score))
        words = prefix.split()
        if len(words) >= 2:
            last_two = tuple(words[-2:])
            if last_two in self.trigram_freq:
                for next_word, count in self.trigram_freq[last_two].items():
                    full_suggestion = prefix + ' ' + next_word
                    score = count / sum(self.trigram_freq[last_two].values())
                    suggestions.append((full_suggestion, score))
        elif len(words) == 1:
            last_word = words[-1]
            if last_word in self.bigram_freq:
                for next_word, count in self.bigram_freq[last_word].items():
                    full_suggestion = prefix + ' ' + next_word
                    score = count / sum(self.bigram_freq[last_word].values())
                    suggestions.append((full_suggestion, score))
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:n_suggestions]
    
    def autocorrect(self, word: str, max_distance: int = 2) -> List[Tuple[str, float]]:
        word = word.lower().strip()
        if word in self.vocabulary:
            return [(word, 1.0)]
        suggestions = []
        close_matches = difflib.get_close_matches(word, self.vocabulary, n=10, cutoff=0.6)
        for match in close_matches:
            distance = self._edit_distance(word, match)
            if distance <= max_distance:
                freq_score = self.word_freq[match] / sum(self.word_freq.values())
                distance_score = 1 / (1 + distance)
                combined_score = freq_score * distance_score
                suggestions.append((match, combined_score))
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:5]
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
    
    def analyze_dataset_for_fraud_detection(self):
        if self.df is None:
            return
        print("\n FRAUD DETECTION ANALYSIS:")
        print("=" * 50)
        if 'Class' in self.df.columns or 'class' in self.df.columns:
            class_col = 'Class' if 'Class' in self.df.columns else 'class'
            class_dist = self.df[class_col].value_counts()
            print(f" Class Distribution:")
            for class_val, count in class_dist.items():
                percentage = (count / len(self.df)) * 100
                label = "Fraud" if class_val == 1 else "Normal"
                print(f"    {label} ({class_val}): {count:,} ({percentage:.3f}%)")
            if len(class_dist) == 2:
                imbalance_ratio = class_dist.max() / class_dist.min()
                print(f"    Imbalance ratio: {imbalance_ratio:.1f}:1")
            if 'Amount' in self.df.columns:
                print(f"\n Transaction Amount Analysis:")
                amount_stats = self.df['Amount'].describe()
                print(f"    Mean: ${amount_stats['mean']:.2f}")
                print(f"    Median: ${amount_stats['50%']:.2f}")
                print(f"    Max: ${amount_stats['max']:.2f}")
                if class_col in self.df.columns:
                    fraud_amounts = self.df[self.df[class_col] == 1]['Amount']
                    normal_amounts = self.df[self.df[class_col] == 0]['Amount']
                    print(f"    Fraud avg amount: ${fraud_amounts.mean():.2f}")
                    print(f"    Normal avg amount: ${normal_amounts.mean():.2f}")
            if 'Time' in self.df.columns:
                print(f"\n Time Analysis:")
                time_stats = self.df['Time'].describe()
                print(f"    Time range: {time_stats['min']:.0f} - {time_stats['max']:.0f} seconds")
                print(f"    Duration: {(time_stats['max'] - time_stats['min'])/3600:.1f} hours")
    
    def display_dataset_info(self):
        if not self.dataset_info:
            print("No dataset loaded. Please load a dataset first.")
            return
        print("\n" + "=" * 60)
        print(" DATASET INFORMATION")
        print("=" * 60)
        print(f" Dataset Shape: {self.dataset_info['shape']}")
        print(f" Memory Usage: {self.dataset_info['memory_usage']:.2f} MB")
        print(f"\n Feature Categories:")
        print(f"    Numerical: {len(self.numerical_features)} features")
        print(f"    Categorical: {len(self.categorical_features)} features")
        print(f"    Text: {len(self.text_features)} features")
        print(f"\n Column Information (showing first 10):")
        for i, col in enumerate(self.dataset_info['columns'][:10]):
            dtype = self.dataset_info['dtypes'][col]
            null_count = self.dataset_info['null_counts'][col]
            print(f"   {i+1:2d}. {col}: {dtype} ({null_count} nulls)")
        if len(self.dataset_info['columns']) > 10:
            print(f"   ... and {len(self.dataset_info['columns']) - 10} more columns")
        if self.df is not None:
            print("\n Sample Data:")
            print(self.df.head(3).to_string(max_cols=8, max_colwidth=20))
    
    def create_fraud_detection_model(self):
        if self.df is None:
            return None
        class_col = None
        for col in ['Class', 'class', 'target', 'label']:
            if col in self.df.columns:
                class_col = col
                break
        if class_col is None:
            print("  No target column found for fraud detection")
            return None
        print(f"\n BUILDING FRAUD DETECTION MODEL:")
        print("-" * 40)
        feature_cols = [col for col in self.df.columns 
                       if col != class_col and col in self.numerical_features]
        if len(feature_cols) == 0:
            print("  No numerical features found for modeling")
            return None
        X = self.df[feature_cols]
        y = self.df[class_col]
        X = X.fillna(X.mean())
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"  Model trained successfully!")
        print(f"    Features used: {len(feature_cols)}")
        print(f"    Training samples: {len(X_train):,}")
        print(f"    Test samples: {len(X_test):,}")
        print(f"\n Classification Report:")
        print(classification_report(y_test, y_pred))
        return model
    
    def evaluate_autocomplete(self, test_cases: List[Tuple[str, str]]) -> Dict[str, float]:
        if not test_cases:
            return {'top_1_accuracy': 0, 'top_3_accuracy': 0, 'top_5_accuracy': 0, 'total_test_cases': 0}
        correct_predictions = 0
        total_predictions = 0
        top_k_correct = {1: 0, 3: 0, 5: 0}
        for prefix, expected in test_cases:
            suggestions = self.autocomplete(prefix, n_suggestions=5)
            if suggestions:
                total_predictions += 1
                predicted_words = [sugg[0] for sugg in suggestions]
                for k in [1, 3, 5]:
                    if expected in predicted_words[:k]:
                        top_k_correct[k] += 1
                        if k == 1:
                            correct_predictions += 1
        metrics = {
            'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0,
            'top_1_accuracy': top_k_correct[1] / total_predictions if total_predictions > 0 else 0,
            'top_3_accuracy': top_k_correct[3] / total_predictions if total_predictions > 0 else 0,
            'top_5_accuracy': top_k_correct[5] / total_predictions if total_predictions > 0 else 0,
            'total_test_cases': total_predictions
        }
        return metrics
    
    def evaluate_autocorrect(self, test_cases: List[Tuple[str, str]]) -> Dict[str, float]:
        if not test_cases:
            return {'top_1_accuracy': 0, 'top_3_accuracy': 0, 'top_5_accuracy': 0, 'total_test_cases': 0}
        correct_predictions = 0
        total_predictions = 0
        top_k_correct = {1: 0, 3: 0, 5: 0}
        for misspelled, expected in test_cases:
            suggestions = self.autocorrect(misspelled)
            if suggestions:
                total_predictions += 1
                predicted_words = [sugg[0] for sugg in suggestions]
                for k in [1, 3, 5]:
                    if expected in predicted_words[:k]:
                        top_k_correct[k] += 1
                        if k == 1:
                            correct_predictions += 1
        metrics = {
            'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0,
            'top_1_accuracy': top_k_correct[1] / total_predictions if total_predictions > 0 else 0,
            'top_3_accuracy': top_k_correct[3] / total_predictions if total_predictions > 0 else 0,
            'top_5_accuracy': top_k_correct[5] / total_predictions if total_predictions > 0 else 0,
            'total_test_cases': total_predictions
        }
        return metrics
    
    def analyze_corpus_statistics(self) -> Dict:
        if not self.vocabulary:
            return {}
        stats = {
            'vocabulary_size': len(self.vocabulary),
            'total_words': sum(self.word_freq.values()),
            'unique_words': len(self.word_freq),
            'most_common_words': self.word_freq.most_common(10),
            'avg_word_length': np.mean([len(word) for word in self.vocabulary]),
            'word_length_distribution': Counter([len(word) for word in self.vocabulary])
        }
        return stats
    
    def visualize_performance(self, autocomplete_metrics: Dict, autocorrect_metrics: Dict):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        ac_metrics = ['top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy']
        ac_values = [autocomplete_metrics.get(metric, 0) for metric in ac_metrics]
        bars1 = axes[0, 0].bar(['Top-1', 'Top-3', 'Top-5'], ac_values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Autocomplete Performance', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(axis='y', alpha=0.3)
        for bar, value in zip(bars1, ac_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom')
        ar_values = [autocorrect_metrics.get(metric, 0) for metric in ac_metrics]
        bars2 = axes[0, 1].bar(['Top-1', 'Top-3', 'Top-5'], ar_values, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Autocorrect Performance', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(axis='y', alpha=0.3)
        for bar, value in zip(bars2, ar_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom')
        if self.word_freq:
            words, freqs = zip(*self.word_freq.most_common(15))
            axes[1, 0].bar(range(len(words)), freqs, color='lightgreen', alpha=0.7)
            axes[1, 0].set_title('Top 15 Most Frequent Words', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Word Rank')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_xticks(range(len(words)))
            axes[1, 0].set_xticklabels(words, rotation=45, ha='right')
        stats = self.analyze_corpus_statistics()
        if stats and 'word_length_distribution' in stats:
            lengths = list(stats['word_length_distribution'].keys())
            counts = list(stats['word_length_distribution'].values())
            axes[1, 1].bar(lengths, counts, color='gold', alpha=0.7)
            axes[1, 1].set_title('Word Length Distribution', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Word Length')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, autocomplete_metrics: Dict, autocorrect_metrics: Dict):
        stats = self.analyze_corpus_statistics()
        print("=" * 60)
        print(" AUTOCOMPLETE AND AUTOCORRECT ANALYSIS REPORT")
        print("=" * 60)
        if stats:
            print("\n CORPUS STATISTICS")
            print(f"    Vocabulary size      : {stats['vocabulary_size']:,}")
            print(f"    Total words          : {stats['total_words']:,}")
            print(f"    Unique words         : {stats['unique_words']:,}")
            print(f"    Avg word length      : {stats['avg_word_length']:.2f}")
            print(f"    Most common words    :")
            for word, freq in stats['most_common_words']:
                print(f"     - {word:15s} → {freq}")
            print("\n  AUTOCOMPLETE PERFORMANCE")
            print(f"    Total Test Cases     : {autocomplete_metrics.get('total_test_cases', 0)}")
            print(f"    Top-1 Accuracy       : {autocomplete_metrics.get('top_1_accuracy', 0):.3f}")
            print(f"    Top-3 Accuracy       : {autocomplete_metrics.get('top_3_accuracy', 0):.3f}")
            print(f"    Top-5 Accuracy       : {autocomplete_metrics.get('top_5_accuracy', 0):.3f}")
            print("\n  AUTOCORRECT PERFORMANCE")
            print(f"    Total Test Cases     : {autocorrect_metrics.get('total_test_cases', 0)}")
            print(f"    Top-1 Accuracy       : {autocorrect_metrics.get('top_1_accuracy', 0):.3f}")
            print(f"    Top-3 Accuracy       : {autocorrect_metrics.get('top_3_accuracy', 0):.3f}")
            print(f"    Top-5 Accuracy       : {autocorrect_metrics.get('top_5_accuracy', 0):.3f}")
            print("\n TIP: Use `visualize_performance()` to see visual metrics.")
            print("=" * 60)
            
if __name__ == "__main__":
    analyzer = AutocompleteAutocorrectAnalyzer()
    csv_path = "C:/Users/admin/Documents/Oasis Infobyte/Autocomplete and Autocorrect Data Analytics/creditcard.csv"
    analyzer.load_dataset_from_csv(csv_path)
    analyzer.display_dataset_info()
    analyzer.analyze_dataset_for_fraud_detection()
    model = analyzer.create_fraud_detection_model()
    autocomplete_tests = [
        ("credit", "card"), 
        ("machine", "learning"), 
        ("data", "analysis")
    ]
    autocomplete_metrics = analyzer.evaluate_autocomplete(autocomplete_tests)
    autocorrect_tests = [
        ("anlysis", "analysis"), 
        ("traning", "training"), 
        ("fraudl", "fraud")
    ]
    autocorrect_metrics = analyzer.evaluate_autocorrect(autocorrect_tests)
    analyzer.generate_report(autocomplete_metrics, autocorrect_metrics)
    analyzer.visualize_performance(autocomplete_metrics, autocorrect_metrics)
