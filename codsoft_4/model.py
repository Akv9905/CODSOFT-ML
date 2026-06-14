
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# Set visualization styles
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("Step 1: Loading and exploring dataset...")
# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Drop unnecessary empty columns
df = df.dropna(how='all', axis=1)
df = df.iloc[:, [0, 1]]
df.columns = ['label', 'text']

print(f"Dataset dimensions: {df.shape}")

# Drop duplicates
df = df.drop_duplicates(keep='first').reset_index(drop=True)
print(f"Dimensions after dropping duplicates: {df.shape}")

# Plot Class Label Distribution
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='label', data=df, hue='label', legend=False, palette='Set2')
plt.title('Distribution of Spam vs Ham Messages')
plt.xlabel('Message Classification')
plt.ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
plt.savefig('class_distribution.png', bbox_inches='tight', dpi=150)
plt.close()

# Message length analysis
df['length'] = df['text'].apply(len)

plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='length', hue='label', bins=50, kde=True, multiple='stack', palette='Set2')
plt.title('Distribution of Message Length by Label')
plt.xlabel('Message Length (Characters)')
plt.ylabel('Frequency')
plt.xlim(0, 300)
plt.savefig('length_distribution.png', bbox_inches='tight', dpi=150)
plt.close()

print("\nStep 2: Preprocessing and text cleaning...")
# Convert label to binary: ham -> 0, spam -> 1
df['label_bin'] = df['label'].map({'ham': 0, 'spam': 1})

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

df['clean_text'] = df['text'].apply(clean_text)

print("\nStep 3: Splitting into Train and Test sets...")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['clean_text'], 
    df['label_bin'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label_bin']
)
print(f"Training size: {X_train_raw.shape[0]}, Test size: {X_test_raw.shape[0]}")

print("\nStep 4: Vectorizing using TF-IDF...")
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_features=5000
)
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

print("\nStep 5: Training Classifiers...")
models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(C=1.0, random_state=42),
    "Support Vector Machine": SVC(C=1.0, kernel='linear', probability=True, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
    
    results[name] = {
        "model": model,
        "predictions": y_pred,
        "probabilities": y_pred_proba,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba) if hasattr(model, "predict_proba") else None
    }

print("\nStep 6: Evaluating Models...")
results_df = pd.DataFrame(results).T.drop(columns=['model', 'predictions', 'probabilities'])
print("\nClassifier Performance Summary:")
print(results_df.to_string())

# Save metric comparison plot
results_df[['accuracy', 'precision', 'recall', 'f1']].plot(kind='bar', figsize=(12, 6))
plt.title('Classifier Performance Comparison')
plt.ylabel('Score')
plt.ylim(0.8, 1.02)
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('performance_comparison.png', bbox_inches='tight', dpi=150)
plt.close()

# Plot Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    axes[idx].set_title(f'{name} Confusion Matrix')
    axes[idx].set_xlabel('Predicted Label')
    axes[idx].set_ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrices.png', bbox_inches='tight', dpi=150)
plt.close()

# Plot ROC Curves
plt.figure(figsize=(10, 7))
for name, res in results.items():
    if res['roc_auc'] is not None:
        fpr, tpr, _ = roc_curve(y_test, res['probabilities'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend()
plt.savefig('roc_curves.png', bbox_inches='tight', dpi=150)
plt.close()

# Print detailed reports
for name, res in results.items():
    print("\n" + "=" * 60)
    print(f"Classification Report for {name}:")
    print("=" * 60)
    print(classification_report(y_test, res['predictions'], target_names=['Ham', 'Spam']))

# Feature Importance
lr_model = results['Logistic Regression']['model']
feature_names = np.array(vectorizer.get_feature_names_out())
coefficients = lr_model.coef_[0]
feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
top_spam = feature_importance.sort_values(by='Coefficient', ascending=False).head(15)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_spam, x='Coefficient', y='Feature', hue='Feature', legend=False, palette='Reds_r')
plt.title('Top 15 Words Predicting Spam (Logistic Regression)')
plt.xlabel('Coefficient Weight (Spam Strength)')
plt.ylabel('Word / Phrase')
plt.savefig('top_spam_words.png', bbox_inches='tight', dpi=150)
plt.close()

print("\nStep 7: Saving the Best Model...")
best_model_name = results_df['f1'].astype(float).idxmax()
best_model = results[best_model_name]['model']
print(f"Best model selected: {best_model_name}")

joblib.dump(best_model, 'spam_classifier.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
print("Model saved to 'spam_classifier.joblib'")
print("Vectorizer saved to 'tfidf_vectorizer.joblib'")

print("\nStep 8: Inference Demonstration...")
def predict_message(message):
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = best_model.predict(vectorized)[0]
    
    if hasattr(best_model, "predict_proba"):
        prob = best_model.predict_proba(vectorized)[0][1]
        label = "SPAM" if prediction == 1 else "LEGITIMATE (HAM)"
        return f"Message: '{message}'\nResult: {label} (Probability of Spam: {prob * 100:.2f}%)"
    else:
        label = "SPAM" if prediction == 1 else "LEGITIMATE (HAM)"
        return f"Message: '{message}'\nResult: {label}"

test_messages = [
    "Hey! Are we still meeting for lunch today? Let me know.",
    "CONGRATULATIONS! You have been selected to win a cash prize of £1000. Text CLAIM to 81010 now to claim your reward. T&Cs apply.",
    "URGENT: Your account balance is low. Click here www.bank-update-alert.com to verify your details immediately.",
    "Can you please send me the class notes from yesterday? I was feeling sick.",
    "Get 50% off on your next purchase! Use code GET50. Valid for 24 hours."
]

for msg in test_messages:
    print("-" * 60)
    print(predict_message(msg))
print("-" * 60)
