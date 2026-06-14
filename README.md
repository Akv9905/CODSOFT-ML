# CodSoft Machine Learning Internship Projects

This repository contains machine learning projects developed as part of the **CodSoft Machine Learning Internship**. It features three distinct projects ranging from tabular classification to natural language processing (NLP) tasks.

---

##  Repository Structure

The workspace is organized as follows:
```text
CODSOFT/
├── .vscode/
│   └── settings.json
├── ML/
│   ├── codsoft_1/
│   │   ├── Credit.csv            # https://www.kaggle.com/datasets/kartik2112/fraud-detection
│   │   └── model.ipynb           # Random Forest training & evaluation
│   ├── codsoft_2/
│   │   ├── train_data.txt        # https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb
│   │   ├── test_data.txt         # https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb
│   │   ├── test_predictions.csv  # Output classification predictions
│   │   └── model.ipynb           # Text classification modeling (TF-IDF + SVM/NB)
│   └── codsoft_3/
│       ├── spam.csv              # Raw SMS message dataset
│       ├── model.py              # End-to-end Python modeling script
│       └── [Output Plots]*       # Generated data analysis & metrics charts
└── README.md                     # Repository documentation
```
*\*Note: Running `model.py` generates model performance plots and saved model assets in the `SMS Spam Detection` directory.*

---

##  Installation & Setup

To run the notebooks and scripts in this repository, set up a Python environment with the required dependencies:

1. **Clone or download** this repository to your local system.
2. **Navigate** to the root folder of the workspace.
3. **Install the dependencies** via `pip`:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn joblib
   ```
4. Configure your preferred Python environment (e.g., using Conda as configured in [.vscode/settings.json](file:///c:/Users/amogh/Downloads/CODSOFT/.vscode/settings.json)).

---

## Projects Overview

### 1. Credit Card Fraud Detection
* **Directory**: [Credit Card fraud](file:///c:/Users/amogh/Downloads/CODSOFT/ML/Credit%20Card%20fraud)
* **Goal**: Detect fraudulent credit card transactions to prevent unauthorized usage.
* **Dataset**: [Credit.csv](file:///c:/Users/amogh/Downloads/CODSOFT/ML/Credit%20Card%20fraud/Credit.csv) containing transaction data (amounts, category, locations, population, etc.).
* **Methodology**:
  1. **Preprocessing**: Removed identifiers and non-informative variables (`Unnamed: 0`, `trans_date_trans_time`, `first`, `last`, `street`, `trans_num`, `dob`).
  2. **Encoding**: Transformed categorical descriptors (`merchant`, `category`, `gender`, `city`, `state`, `job`) using label encoding.
  3. **Handling Imbalance**: Split into an 80/20 train/test split utilizing a stratified approach to handle target class imbalance (`is_fraud` label).
  4. **Modeling**: Trained a `RandomForestClassifier` (100 estimators, all cores).
* **Evaluation**:
  * **Overall Accuracy**: **99.83%**
  * **Class 0 (Legitimate)**: Precision: **1.00** | Recall: **1.00** | F1-Score: **1.00**
  * **Class 1 (Fraudulent)**: Precision: **0.94** | Recall: **0.60** | F1-Score: **0.73**
  * **Confusion Matrix**:
    ```text
    [[110699     16]  <- Legitimate (Ham)
     [   173    256]] <- Fraudulent (Spam)
    ```

---

### 2. Movie Genre Classification
* **Directory**: [Movie](file:///c:/Users/amogh/Downloads/CODSOFT/ML/Movie)
* **Goal**: Predict the genre of a movie based on its raw textual plot summary.
* **Dataset**:
  * [train_data.txt](file:///c:/Users/amogh/Downloads/CODSOFT/ML/Movie/train_data.txt) (for training labels)
  * [test_data.txt](file:///c:/Users/amogh/Downloads/CODSOFT/ML/Movie/test_data.txt) (unlabeled validation test summaries)
* **Methodology**:
  1. **Text Cleaning**: Sanitized inputs by lowercasing, stripping HTML markup, deleting punctuation and digits, and shrinking extra whitespaces.
  2. **Feature Extraction**: Applied `TfidfVectorizer` mapping up to 20,000 unigram/bigram features with standard English stop word filtering.
  3. **Modeling**: Trained and validated two text classification algorithms:
     * **Multinomial Naive Bayes** (`MultinomialNB`): Accuracy **50.3%**
     * **Linear Support Vector Classifier** (`LinearSVC`): Accuracy **57.0%**
  4. **Output**: Predictions exported to [test_predictions.csv](file:///c:/Users/amogh/Downloads/CODSOFT/ML/Movie/test_predictions.csv).
* **Inference Usage**:
  Within the notebook, the helper function `predict_genre(description)` classifies custom text inputs. Example:
  ```python
  predict_genre("In a futuristic city, an astronaut sets off on a dangerous space flight...")
  # Output: sci-fi
  ```

---

### 3. SMS Spam Detection
* **Directory**: [SMS Spam Detection](file:///c:/Users/amogh/Downloads/CODSOFT/ML/SMS%20Spam%20Detection)
* **Goal**: Build an NLP model that classifies SMS messages as either `spam` or `ham` (legitimate).
* **Dataset**: [spam.csv](file:///c:/Users/amogh/Downloads/CODSOFT/ML/SMS%20Spam%20Detection/spam.csv)
* **Pipeline** ([model.py](file:///c:/Users/amogh/Downloads/CODSOFT/ML/SMS%20Spam%20Detection/model.py)):
  1. **Data Preprocessing & EDA**: Dropped duplicates and parsed binary spam classifications (`ham` -> 0, `spam` -> 1). Computed message length variables and exported descriptive histograms.
  2. **Cleaning & Vectorization**: Formatted text (removed URLs, emails, HTML tags, punctuation, numbers) and extracted features using `TfidfVectorizer` (max 5,000 features).
  3. **Model Evaluation**: Fitted three separate classification models:
     * Naive Bayes (MultinomialNB)
     * Logistic Regression
     * Support Vector Machine (SVC with linear kernel)
  4. **Visual Analysis outputs**:
     * Class distributions and message length distributions.
     * Classifier performance comparisons (Accuracy, Precision, Recall, F1).
     * Confusion matrices and ROC Curves.
     * Feature importance coefficient weights for Logistic Regression.
  5. **Model Persistence**: Serializes the best classifier and vectorizer to `spam_classifier.joblib` and `tfidf_vectorizer.joblib`.
* **Execution**:
  Run the script directly via terminal:
  ```bash
  python "ML/SMS Spam Detection/model.py"
  ```
  It prints evaluation reports and tests custom inputs dynamically.
