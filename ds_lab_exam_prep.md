# Data Science Lab Exam Preparation (10 Programs)

## One-time Jupyter Setup (run once)
```python
# Run this once in Jupyter if packages are missing
# !pip -q install numpy pandas matplotlib seaborn scikit-learn wordcloud mlxtend
```

## Program 1: NumPy and Pandas Basic Operations
**Description:** Demonstrates core NumPy array operations and Pandas Series/DataFrame operations (5 each conceptually).

**Code:**
```python
import numpy as np
import pandas as pd

# NumPy operations
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])

print("NumPy Add:", a + b)
print("NumPy Sub:", a - b)
print("NumPy Mul:", a * b)
print("NumPy Div:", a / b)
print("NumPy Mean:", np.mean(a), "Std:", np.std(a))

# Pandas operations
s = pd.Series([10, 20, 30, 40, 50], name="marks")
print("\nSeries Mean:", s.mean())

df = pd.DataFrame({
    "Name": ["A", "B", "C", "D"],
    "Marks": [78, 65, 92, 55],
    "Dept": ["CS", "IS", "CS", "EC"]
})

print("\nDataFrame:\n", df)
print("\nRows with Marks > 70:\n", df[df["Marks"] > 70])
print("\nGroupby Dept (avg marks):\n", df.groupby("Dept")["Marks"].mean())
print("\nSorted by Marks:\n", df.sort_values("Marks", ascending=False))
```

**Expected Output:**
- NumPy arithmetic arrays printed correctly.
- Mean and standard deviation values displayed.
- DataFrame filtering, grouping, and sorting outputs shown.

**Viva Questions:**
1. What is the difference between Python list and NumPy array? - NumPy arrays are faster, homogeneous, and support vectorized operations.
2. Why use `groupby()` in Pandas? - To aggregate data category-wise (sum/mean/count etc.).
3. What does vectorization mean? - Performing operations on whole arrays without Python loops.

**Key Concepts:** `ndarray`, element-wise operations, `mean/std`, Pandas `Series`, `DataFrame`, filtering, `groupby`, sorting.

---

## Program 2: AutoMPG Data Preprocessing and Summary
**Description:** Loads AutoMPG dataset, inspects structure (`shape`, `info`, `describe`), handles missing values, and plots histograms.

**Code:**
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Use your lab CSV if available:
# df = pd.read_csv("auto-mpg.csv", na_values='?')
df = sns.load_dataset("mpg").copy()

print("Shape:", df.shape)
print("\nInfo:")
df.info()
print("\nSummary:\n", df.describe(numeric_only=True))
print("\nNull values:\n", df.isna().sum())

# Treat missing values
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())
df = df.dropna().reset_index(drop=True)

# Histograms (continuous variables)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(df["mpg"], kde=True, ax=ax[0])
ax[0].set_title("MPG Distribution")
sns.histplot(df["horsepower"], kde=True, ax=ax[1])
ax[1].set_title("Horsepower Distribution")
plt.tight_layout()
plt.show()
```

**Expected Output:**
- Dataset dimensions, structure, and summary table.
- Null count per column.
- Two histograms for `mpg` and `horsepower`.

**Viva Questions:**
1. Why do we handle missing values before modeling? - ML algorithms fail or become biased with nulls.
2. Difference between `info()` and `describe()`? - `info()` gives schema/null counts; `describe()` gives statistical summary.
3. Why histogram here? - To understand distribution/skewness of continuous variables.

**Key Concepts:** preprocessing, null handling, data inspection, descriptive statistics, histogram.

---

## Program 3: AutoMPG EDA (Histogram, Scatter, Count, Point Plot)
**Description:** Performs EDA on AutoMPG: distribution of continuous variables, relationship between two continuous variables, frequency of categorical values, and point plot.

**Code:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Use preprocessed AutoMPG in lab if available
# autompg = pd.read_csv("auto-mpg-clean.csv")
autompg = sns.load_dataset("mpg").dropna().copy()

cats = list(autompg.select_dtypes(include=["object", "category"]).columns)
nums = list(autompg.select_dtypes(exclude=["object", "category"]).columns)
print("Categorical variables:", cats)
print("Numerical variables:", nums)

fig, ax = plt.subplots(2, 2, figsize=(12, 8))

sns.histplot(autompg["mpg"], kde=True, ax=ax[0, 0])
ax[0, 0].set_title("Histogram: MPG")

sns.scatterplot(data=autompg, x="horsepower", y="mpg", ax=ax[0, 1])
ax[0, 1].set_title("Scatter: Horsepower vs MPG")

sns.countplot(data=autompg, x="origin", ax=ax[1, 0])
ax[1, 0].set_title("Count Plot: Origin")

sns.pointplot(data=autompg, x="origin", y="mpg", ax=ax[1, 1])
ax[1, 1].set_title("Point Plot: Origin vs MPG")

plt.tight_layout()
plt.show()
```

**Expected Output:**
- Printed list of categorical and numerical columns.
- 4 plots: histogram, scatter, countplot, pointplot.

**Viva Questions:**
1. What does scatter plot indicate here? - Correlation trend between horsepower and mpg.
2. Why use count plot for categorical data? - To view frequency distribution of categories.
3. What insight does point plot provide? - Category-wise average trend of a continuous variable.

**Key Concepts:** EDA, continuous vs categorical variables, correlation visualization, categorical frequency.

---

## Program 4: KNN for Breast Cancer Diagnosis
**Description:** Uses KNN classification to predict diagnosis (`malignant`/`benign`) using `mean texture` and `mean radius`.

**Code:**
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Lab option: df = pd.read_csv("breast_cancer.csv")
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

X = df[["mean texture", "mean radius"]]
y = pd.Series(data.target).map({0: "malignant", 1: "benign"})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=["malignant", "benign"])
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["malignant", "benign"],
            yticklabels=["malignant", "benign"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion Matrix")
plt.show()
```

**Expected Output:**
- Accuracy (generally high, around 0.85+ depending on split).
- Classification report with precision/recall/F1.
- 2x2 confusion matrix heatmap.

**Viva Questions:**
1. Why is feature scaling important in KNN? - KNN uses distance; larger-scale features can dominate.
2. What does `k=5` mean? - Classification is based on the majority class among 5 nearest neighbors.
3. Is KNN parametric? - No, KNN is a non-parametric, instance-based learner.

**Key Concepts:** supervised learning, KNN, Euclidean distance, train/test split, scaling, confusion matrix.

---

## Program 5: Decision Tree for Student Performance (Pass/Fail)
**Description:** Preprocesses student data, predicts `Pass`/`Fail` using Decision Tree, plots tree, evaluates model, and identifies important features.

**Code:**
```python
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

file = Path("student_performance.csv")
if file.exists():
    df = pd.read_csv(file)
else:
    np.random.seed(42)
    n = 250
    df = pd.DataFrame({
        "study_hours": np.random.randint(1, 10, n),
        "attendance": np.random.randint(50, 100, n),
        "internal": np.random.randint(20, 50, n),
        "assignment": np.random.randint(10, 30, n)
    })
    score = 3*df.study_hours + 0.4*df.attendance + 0.7*df.internal + 0.5*df.assignment
    df["result"] = np.where(score >= np.median(score), "Pass", "Fail")

if "result" not in df.columns:
    # If your column name differs, change here
    df["result"] = np.where(df.iloc[:, -1] >= df.iloc[:, -1].median(), "Pass", "Fail")

X = pd.get_dummies(df.drop(columns=["result"]), drop_first=True)
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

clf = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=20)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, pos_label="Pass"), 4))
print("Recall:", round(recall_score(y_test, y_pred, pos_label="Pass"), 4))
print("F1:", round(f1_score(y_test, y_pred, pos_label="Pass"), 4))

cm = confusion_matrix(y_test, y_pred, labels=["Pass", "Fail"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=["Pass", "Fail"], yticklabels=["Pass", "Fail"])
plt.title("Decision Tree Confusion Matrix")
plt.show()

imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop features:\n", imp.head())

plt.figure(figsize=(10, 5))
plot_tree(clf, feature_names=X.columns, class_names=["Fail", "Pass"], filled=True)
plt.title("Decision Tree")
plt.show()
```

**Expected Output:**
- Accuracy, precision, recall, and F1 score.
- Confusion matrix heatmap.
- Feature-importance list.
- Decision tree plot.

**Viva Questions:**
1. What is Gini index? - A measure of impurity used to choose best split in classification trees.
2. Why limit `max_depth`? - To reduce overfitting.
3. How do you interpret feature importance? - Higher value means stronger contribution in split decisions.

**Key Concepts:** Decision Tree, impurity, overfitting, classification metrics, feature importance.

---

## Program 6: Random Forest for Student Performance + Tuning
**Description:** Applies Random Forest on student data, tunes hyperparameters, plots feature importance, and evaluates performance.

**Code:**
```python
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

file = Path("student_performance.csv")
if file.exists():
    df = pd.read_csv(file)
else:
    np.random.seed(42)
    n = 250
    df = pd.DataFrame({
        "study_hours": np.random.randint(1, 10, n),
        "attendance": np.random.randint(50, 100, n),
        "internal": np.random.randint(20, 50, n),
        "assignment": np.random.randint(10, 30, n)
    })
    score = 3*df.study_hours + 0.4*df.attendance + 0.7*df.internal + 0.5*df.assignment
    df["result"] = np.where(score >= np.median(score), "Pass", "Fail")

if "result" not in df.columns:
    df["result"] = np.where(df.iloc[:, -1] >= df.iloc[:, -1].median(), "Pass", "Fail")

X = pd.get_dummies(df.drop(columns=["result"]), drop_first=True)
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

rf = RandomForestClassifier(random_state=42)
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

search = RandomizedSearchCV(rf, param_distributions=param_dist,
                            n_iter=8, cv=5, scoring="accuracy",
                            random_state=42, n_jobs=-1)
search.fit(X_train, y_train)

best_rf = search.best_estimator_
y_pred = best_rf.predict(X_test)

print("Best Params:", search.best_params_)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["Pass", "Fail"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["Pass", "Fail"], yticklabels=["Pass", "Fail"])
plt.title("Random Forest Confusion Matrix")
plt.show()

imp = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=imp.values, y=imp.index)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.show()
```

**Expected Output:**
- Best hyperparameters from randomized search.
- Test accuracy and classification report.
- Confusion matrix and feature-importance bar chart.

**Viva Questions:**
1. Why Random Forest over one Decision Tree? - It reduces variance using ensemble of many trees.
2. What does `n_estimators` mean? - Number of trees in the forest.
3. Why hyperparameter tuning? - To improve generalization and avoid under/overfitting.

**Key Concepts:** ensemble learning, bagging, hyperparameter tuning, RandomizedSearchCV, feature importance.

---

## Program 7: Naive Bayes for Placement Prediction
**Description:** Builds Naive Bayes classifier on placement data, displays prior/posterior probabilities, and evaluates model.

**Code:**
```python
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

file = Path("Placement data.csv")
if file.exists():
    placement = pd.read_csv(file)
else:
    np.random.seed(7)
    n = 250
    placement = pd.DataFrame({
        "tenthmarks": np.random.randint(50, 100, n),
        "twelthmarks": np.random.randint(45, 100, n),
        "Ugmarks": np.random.randint(50, 95, n),
        "Pgmarks": np.random.randint(50, 95, n),
        "communication": np.random.randint(1, 11, n),
        "internships": np.random.randint(0, 4, n),
        "stream": np.random.choice(["CS", "IS", "EC"], n)
    })
    s = (placement.tenthmarks + placement.twelthmarks + placement.Ugmarks + placement.Pgmarks + 5*placement.communication + 6*placement.internships)
    placement["Placement"] = np.where(s >= np.percentile(s, 55), "Placed", "Not Placed")

# Encode X
X = pd.get_dummies(placement.drop(columns=["Placement"]), drop_first=True)
y = placement["Placement"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

# Prior and posterior probabilities
print("Class labels:", nb.classes_)
print("Prior probabilities:", nb.class_prior_)
posterior = nb.predict_proba(X_test.iloc[[0]])[0]
print("Posterior (first test row):", dict(zip(nb.classes_, posterior)))

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["Placed", "Not Placed"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Placed", "Not Placed"], yticklabels=["Placed", "Not Placed"])
plt.title("Naive Bayes Confusion Matrix")
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(nb.classes_, nb.class_prior_)
ax[0].set_title("Prior Probability")
ax[1].bar(nb.classes_, posterior)
ax[1].set_title("Posterior Probability (1 sample)")
plt.tight_layout()
plt.show()
```

**Expected Output:**
- Prior probabilities per class.
- Posterior probabilities for sample input.
- Accuracy, report, and confusion matrix.

**Viva Questions:**
1. What is the “naive” assumption? - Features are conditionally independent given class.
2. Difference between prior and posterior probability? - Prior is before seeing data; posterior is after evidence.
3. When to use GaussianNB? - When numeric features are assumed normally distributed.

**Key Concepts:** Bayes theorem, prior/posterior, conditional probability, Gaussian Naive Bayes, classification metrics.

---

## Program 8: Association Rule Mining (Apriori)
**Description:** Performs market basket analysis using Apriori and extracts best association rules based on support/confidence/lift.

**Code:**
```python
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Option A (lab file): one row = comma-separated items
# raw = pd.read_csv("market_basket.csv")
# transactions = raw.iloc[:, 0].dropna().apply(lambda x: [i.strip() for i in str(x).split(',') if i.strip()]).tolist()

# Option B: demo transactions (runs directly)
transactions = [
    ["milk", "bread", "eggs"],
    ["milk", "bread"],
    ["milk", "butter"],
    ["bread", "butter"],
    ["milk", "bread", "butter"],
    ["bread", "eggs"],
    ["milk", "eggs"],
    ["milk", "bread", "butter", "eggs"]
]

te = TransactionEncoder()
arr = te.fit(transactions).transform(transactions)
basket = pd.DataFrame(arr, columns=te.columns_)

freq = apriori(basket, min_support=0.2, use_colnames=True)
rules = association_rules(freq, metric="confidence", min_threshold=0.5)
rules = rules.sort_values(["support", "confidence"], ascending=False)

print("Frequent itemsets:\n", freq)
print("\nTop rules:\n", rules[["antecedents", "consequents", "support", "confidence", "lift"]].head())
```

**Expected Output:**
- Frequent itemsets table with support.
- Rules table with antecedents, consequents, support, confidence, and lift.
- Example rule like `{milk} -> {bread}` with high confidence.

**Viva Questions:**
1. What is support? - Fraction of transactions containing an itemset.
2. What is confidence? - Probability of consequent given antecedent.
3. What does lift > 1 mean? - Positive association (items co-occur more than random chance).

**Key Concepts:** Apriori, frequent itemset, support, confidence, lift, market basket analysis.

---

## Program 9: K-Means Clustering for Mall Customers
**Description:** Applies K-Means on customer data, finds optimal K using elbow method, and visualizes clusters.

**Code:**
```python
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

file = Path("Mall_Customers.csv")
if file.exists():
    df = pd.read_csv(file)
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
else:
    X, _ = make_blobs(n_samples=200, centers=5, cluster_std=1.2, random_state=42)

# Elbow method
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    wcss.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# Train KMeans with K=5
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

plt.figure(figsize=(7, 5))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette="tab10", s=60)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="black", s=180, marker="X", label="Centroids")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("K-Means Clusters")
plt.legend()
plt.show()
```

**Expected Output:**
- Elbow graph with visible bend near optimal K.
- Cluster scatter plot with 5 colored groups and centroid markers.

**Viva Questions:**
1. Why is K-Means unsupervised? - It uses no target labels.
2. What is WCSS? - Sum of squared distances of points from their cluster centroid.
3. Why use elbow method? - To choose suitable number of clusters.

**Key Concepts:** clustering, centroid, WCSS/inertia, elbow method, unsupervised learning.

---

## Program 10: Text Preprocessing and Word Cloud
**Description:** Performs basic text preprocessing (tokenization, punctuation/stopword removal), identifies frequent words, and generates a word cloud.

**Code:**
```python
from pathlib import Path
import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

file = Path("text_data.txt")
if file.exists():
    text = file.read_text(encoding="utf-8", errors="ignore")
else:
    text = """
    Data science uses statistics, machine learning, and visualization.
    Data preprocessing, feature engineering, and model evaluation are important.
    Visualization helps explain data clearly.
    """

# Tokenize and clean
tokens = re.findall(r"[A-Za-z]+", text.lower())
stop_words = {"the", "is", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "are"}
clean_words = [w for w in tokens if w not in stop_words and len(w) > 2]

freq = Counter(clean_words)
print("Total words:", len(tokens))
print("Words after cleaning:", len(clean_words))
print("\nTop 10 frequent words:")
print(pd.DataFrame(freq.most_common(10), columns=["Word", "Count"]))

wc = WordCloud(width=900, height=450, background_color="white", colormap="viridis")
wc = wc.generate(" ".join(clean_words))

plt.figure(figsize=(12, 5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud")
plt.show()
```

**Expected Output:**
- Total and cleaned word counts.
- Top frequent words table.
- Word cloud image where frequent words appear larger.

**Viva Questions:**
1. Why remove stopwords? - They are common words with low semantic value for analysis.
2. What is tokenization? - Splitting text into smaller units (tokens/words).
3. How does word cloud represent frequency? - Higher frequency words appear with larger font size.

**Key Concepts:** NLP preprocessing, tokenization, stopword removal, frequency analysis, word cloud visualization.

---

## Quick Exam Tips (Handwriting + Viva)
1. Write imports first, then data loading, then model/training, then evaluation.
2. For viva, always explain: input data, preprocessing, algorithm choice, metrics used.
3. If output varies (random split), mention `random_state` ensures reproducibility.
4. For clustering/association/text tasks, explain interpretation, not just syntax.
