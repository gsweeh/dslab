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



```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load Dataset
# -----------------------------
df = sns.load_dataset("mpg").copy()

# -----------------------------
# (a) Dimension, Structure, Summary
# -----------------------------
print("Shape (Rows, Columns):", df.shape)
print("\nInfo:")
df.info()
print("\nSummary Statistics:")
print(df.describe(numeric_only=True))
print("\nNull Values per Column:")
print(df.isna().sum())

# -----------------------------
# (b) Preprocessing
# -----------------------------
# Convert horsepower to numeric (handle '?' or text)
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")

# Fill missing horsepower with median
df["horsepower"].fillna(df["horsepower"].median(), inplace=True)

# Drop remaining null rows
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print("\nAfter Cleaning, Shape:", df.shape)

# -----------------------------
# (c) Histogram (at least two)
# -----------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.histplot(df["mpg"], kde=True)
plt.title("MPG Distribution")

plt.subplot(1,2,2)
sns.histplot(df["horsepower"], kde=True)
plt.title("Horsepower Distribution")

plt.tight_layout()
plt.show()

# -----------------------------
# (d) Violin Plot
# -----------------------------
plt.figure(figsize=(6,4))
sns.violinplot(y=df["mpg"])
plt.title("Violin Plot of MPG")
plt.show()

# -----------------------------
# (e) Box Plot (Before Outlier Treatment)
# -----------------------------
plt.figure(figsize=(6,4))
sns.boxplot(y=df["horsepower"])
plt.title("Horsepower - Before Outlier Treatment")
plt.show()

# Outlier Treatment using IQR method
Q1 = df["horsepower"].quantile(0.25)
Q3 = df["horsepower"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_no_out = df[(df["horsepower"] >= lower) & (df["horsepower"] <= upper)]

# Box Plot After Treatment
plt.figure(figsize=(6,4))
sns.boxplot(y=df_no_out["horsepower"])
plt.title("Horsepower - After Outlier Treatment")
plt.show()

# -----------------------------
# (f) Heatmap (Correlation)
# -----------------------------
plt.figure(figsize=(8,6))
corr = df_no_out.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# (g) Standardization
# -----------------------------
scaler = StandardScaler()

continuous_cols = ["mpg", "horsepower", "weight", "acceleration"]

df_no_out[continuous_cols] = scaler.fit_transform(df_no_out[continuous_cols])

print("\nStandardized Data (First 5 Rows):")
print(df_no_out[continuous_cols].head())
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
# Program 3 – Exploratory Data Analysis

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset("mpg")

# -----------------------------
# Step 1: Identify categorical and numerical variables
# -----------------------------
cats = list(df.select_dtypes(include=['object']).columns)
nums = list(df.select_dtypes(exclude=['object']).columns)

print("Categorical Variables:", cats)
print("Numerical Variables:", nums)

# -----------------------------
# (a) Histogram for continuous variables
# -----------------------------
plt.figure(figsize=(6,4))
plt.hist(df["mpg"], bins=20, color="skyblue")
plt.title("Histogram of MPG")
plt.xlabel("mpg")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df["weight"], bins=20, color="lightgreen")
plt.title("Histogram of Weight")
plt.xlabel("weight")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# (b) Scatter plot (relationship between two continuous variables)
# -----------------------------
plt.figure(figsize=(6,4))
plt.scatter(df["horsepower"], df["mpg"])
plt.title("Horsepower vs MPG")
plt.xlabel("horsepower")
plt.ylabel("mpg")
plt.show()

# -----------------------------
# (c) Count plot (categorical frequency)
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="origin", data=df)
plt.title("Count Plot of Origin")
plt.show()

# -----------------------------
# (d) Point plot (one categorical + one continuous)
# -----------------------------
plt.figure(figsize=(6,4))
sns.pointplot(x="origin", y="mpg", data=df)
plt.title("Average MPG by Origin")
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
# Program 4 – KNN (Breast Cancer)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN (try k=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import plot_tree

df = pd.read_csv("./student_performance_new.csv")
df.head()
df.columns
df.isna().sum()

X = df[["Test Result ","Quiz Result ","Assignment Result "]]
y = df.Result

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
clf = DecisionTreeClassifier(criterion='gini',splitter='random',random_state=42,max_depth=5)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))

plot_tree(clf)

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
#Naive Baye's Classifictaion algorithm
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

dataset = pd.read_csv("C:/Users/user1/Downloads/Placement_Data.csv")

print(dataset.head())

dataset.info()

dataset = dataset.drop(["sl_no"], axis = 1)
dataset = dataset.drop(["salary"], axis = 1)
dataset = dataset.drop(["gender"], axis =1)

#Placed and Not placed dataframes
Placed = dataset[dataset.status == "Placed"]
NPlaced = dataset[dataset.status == "Not Placed"]

print(Placed)

print(NPlaced)

X=dataset[['ssc_p','hsc_p','degree_p','etest_p', 'mba_p']]
y = dataset.status.values


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
#dataset['gender'].map({'M': 1, 'F': 0})

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

nb = GaussianNB()
nb.fit(x_train, y_train)

print("Naive Bayes score: ",nb.score(x_test, y_test)*100)

predictions = nb.predict(x_test)
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy Score:", accuracy)

precision = precision_score(y_test, predictions)
print("Precision Score:", precision)

recall = recall_score(y_test, predictions)
print("Recall Score:", recall)

f1 = f1_score(y_test, predictions)
print("F1 Score:", f1)

clf = MultinomialNB()
clf.fit(x_train, y_train)

print("Naive Bayes score with Multionmial NB classfier: ",clf.score(x_test, y_test)*100)

bnb = BernoulliNB(binarize=0.0)
model = bnb.fit(x_train, y_train)
y_pred = bnb.predict(x_test)
print(classification_report(y_test, y_pred))
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
from mlxtend.frequent_patterns import apriori,association_rules

df = pd.read_csv("basket.csv")

df.describe()
df.info()
df.head()
df.isnull().sum()

df.fillna('',inplace=True)

df_dum = pd.get_dummies(df)
df_dum.head()

frequent_items = apriori(df_dum,min_support=0.01,
                         use_colnames=True)
frequent_items


rules = association_rules(frequent_items,metric='confidence',
                          min_threshold=0.01)
rules = rules.sort_values(['support','confidence'],ascending=[False,False])


rules
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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv("./Mall_Customers .csv")

data.head()
data.shape
data.describe()
data.info()

# Encode 'Gender' column
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data["Gender"])

# Selecting features for clustering
X = data[['Gender','Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Using the Elbow Method to determine the optimal number of clusters
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, n_init=10, random_state=42)
    kmeans.fit(X_scaled)  # Use scaled data
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='-')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Choosing k=5 as the optimal number of clusters
k = 5
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)  # Assign clusters to data

# Plotting the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['Annual Income (k$)'], y=data['Spending Score (1-100)'], hue=data['Cluster'], palette='viridis', s=100)
plt.title('K-Means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

# Choosing k=5 as the optimal number of clusters
k = 5
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)  # Assign clusters to data

# Plotting the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['Annual Income (k$)'], y=data['Spending Score (1-100)'], hue=data['Cluster'], palette='viridis', s=100)
# Plot the centroids on the original scale
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], s=300, c='red', marker='x', label='Centroids')

plt.title('K-Means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

kmeans.cluster_centers_

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
