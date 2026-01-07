# ===========================================
# 1. Import Libraries
# ===========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# ===========================================
# 2. Load Dataset
# ===========================================
df = pd.read_csv("Hotel Reservations.csv")
print(df.head())

# ===========================================
# 3. EDA - Basic Info
# ===========================================
print("\n--- Shape of Data ---")
print(df.shape)

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Data Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

# Histogram for all numeric columns
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# ===========================================
# 4. Visualization - Target Distribution
# ===========================================
sns.countplot(x=df["booking_status"])
plt.title("Booking Status Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ===========================================
# 5. Skewness of Numerical Columns
# ===========================================
print("\n--- Skewness of Numerical Columns ---")
print(df.skew(numeric_only=True))

df.skew(numeric_only=True).plot(kind='bar', figsize=(8,4))
plt.title("Skewness of Numerical Features")
plt.show()

df['previous_cancellations'] = np.log1p(df['no_of_previous_cancellations'])
df['previous_bookings_not_canceled'] = np.log1p(df['no_of_previous_bookings_not_canceled'])
df['total_of_special_requests'] = np.log1p(df['no_of_special_requests'])

df['lead_time'] = np.sqrt(df['lead_time'])

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
# ===========================================
# 6. UNIVARIATE ANALYSIS (Histograms and Boxplots)
for col in numerical_cols:
    print(f"\nAnalyzing Column: {col}")

    plt.figure(figsize=(12, 4))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f"Histogram of {col}")

    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")

    plt.tight_layout()
    plt.show()

# ===========================================
# 7. Outlier Detection (IQR Method)
# ===========================================
def find_outliers_IQR(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return column[(column < lower) | (column > upper)]

print("\n--- Outliers per Column ---")
for col in numerical_cols:
    outliers = find_outliers_IQR(df[col])
    print(f"{col}: {len(outliers)} outliers")

# Removing outliers
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    df = df[(df[col] >= lower) & (df[col] <= upper)]



# ===========================================
# 8. Encoding Categorical Variables
# ===========================================
le = LabelEncoder()
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("\n--- Data After Encoding ---")
print(df.head())

# ===========================================
# 9. Train / Test Split
# ===========================================
X = df.drop("booking_status", axis=1)
y = df["booking_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===========================================
# 10. Feature Scaling
# ===========================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===========================================
# 11. Train ML Models
# ===========================================

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

# ===========================================
# 12. Evaluation (Separate for Each Model)
# ===========================================

# ----- Decision Tree -----
print("\n===== Decision Tree Evaluation =====")
print("Accuracy:", accuracy_score(y_test, pred_dt))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred_dt))
print("\nClassification Report:")
print(classification_report(y_test, pred_dt))

# ----- Random Forest -----
print("\n===== Random Forest Evaluation =====")
print("Accuracy:", accuracy_score(y_test, pred_rf))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, pred_rf))

# ----- Logistic Regression -----
print("\n===== Logistic Regression Evaluation =====")
print("Accuracy:", accuracy_score(y_test, pred_lr))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred_lr))
print("\nClassification Report:")
print(classification_report(y_test, pred_lr))

# ===========================================
# 13. Confusion Matrix Heatmaps
# ===========================================

# Decision Tree Heatmap
plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, pred_dt), annot=True, fmt='d', cmap='Blues')
plt.title("Decision Tree - Confusion Matrix")
plt.show()

# Random Forest Heatmap
plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, pred_rf), annot=True, fmt='d', cmap='Greens')
plt.title("Random Forest - Confusion Matrix")
plt.show()

# Logistic Regression Heatmap
plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, pred_lr), annot=True, fmt='d', cmap='Oranges')
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

# =============================================
# Hyperparameter Tuning - Decision Tree
# =============================================
dt_params = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10]
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(),
    dt_params,
    cv=5,
    scoring="accuracy"
)

dt_grid.fit(X_train, y_train)

print("\nBest Decision Tree Parameters:", dt_grid.best_params_)
print("Best Decision Tree Score:", dt_grid.best_score_)

# =============================================
# Hyperparameter Tuning - Random Forest
# =============================================
rf_params = {
    "n_estimators": [50, 100, 200],
    "criterion": ["gini", "entropy"],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
}

rf_grid = GridSearchCV(
    RandomForestClassifier(),
    rf_params,
    cv=5,
    scoring="accuracy",
    n_jobs=1
)

rf_grid.fit(X_train, y_train)

print("\nBest Random Forest Parameters:", rf_grid.best_params_)
print("Best Random Forest Score:", rf_grid.best_score_)

# ===========================================
# 14. Feature Importance (Decision Tree)
# ===========================================
importances_dt = dt.feature_importances_
plt.figure(figsize=(8,4))
plt.barh(X.columns, importances_dt)
plt.title("Decision Tree - Feature Importance")
plt.show()

# ===========================================
# 15. Feature Importance (Random Forest)
# ===========================================
importances_rf = rf.feature_importances_
plt.figure(figsize=(8,4))
plt.barh(X.columns, importances_rf)
plt.title("Random Forest - Feature Importance")
plt.show()