import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

# ============================
# Load and Prepare Dataset
# ============================

# Load dataset
df = pd.read_csv(r"E:\Paper\Fall Detection\augmented_images\Final_CSV.csv")

# Select only numeric columns to handle missing values
numeric_cols = df.select_dtypes(include=['number']).columns

# Fill missing values in numeric columns with the mean
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Drop non-numeric and label columns
drop_columns = ['LABEL', 'Filename']
X = df.drop(drop_columns, axis=1)
y = df['LABEL']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=150, stratify=y)

# ============================
# Handle Zero-Variance Features
# ============================

# Identify and remove zero-variance columns (those with only one unique value)
zero_variance_cols = X_train.columns[X_train.nunique() <= 1]
if len(zero_variance_cols) > 0:
    print("Removing zero-variance columns:", zero_variance_cols.tolist())
    X_train = X_train.drop(columns=zero_variance_cols)
    X_test = X_test.drop(columns=zero_variance_cols)

# ============================
# Normalize Data (Standardization)
# ============================

# Standardize the data (important for some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================
# Initialize and Train Models
# ============================

# Initialize classifiers
cb = CatBoostClassifier(auto_class_weights='Balanced', verbose=0)
xgb = XGBClassifier(scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]))
rf = RandomForestClassifier(n_estimators=300, random_state=150, class_weight='balanced')

models = [
    ("XGBoost", xgb),
    ("Random Forest", rf),
    ("CatBoost", cb),
]

# Dictionary to store accuracy scores
acc_score = {name: 0 for name, model in models}

# Train and evaluate each model
for name, model in models:
    model.fit(X_train_scaled, y_train)  # Train model
    y_pred = model.predict(X_test_scaled)  # Predict on test set
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    acc_score[name] = accuracy  # Store accuracy
    print(f'Accuracy for {name}: {accuracy * 100:.2f}%')  # Print accuracy
    print(classification_report(y_test, y_pred, zero_division=1))  # Print classification report
