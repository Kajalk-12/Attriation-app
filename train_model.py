import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("employee_attrition.csv")

# Define categorical and numerical features
CATEGORICAL_FEATURES = [
    'JobSatisfaction', 'WorkLifeBalance', 'PerformanceRating', 'OverTime',
    'RelationshipSatisfaction', 'CareerGrowthOpportunity', 'StockOptionLevel', 'JobLevel'
]

NUMERIC_FEATURES = [
    'Age', 'MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears',
    'WorkHours', 'DistanceFromHome', 'TrainingHoursLastYear'
]

TARGET = 'Attrition'

# Drop rows with missing values
df = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET]].dropna()

# Encode categorical features
label_encoders = {}
for feature in CATEGORICAL_FEATURES + [TARGET]:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature].astype(str))
    label_encoders[feature] = le

# Prepare features and target
X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
y = df[TARGET]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "attrition_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

# Print LabelEncoder classes for debugging
print("\n🔍 LabelEncoder classes used during training:")
for feature, le in label_encoders.items():
    print(f"{feature}: {list(le.classes_)}")

print("\n✅ Model and files saved: attrition_model.pkl, label_encoders.pkl, feature_names.pkl")
