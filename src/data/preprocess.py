import pandas as pd
from sklearn.preprocessing import StandardScaler


# 1. Drop unnecessary columns
def clean_data(df):
    df = df.drop(columns=['id', 'City', 'Profession', 'Degree'])
    return df


# 2. Handle missing values
def handle_missing(df):
    # Financial Stress → fill with median
    df['Financial Stress'] = df['Financial Stress'].fillna(df['Financial Stress'].median())
    return df


# 3. Encode categorical features
def encode_features(df):

    # -------------------------
    # Binary Encoding
    # -------------------------
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})

    df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})


    # -------------------------
    # Ordinal Encoding
    # -------------------------

    # Clean text (important)
    df['Sleep Duration'] = df['Sleep Duration'].str.strip()

    sleep_map = {
        "Less than 5 hours": 0,
        "5-6 hours": 1,
        "7-8 hours": 2,
        "More than 8 hours": 3
    }

    df['Sleep Duration'] = df['Sleep Duration'].map(sleep_map)

    # Handle unmapped values safely
    df['Sleep Duration'] = df['Sleep Duration'].fillna(df['Sleep Duration'].median())


    # -------------------------
    # One-Hot Encoding (Nominal)
    # -------------------------
    df = pd.get_dummies(df, columns=['Dietary Habits'], drop_first=True)


    # -------------------------
    # Convert boolean → int
    # -------------------------
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    return df


# 4. Scale numerical features
def scale_features(df):
    scaler = StandardScaler()

    numerical_cols = ['Age', 'CGPA', 'Work/Study Hours']

    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df


# 5. Full preprocessing pipeline
def preprocess_data(df):

    df = clean_data(df)
    df = handle_missing(df)
    df = encode_features(df)
    df = scale_features(df)

    return df
