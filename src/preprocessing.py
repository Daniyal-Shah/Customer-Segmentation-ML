import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1. Remove unnecessary columns
    df = df.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue', 'Response'])

    # 2. Feature engineering with dates
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
    df['Customer_Since'] = (pd.to_datetime('today') - df['Dt_Customer']).dt.days

    df['Year_Birth'] = pd.to_numeric(df['Year_Birth'], errors='coerce')
    df['Age'] = pd.Timestamp.today().year - df['Year_Birth']

    df = df.drop(columns=['Dt_Customer', 'Year_Birth'])

    # 3. Handle missing values
    missing_value_columns = df.isnull().sum().sort_values(ascending=False)

    for column, null_count in missing_value_columns.items():
        if null_count > 0:
            if column in ['Income', 'Kidhome', 'Teenhome']:
                df[column] = df[column].fillna(df[column].median())
            else:
                df[column] = df[column].fillna(df[column].mode()[0])

    return df


def build_preprocessor(df: pd.DataFrame):
    # 4. Identify numeric and categorical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # 5. Pipelines
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 6. Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )
    
    # 7. Return preprocessor
    return preprocessor