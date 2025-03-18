from aif360.sklearn.datasets import fetch_adult
from sklearn.model_selection import train_test_split
import os

def adult_data(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)

    adult = fetch_adult()
    df = adult.X.copy()
    df['annual_income'] = adult.y.values
    df['g_income'] = df['sex'].astype(str) + "_" + df['annual_income'].astype(str)
    train_data, test_data = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['g_income']
    )
    train_data.drop(columns=['g_income'], inplace=True)
    test_data.drop(columns=['g_income'], inplace=True)

    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    for file_path in [train_path, test_path]:
        if os.path.exists(file_path):
            os.remove(file_path)

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"Train data saved to {train_path}")
    print(f"Test data saved to {test_path}")

    print(f"Train Data Shape: {train_data.shape}")
    print(f"Test Data Shape: {test_data.shape}")

    return train_path, test_path


