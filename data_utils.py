import server_utlis
import data_creation
import pandas as pd
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

def preprocess_adult_dataset(train_path: str, test_path: str):
    dataset_train = pd.read_csv(train_path)
    dataset_test = pd.read_csv(test_path)
    
    sensitive = 'sex'
    label = 'annual_income'
    drop_attrs = ['annual_income', 'sex'] # Not dropping 'race'
    
    label_replacement = {'<=50K': 0, '>50K': 1}
    sensitive_replacement = {'Female': 0, 'Male': 1}
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]
    favorable_label = 1
    unfavorable_label = 0

    y_train = dataset_train[label].replace(label_replacement).infer_objects(copy=False)
    y_test = dataset_test[label].replace(label_replacement).infer_objects(copy=False)
    z_train = dataset_train[sensitive].replace(sensitive_replacement).infer_objects(copy=False)
    z_test = dataset_test[sensitive].replace(sensitive_replacement).infer_objects(copy=False)
    x_train = dataset_train.drop(columns=drop_attrs)
    x_test = dataset_test.drop(columns=drop_attrs)

    numerical_columns = selector(dtype_exclude=object)(x_train)
    categorical_columns = selector(dtype_include=object)(x_train)

    preprocessor = ColumnTransformer([
        ('one-hot-encoder', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('standard-scaler', StandardScaler(), numerical_columns)
    ]).fit(x_train)

    x_train_processed = preprocessor.transform(x_train)
    x_test_processed = preprocessor.transform(x_test)
    
    # Convert to DataFrame
    x_train_df = pd.DataFrame(x_train_processed.toarray() if hasattr(x_train_processed, 'toarray') else x_train_processed)
    x_test_df = pd.DataFrame(x_test_processed.toarray() if hasattr(x_test_processed, 'toarray') else x_test_processed)
    
    z_train = z_train.reindex(x_train_df.index)
    z_test = z_test.reindex(x_test_df.index)

    print(f"Train Data Shape: {x_train_df.shape}")
    print(f"Test Data Shape: {x_test_df.shape}")
    print(f"Train Data Labels Shape: {y_train.shape}")
    print(f"Test Data Labels Shape: {y_test.shape}")
    
    return x_train_df, x_test_df, z_train, z_test, y_train, y_test