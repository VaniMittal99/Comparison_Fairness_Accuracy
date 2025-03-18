from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import numpy as np

def majority_vote (A_preds):
    A_hat = mode(A_preds, axis=1, keepdims=True).mode.ravel()
    print("Generated A-hat (Noisy A):", A_hat)
    print("Unique values in A_hat:", np.unique(A_hat))
    print("Shape of A_hat:", A_hat.shape)
    return A_hat

def train_proxy_classifier(X, A, k=3, hidden_layer_sizes=(32, 16), max_iter=1000, test_size=0.2, random_state=42):
    
    A = A.values.ravel()
    A_preds_train, A_preds_test = [], []

    models = []

    for i in range(k):
        print(f"Training Proxy Model {i+1}/{k}...")

        X_train, X_test, A_train, A_test = train_test_split(X, A, test_size=test_size, random_state=random_state+i)
        print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, A_train: {A_train.shape}, A_test: {A_test.shape}")
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='relu', solver='adam', max_iter=max_iter, random_state=random_state+i)
        model.fit(X_train, A_train)
        A_preds_train.append(model.predict(X_train))
        A_preds_test.append(model.predict(X_test))
        models.append(model)
        acc = accuracy_score(A_test, A_preds_test[-1])

        # print(f"Proxy Model {i+1} Accuracy: {acc:.4f}")
        # print(f"Unique values in A_pred (Model {i+1}): {np.unique(A_preds_test[:, i])}")
    A_preds_train = np.column_stack(A_preds_train) 
    A_preds_test = np.column_stack(A_preds_test)

    print(f" Shape of A_preds_train: {A_preds_train.shape}")
    print(f" Shape of A_preds_test: {A_preds_test.shape}")

    A_hat_train = majority_vote(A_preds_train)
    A_hat_test = majority_vote(A_preds_test)

    print(f"Shape of A_hat_train (should match X_train): {A_hat_train.shape}")
    print(f"Shape of A_hat_test (should match X_test): {A_hat_test.shape}")

    return models, A_hat_train, A_hat_test

def generate_A_hat_full(models, x_train_df, x_test_df):
    
    A_preds_train_full, A_preds_test_full = [], []

    for model in models:
        A_preds_train_full.append(model.predict(x_train_df))
        A_preds_test_full.append(model.predict(x_test_df))

    A_preds_train_full = np.column_stack(A_preds_train_full)
    A_preds_test_full = np.column_stack(A_preds_test_full)

    A_hat_train_full = majority_vote(A_preds_train_full)
    A_hat_test_full = majority_vote(A_preds_test_full)

    print(f"Final Shape of A_hat_train (should match x_train_df): {A_hat_train_full.shape}")
    print(f"Final Shape of A_hat_test (should match x_test_df): {A_hat_test_full.shape}")

    return A_hat_train_full, A_hat_test_full

