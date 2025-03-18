import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds


def prepare_data(X, A):
    """Concatenate features and sensitive attributes."""
    return np.column_stack([X, A])


def train_mlp(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu',
                          solver='adam', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_fair_classifier(X_train, A_train, y_train):
    """Train a fair classifier using Exponentiated Gradient with Equalized Odds."""
    base_model = LogisticRegression(solver='liblinear', random_state=42)
    expgrad = ExponentiatedGradient(base_model, constraints=EqualizedOdds(), max_iter=500)
    expgrad.fit(X_train, y_train, sensitive_features=A_train)
    return expgrad


def evaluate_fairness(X_train, X_test, true_A_train, true_A_test,
                      y_train, y_test, A_hat_train, A_hat_test):
    # Safety Checks
    assert len(X_train) == len(true_A_train) == len(A_hat_train) == len(y_train), "Train set length mismatch"
    assert len(X_test) == len(true_A_test) == len(A_hat_test) == len(y_test), "Test set length mismatch"

    # Prepare Data
    X_A_train = prepare_data(X_train, true_A_train)
    X_A_test = prepare_data(X_test, true_A_test)

    X_A_hat_train = prepare_data(X_train, A_hat_train)
    X_A_hat_test = prepare_data(X_test, A_hat_test)

    results = {}

    # ---- MLP Models (Without fairness constraints) ----
    for label, X_tr, X_te, sensitive_test in [
        ("MLP (X + A)", X_A_train, X_A_test, true_A_test),
        ("MLP (X + A_hat)", X_A_hat_train, X_A_hat_test, A_hat_test),
    ]:
        model = train_mlp(X_tr, y_train)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_test, y_pred)
        fairness = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_test)
        results[label] = (acc, fairness)

    # ---- Fair Classifiers (Exponentiated Gradient) ----
    for label, X_tr, X_te, sensitive_test, sensitive_train in [
        ("Fair Classifier (X + A)", X_A_train, X_A_test, true_A_test, true_A_train),
        ("Fair Classifier (X + A_hat)", X_A_hat_train, X_A_hat_test, A_hat_test, A_hat_train),
    ]:
        model = train_fair_classifier(X_tr, sensitive_train, y_train)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_test, y_pred)
        fairness = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_test)
        results[label] = (acc, fairness)

    # ---- Print Final Results ----
    print("\n🔹 Evaluation Results 🔹")
    for label, (acc, fairness) in results.items():
        print(f"{label} → Accuracy: {acc:.4f} | Fairness (EOD): {fairness:.4f}")
