import server_utlis
import data_creation
import data_utils
import synthetic_data
import proxy_classifiers
import evaluate_classifers
import numpy as np
from data_creation import adult_data  
from data_utils import preprocess_adult_dataset
from synthetic_data import GaussianCopulaSynthesizer, generate_synthetic_data, train_generative_model
from proxy_classifiers import train_proxy_classifier, majority_vote, generate_A_hat_full
from evaluate_classifers import evaluate_fairness

def main():
    output_dir = "data"  # Directory to save train & test data

    #Step1 : Create the adult dataset
    train_path, test_path = adult_data(output_dir) 
    print(f"Train data saved at: {train_path}")
    print(f"Test data saved at: {test_path}")

    # Step 2 : Preprocess the adult dataset
    x_train_df, x_test_df, z_train, z_test, y_train, y_test = preprocess_adult_dataset(train_path, test_path)
    print("Preprocessing complete!")

    # Step 3 : Train generative model
    X, true_A = generate_synthetic_data(x_train_df, z_train, sensitive_attr='sex', output_dir=output_dir)
    print("Synthetic data generation complete.")
    print("Shape of synthetic data: ", X.shape)
    print("Shape of true sensitive attribute: ", true_A.shape)
    # Step 4 : Train the classifier
    models, A_hat_train, A_hat_test = train_proxy_classifier(X, true_A, k=3)
    A_hat_train_f, A_hat_test_f = generate_A_hat_full(models, x_train_df, x_test_df)
                        
    # Step 5 : Evaluate the classifier
    evaluate_fairness(x_train_df, x_test_df, z_train, z_test, y_train, y_test, A_hat_train_f, A_hat_test_f)

if __name__ == "__main__":
    main()