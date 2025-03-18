import os
import server_utlis
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
from sdv.metadata import SingleTableMetadata
from copulas.multivariate import GaussianMultivariate

# class GaussianCopulaSynthesizer:
#     def __init__(self, metadata):
#         super().__init__(metadata)
#         self.model = GaussianMultivariate()

#     def fit(self, data):
#         self.model.fit(data)
#         print("Gaussian Copula model fitted.")

#     def generate_samples(self, num_samples):
#         synthetic_data = self.model.sample(num_samples)
#         print(f"Generated {num_samples} synthetic samples.")
#         return synthetic_data
    
def train_generative_model(train_data, sensitive_data, sensitive_attr):
    X_A = train_data.copy()
    X_A[sensitive_attr] = sensitive_data

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(X_A)

    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(X_A)

    print("Generative model trained.")
    return synthesizer

def generate_synthetic_data(train_data, sensitive_data, sensitive_attr='sex', output_dir='data'):
    synthesizer = train_generative_model(train_data, sensitive_data, sensitive_attr)

    # num_synthetic_samples = int(num_synthetic_samples_ratio * len(train_data))
    num_synthetic_samples = len(train_data)
    synthetic_data = synthesizer.sample(num_synthetic_samples)

    synthetic_true_A = synthetic_data[[sensitive_attr]].copy()
    synthetic_X = synthetic_data.drop(columns=[sensitive_attr]).copy()

    features_path = os.path.join(output_dir, 'synthetic_X.csv')
    sensitive_path = os.path.join(output_dir, 'synthetic_true_A.csv')

    synthetic_X.to_csv(features_path, index=False)
    synthetic_true_A.to_csv(sensitive_path, index=False)

    print(f"Synthetic data saved in '{output_dir}' directory.")

    return synthetic_X, synthetic_true_A