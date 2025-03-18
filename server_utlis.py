import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'  # Set before importing torch

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# All packages imported
import pandas as pd
import numpy as np
from aif360.sklearn.datasets import fetch_adult
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from scipy.interpolate import make_interp_spline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
from sdv.metadata import SingleTableMetadata
from copulas.multivariate import GaussianMultivariate
from sdv.single_table.base import BaseSingleTableSynthesizer
from scipy.stats import mode
from sdv.single_table import CTGANSynthesizer
from sklearn.compose import make_column_selector as selector
