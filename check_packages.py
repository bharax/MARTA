# ---
# Function
# ---
def print_line():
    print("-----------------------------------------------------")

# ---
# Torch packages used for the definition of tensors, modules and optimizers
# ---
print_line()
print("Loading Torch ...")
import torch
import torch.nn as nn
import torch.optim as optim
print("Torch version:", torch.__version__)

# ---
# Fundamental package for scientific computing with Python
# ---
print_line()
print("Loading numpy ...")
import sys
import numpy as np
print("Python version:", sys.version)
print("NumPy version:", np.__version__)

# ---
# Figures and graphics with Python
# ---
print_line()
print("Loading matplotlib ...")
import time
import matplotlib.pyplot as plt
import matplotlib
print("Matplotlib version:", matplotlib.__version__)

# ---
# Database from sklearn
# ---
print_line()
print("Loading sklearn tools ...")
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# ---
# Check the forward pass through a linear layer
# ---
batch = torch.rand(32,10)
lin0 = nn.Linear(10,1)
print_line()
print("Forward pass:", lin0(batch))
print_line()
print("All checks have been successfully completed")
print_line()
