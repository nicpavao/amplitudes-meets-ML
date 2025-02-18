# This is my first transformer project
# The idea is to create a transformer that can take as imput a list (N mod p1, N mod p2, ..., N mod pn)
# where N is a number (integer, rational, etc) and p1,p2,...,pn are prime numbers
# and returns the number N

# Let us start by importing the necessary libraries
import torch  # Main framework for defining and training the transformer
import torch.nn as nn  # Neural network module
import torch.optim as optim  # Optimization functions
import numpy as np  # For numerical operations
import random  # For generating random numbers
import itertools  # (Optional) For generating structured datasets

import matplotlib.pyplot as plt  # (Optional) For visualization
from torch.utils.data import Dataset, DataLoader  # To handle training data efficiently

#---------------------------------------------------



# Now we move onto data generation
# the idea is to first define a list of primes (p1,p2,...,pn)
# then we pick integers N and compute the remainders of N when divided by each prime
# then the pairs (N mod p1, N mod p2, ..., N mod pn) are stored as input and N is stored as output
# this will form our training dataset

# Define a list of small primes
primes = [2, 3, 5, 7, 11]

# Compute the product P = p1 * p2 * ... * pn
P = np.prod(primes)

# Check what we have done so far
print(f"Chosen primes: {primes}")
print(f"Product of primes (P): {P}")


#---------------------------------------------------

# Now we define a PyTorch Dataset to handle our data
# This creates a class that inherits from the `Dataset` class in PyTorch
# so that we can use PyTorch's `DataLoader` to load the data efficiently

# Define a PyTorch Dataset for our data
class ModuloDataset(Dataset):
    def __init__(self, num_samples=1000, primes=None):
        super(ModuloDataset, self).__init__() # Initialize the base class 'Dataset', not strictly required here but a good practice
        self.primes = primes if primes else [2, 3, 5, 7, 11]
        self.P = np.prod(self.primes)
        self.samples = []

        for _ in range(num_samples):
            N = random.randint(0, self.P - 1)  # Pick a random integer N
            remainders = [N % p for p in self.primes]  # Compute remainders
            self.samples.append((torch.tensor(remainders, dtype=torch.float32), 
                                 torch.tensor(N, dtype=torch.float32)))  # Convert to tensors

    def __len__(self):
        return len(self.samples)  # Return the total number of samples

    def __getitem__(self, idx):
        return self.samples[idx]  # Return the (input, output) pair at index `idx`

# Create the dataset
dataset = ModuloDataset(num_samples=1000)

# Check some samples
for i in range(5):
    print(f"Sample {i}: Input (moduli) {dataset[i][0]}, Output (N) {dataset[i][1]}")



#---------------------------------------------------
# Now we define the DataLoaders to load the data efficiently during training
# This will allow us to load the data in batches, shuffle it, etc.

# Create a DataLoader to load the dataset in batches
batch_size = 32  # Number of samples per batch

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Check one batch of data
for batch in dataloader:
    inputs, targets = batch  # Unpack batch
    print(f"Batch Inputs (moduli): {inputs.shape}")  # Should be (batch_size, num_primes)
    print(f"Batch Targets (N values): {targets.shape}")  # Should be (batch_size,)
    break  # Only print the first batch



#---------------------------------------------------
# Now we define the transformer model
# We use the nn.Module class in PyTorch to define our model
# The transformer model consists of an input embedding layer, followed by a transformer encoder
# and finally a linear layer to output the predicted value
# Note: here we need to use super() or else some necessary initializations from nn.Module will be missed
class ModuloTransformer(nn.Module):
    def __init__(self, num_primes, d_model=128, num_heads=4, num_layers=2, hidden_dim=256):
        super(ModuloTransformer, self).__init__()

        self.embedding = nn.Linear(num_primes, d_model)  # Input embedding
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, 1)  # Final output layer

    def forward(self, x):
        x = self.embedding(x)  # Project input to d_model dimension
        x = self.transformer_encoder(x)  # Pass through transformer layers
        x = self.fc_out(x).squeeze(-1)  # Final output (scalar prediction)
        return x
# define model
model = ModuloTransformer(num_primes=len(primes), d_model=128, num_heads=4, num_layers=2, hidden_dim=256)