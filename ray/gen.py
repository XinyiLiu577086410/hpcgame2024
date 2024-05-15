import numpy as np
import os

# Create directories if they don't exist
os.makedirs('inputs', exist_ok=True)
os.makedirs('weights', exist_ok=True)

# Generate 100 (200, 40000) matrices and save them
for i in range(100):
    matrix = np.random.rand(200, 200)
    np.save(f'inputs/input_{i}.npy', matrix)

# # Generate 4 (40000, 40000) matrices and save them
for i in range(4):
    matrix = np.random.rand(200, 200)
    np.save(f'weights/weight_{i}.npy', matrix)