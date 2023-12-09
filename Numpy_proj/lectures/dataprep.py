import os

import numpy as np
import matplotlib.pyplot as plt

# Noisy Data - identify outliers based on distance between points on plots
# Data transformation
# #Decimal Scaling moves the decimal point of numbers to normalize them.
# Min-max Method scales the numbers such that they lie between a specified range (in this case, 0 and 1).
# Z-score Normalization scales the numbers based on the mean and standard deviation, making the mean of the normalized data 0 and the standard deviation 1. decimal scaling

scores_1 = np.array([24, 23, 30, 29, 17, 16, 15])
# 1. Decimal Scaling
j = np.ceil(np.log10(np.max(np.abs(scores_1))))
decimal_scaled_scores = scores_1 / (10 ** j)

# 2. Min-max Method
min_score = np.min(scores_1)
max_score = np.max(scores_1)
min_max_scaled_scores = (scores_1 - min_score) / (max_score - min_score)

# 3. Z-score Normalization (Z-index method)
mean_score = np.mean(scores_1)
std_score = np.std(scores_1)
z_index_scaled_scores = (scores_1 - mean_score) / std_score

# print(decimal_scaled_scores, min_max_scaled_scores, z_index_scaled_scores)

# Data reduction - datasets can be sampled 2 ways (simple & stratified)
# record reduction by SAMPLING
# attribute reduction by SELECTION or PROJECTION
# values reduction by DISCRETIZATION or AGGREGATION

# Principal component analysis. Аims to find the directions (components) in the dataset that maximize the variance. Once these directions are found, they can be used to project the original dataset into a reduced-dimensional space.
# Standardization: Typically, the data is first standardized so that each feature has a mean of 0 and a standard deviation of 1.
# Covariance Matrix Computation: Compute the covariance matrix of the standardized data.
# Eigen Decomposition: Decompose the covariance matrix into its eigenvectors and eigenvalues.
# Sort Eigen Pairs: Sort the eigenvectors by the eigenvalues in decreasing order.
# Projection: Select the top k eigenvectors (where k is the number of dimensions you want in your reduced data), and use this eigenvector matrix to transform the original data to the reduced dimension.
# Generate a sample dataset with two correlated features
# np.random.seed(42)
# X = 2 * np.random.rand(100, 1)
# Y = 4 + 3 * X + np.random.rand(100, 1)
#
# data = np.hstack((X, Y))
#
# # Standardize the data
# mean = np.mean(data, axis=0)
# std_dev = np.std(data, axis=0)
# data_standardized = (data - mean) / std_dev
#
# # Compute the covariance matrix
# cov_matrix = np.cov(data_standardized, rowvar=False)
#
# # Eigen decomposition
# eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
#
# # Sort eigenvectors by eigenvalues in descending order
# sorted_index = np.argsort(eigenvalues)[::-1]
# sorted_eigenvectors = eigenvectors[:, sorted_index]
#
# # Project data onto the first principal component
# pc1 = data_standardized.dot(sorted_eigenvectors[:, :1])
#
# # Visualization
# plt.figure(figsize=(12, 6))
#
# plt.subplot(1, 2, 1)
# plt.scatter(data_standardized[:, 0], data_standardized[:, 1], alpha=0.5)
# plt.title('Original Data')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.grid(True)
#
# plt.subplot(1, 2, 2)
# plt.scatter(pc1, [0] * len(pc1), alpha=0.5)
# plt.title('Projection onto First Principal Component')
# plt.xlabel('PC1')
# plt.grid(True)
#
# plt.tight_layout()
# plt.show()

print(os.getcwd())
