#!/usr/bin/env python
# coding: utf-8

import numpy as np
from surrogate import *
from scipy.linalg import expm
from nevergrad.optimization import optimizerlib
from nevergrad.parametrization import parameter as p
import nevergrad as ng
from nevergrad.optimization import registry
from utils import loss_plot, plot_sigma_values
import cma
from scipy.optimize import minimize


def cost_function_RF(x, y, weight, translation, m):
    x_affine = (weight @ x.T).T + translation
    mean = m.predict(x_affine)
    mean = mean.reshape(-1, 1)  # Ensure mean is a column vector
    diff = mean - y
    differences_squared = diff ** 2
    mean_diff = np.sum(differences_squared) / 2
    return mean_diff

def generate_gaussian_random_antisymmetric_matrix(n):
    gaussian_random_antisymmetric_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):  # Only fill upper triangle
            gaussian_random_antisymmetric_matrix[i, j] = np.random.uniform(-np.pi, np.pi)  # Changed to [-π, π]
            gaussian_random_antisymmetric_matrix[j, i] = -gaussian_random_antisymmetric_matrix[i, j]
    return gaussian_random_antisymmetric_matrix

def calculate_initial_sigma(lower_bounds, upper_bounds):
    # Calculate the range for each dimension
    ranges = [upper - lower for lower, upper in zip(lower_bounds, upper_bounds)]
    
    # Use the average range divided by a heuristic value (e.g., 5) as the initial sigma
    sigma0 = sum(ranges) / len(ranges) / 5
    
    return sigma0
    
def run_optimization_cma(dimension, X_TL_training, Y_TL_training, m, problem_selection, save_path, k):
    loss_record = []

    # Initial parameters setup
    initial_tangent_vector = generate_gaussian_random_antisymmetric_matrix(dimension)
    initial_rotation = expm(initial_tangent_vector)
    initial_values = initial_tangent_vector[np.triu_indices(dimension, 1)]
    initial_translation = np.random.uniform(-0.5, 0.5, (1, dimension)).flatten()
    initial_params = np.concatenate((initial_values, initial_translation))

    # Adjusting bounds setup for your existing function
    num_rotation_params = len(initial_values)
    rotation_bounds_lower = [-np.pi] * num_rotation_params
    rotation_bounds_upper = [np.pi] * num_rotation_params
    translation_bounds_lower = [-1.5] * dimension
    translation_bounds_upper = [1.5] * dimension

    # Bounds setup for your function
    lower_bounds = [-np.pi] * len(initial_values) + [-1.5] * dimension
    upper_bounds = [np.pi] * len(initial_values) + [1.5] * dimension
    sigma0 = calculate_initial_sigma(lower_bounds, upper_bounds)

    # Options for BIPOP-CMA-ES
    options = {
        'bounds': [lower_bounds, upper_bounds],
        'maxfevals': 500000,
        'popsize': 64,
        'verbose': 0,
        'verb_log': 0  # Disables logging of data to files
    }

    # Objective function wrapper to include additional arguments
    def objective_wrapper(x):
        rotation_matrix = reconstruct_rotation(x[:dimension * (dimension - 1) // 2], dimension)
        translation_vector = x[dimension * (dimension - 1) // 2:].reshape(1, dimension)
        return cost_function_RF(X_TL_training, Y_TL_training, rotation_matrix, translation_vector, m)

    # Using fmin with BIPOP
    result = cma.fmin(
        objective_function=objective_wrapper,
        x0=initial_params,
        sigma0=sigma0,
        options=options,
        bipop=True,
        restarts=9
    )

    print("Start")
    print(result)  # Debug print to inspect the structure

    # Unpack results directly
    xbest, fbest, iterations, evaluations, eff_evaluations, _, _, _, _, es, logger = result

    # Further use of results
    best_rotation = reconstruct_rotation(xbest[:dimension * (dimension - 1) // 2], dimension)
    best_translation = xbest[dimension * (dimension - 1) // 2:].reshape(1, dimension)
    final_loss = cost_function_RF(X_TL_training, Y_TL_training, best_rotation, best_translation, m)

    print("Final Loss value: ", final_loss)
    return final_loss, best_rotation, best_translation, loss_record

def reconstruct_rotation(vector, dimension):
    # Reconstruct the antisymmetric matrix and compute the rotation matrix
    candidate_tangent_vector = np.zeros((dimension, dimension))
    indices_upper = np.triu_indices(dimension, 1)
    candidate_tangent_vector[indices_upper] = vector
    candidate_tangent_vector[indices_upper[::-1]] = -vector  # Reverse the indices for the lower triangle
    return expm(candidate_tangent_vector)
