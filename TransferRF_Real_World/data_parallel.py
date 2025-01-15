#!/usr/bin/env python
# coding: utf-8
import os
import ioh
import numpy as np
from scipy.stats import special_ortho_group
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from decimal import *
from utils import *
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from scipy.stats.qmc import Sobol
import gc
import scipy


def generate_data_kinematics_robot_arm_all_training_transfer(problem_selection, problem_origin, problem_target, dimension, number_of_data_RF_training, 
                number_of_data_RF_test, number_of_data_TL_training, number_of_data_TL_test, k, save_path):
    axis_names = ['Position1', 'Position2', 'Position3', 'Position4', 'Position5', 'Position6', 'Position7',
                  'Velocity1', 'Velocity2', 'Velocity3', 'Velocity4', 'Velocity5', 'Velocity6', 'Velocity7',
                  'Acceleration1', 'Acceleration2', 'Acceleration3', 'Acceleration4', 'Acceleration5', 
                  'Acceleration6', 'Acceleration7', 'Torque1', 'Torque2', 'Torque3', 'Torque4', 'Torque5', 
                  'Torque6', 'Torque7']
    RF_data = scipy.io.loadmat('data/kinematics_robot_arm/sarcos_inv.mat')
    RF_data = pd.DataFrame(RF_data['sarcos_inv'], columns=axis_names)

    TL_data = scipy.io.loadmat('data/kinematics_robot_arm/sarcos_inv_test.mat')
    TL_data = pd.DataFrame(TL_data['sarcos_inv_test'], columns=axis_names)

    # Combine RF and TL datasets
    combined_data = pd.concat([RF_data, TL_data], ignore_index=True)
    print(f"Size of Combined Dataset: {combined_data.shape}")

    RF_sampled = combined_data.sample(n=30000, random_state=k)
    print(f"Size of Sampled RF Data: {RF_sampled.shape}")
    X_RF_full = RF_sampled.iloc[:, :dimension].values  # Input features
    Y_RF_full = RF_sampled[problem_origin].values  # Target torque

    # Scale inputs
    X_RF_mean = X_RF_full.mean(axis=0)
    X_RF_std = X_RF_full.std(axis=0)
    X_RF_scaled = (X_RF_full - X_RF_mean) / np.where(X_RF_std == 0, 1, X_RF_std)  # Avoid division by zero
    Y_RF_mean = Y_RF_full.mean()
    Y_RF_std = Y_RF_full.std()
    Y_RF_scaled = (Y_RF_full - Y_RF_mean) / (Y_RF_std if Y_RF_std != 0 else 1)  # Standard scaling
    Y_RF_full_transformed = np.log10(Y_RF_scaled - Y_RF_scaled.min() + 1).reshape(-1, 1)  # Log transformation
    print(f"Transformed RF Target Shape: {Y_RF_full_transformed.shape}")

    X_RF_training = X_RF_scaled
    Y_RF_training = Y_RF_full_transformed
    X_RF_test = X_RF_scaled
    Y_RF_test = Y_RF_full_transformed
    plot_and_save_histogram(Y_RF_scaled, save_path, k, f'{problem_origin}_RF_original')
    plot_and_save_histogram(Y_RF_full_transformed, save_path, k, f'{problem_origin}_RF_transformed')

    del X_RF_full, Y_RF_full, X_RF_scaled, Y_RF_scaled, Y_RF_full_transformed

    ### Random Sampling for TL task
    TL_sampled = combined_data.sample(n=30000, random_state=k)
    print(f"Size of Sampled TL Data: {TL_sampled.shape}")
    X_TL_full = TL_sampled.iloc[:, :dimension].values  # Input features
    Y_TL_full = TL_sampled[problem_target].values  # Target torque

    # Scale inputs
    X_TL_mean = X_TL_full.mean(axis=0)
    X_TL_std = X_TL_full.std(axis=0)
    X_TL_scaled = (X_TL_full - X_TL_mean) / np.where(X_TL_std == 0, 1, X_TL_std)  # Avoid division by zero
    Y_TL_mean = Y_TL_full.mean()
    Y_TL_std = Y_TL_full.std()
    Y_TL_scaled = (Y_TL_full - Y_TL_mean) / (Y_TL_std if Y_TL_std != 0 else 1)  # Standard scaling
    Y_TL_full_transformed = np.log10(Y_TL_scaled - Y_TL_scaled.min() + 1).reshape(-1, 1)  # Log transformation
    print(f"Transformed TL Target Shape: {Y_TL_full_transformed.shape}")

    X_TL_training = X_TL_scaled
    Y_TL_training = Y_TL_full_transformed
    X_TL_test = X_TL_scaled
    Y_TL_test = Y_TL_full_transformed
    plot_and_save_histogram(Y_TL_scaled, save_path, k, f'{problem_target}_TL_original')
    plot_and_save_histogram(Y_TL_full_transformed, save_path, k, f'{problem_target}_TL_transformed')

    del X_TL_full, X_TL_scaled, Y_TL_full, Y_TL_scaled, Y_TL_full_transformed # Clear memory

    TL_training_data = np.concatenate((X_TL_training, Y_TL_training), axis=1)

    gc.collect()

    return X_RF_training, Y_RF_training, X_RF_test, Y_RF_test, X_TL_training, Y_TL_training, \
           X_TL_test, Y_TL_test, TL_training_data

def generate_data_asteroid_all_training_transfer(problem_selection, problem_origin, problem_target, dimension, number_of_data_RF_training, number_of_data_RF_test,
                  number_of_data_TL_training, number_of_data_TL_test, k, save_path):
    RF_data = pd.read_csv(problem_origin)
    RF_data = RF_data.sample(n=40000, random_state=k)
    X_RF_full = RF_data[['x1', 'x2']].values
    Y_RF_full = RF_data['y'].values
    Y_RF_full_transformed = np.log10(Y_RF_full + 1).reshape(-1, 1)
    print(f"Loaded RF Data from: {problem_origin}")
    print("Size of X_RF_full:", X_RF_full.shape)
    print("Size of Y_RF_full_transformed:", Y_RF_full_transformed.shape)
    plot_and_save_histogram(Y_RF_full, save_path, k, 'original_original')
    plot_and_save_histogram(Y_RF_full_transformed, save_path, k, 'original_transformed')

    X_RF_training = X_RF_full
    Y_RF_training = Y_RF_full_transformed
    X_RF_test = X_RF_full
    Y_RF_test = Y_RF_full_transformed

    del X_RF_full, Y_RF_full, Y_RF_full_transformed

    # Load TL (target) data
    TL_data = pd.read_csv(problem_target)
    TL_data = TL_data.sample(n=40000, random_state=k + 1)
    X_TL_full = TL_data[['x1', 'x2']].values
    Y_TL_full = TL_data['y'].values
    Y_TL_full_transformed = np.log10(Y_TL_full + 1).reshape(-1, 1)
    print(f"Loaded TL Data from: {problem_target}")
    print("Size of X_TL_full:", X_TL_full.shape)
    print("Size of Y_TL_full_transformed:", Y_TL_full_transformed.shape)
    plot_and_save_histogram(Y_TL_full, save_path, k, 'target_original')
    plot_and_save_histogram(Y_TL_full_transformed, save_path, k, 'target_transformed')

    X_TL_training = X_TL_full
    Y_TL_training = Y_TL_full_transformed
    X_TL_test = X_TL_full
    Y_TL_test = Y_TL_full_transformed

    del X_TL_full, Y_TL_full, Y_TL_full_transformed # Clear memory

    TL_training_data = np.concatenate((X_TL_training, Y_TL_training), axis=1)

    gc.collect()  # Explicitly invoke garbage collection

    return X_RF_training, Y_RF_training, X_RF_test, Y_RF_test, X_TL_training, Y_TL_training, \
           X_TL_test, Y_TL_test, TL_training_data

def generate_data_gbea_all_training_transfer(problem_selection, problem_origin, problem_target, dimension, number_of_data_RF_training, number_of_data_RF_test,
                  number_of_data_TL_training, number_of_data_TL_test, k, save_path):
    RF_data = pd.read_csv('data/marioGAN/decoration_frequency_overworld_dim_10.csv')
    X_RF_full = RF_data[['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10']].values
    Y_RF_full = RF_data[problem_origin].values
    Y_RF_full_transformed = np.log10(Y_RF_full + 1).reshape(-1, 1)
    print("Size of X_RF_full:", X_RF_full.shape)
    print("Size of Y_RF_full_transformed:", Y_RF_full_transformed.shape)
    plot_and_save_histogram(Y_RF_full, save_path, k, 'original_original')
    plot_and_save_histogram(Y_RF_full_transformed, save_path, k, 'original_transformed')

    X_RF_training = X_RF_full
    Y_RF_training = Y_RF_full_transformed
    X_RF_test = X_RF_full
    Y_RF_test = Y_RF_full_transformed

    del X_RF_full, Y_RF_full, Y_RF_full_transformed

    TL_data = pd.read_csv('data/marioGAN/decoration_frequency_overworld_dim_10.csv')
    X_TL_full = TL_data[['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10']].values
    Y_TL_full = TL_data[problem_target].values
    Y_TL_full_transformed = np.log10(Y_TL_full + 1).reshape(-1, 1)
    print("Size of X_TL_full:", X_TL_full.shape)
    print("Size of Y_TL_full_transformed:", Y_TL_full_transformed.shape)
    plot_and_save_histogram(Y_TL_full, save_path, k, 'target_original')
    plot_and_save_histogram(Y_TL_full_transformed, save_path, k, 'target_transformed')

    X_TL_training = X_TL_full
    Y_TL_training = Y_TL_full_transformed
    X_TL_test = X_TL_full
    Y_TL_test = Y_TL_full_transformed

    del X_TL_full, Y_TL_full, Y_TL_full_transformed # Clear memory

    TL_training_data = np.concatenate((X_TL_training, Y_TL_training), axis=1)

    gc.collect()  # Explicitly invoke garbage collection

    return X_RF_training, Y_RF_training, X_RF_test, Y_RF_test, X_TL_training, Y_TL_training, \
           X_TL_test, Y_TL_test, TL_training_data