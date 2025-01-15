#!/usr/bin/env python
# coding: utf-8
import os
num_threads = "1"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
import ioh
import time
import pickle
import numpy as np
from decimal import *
from multiprocessing import Pool
import matplotlib.pyplot as plt
from utils import *
from data import *
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.stats import randint
from search import *
import json
import gc
from scipy.stats import kruskal
import scikit_posthocs as sp
import pandas as pd


def main(problem_selection):
    dimension = 2
    
    # Set the number of training dataset and test dataset for training the original Random forest regression
    number_of_data_RF_training = 1000 * dimension
    number_of_data_RF_test = 1000 * dimension

    # Set the number of training dataset, and test dataset for transfer learning (TL) process
    number_of_data_TL_training = 50 * dimension
    max_number_of_data_TL_training = 50 * dimension
    number_of_data_TL_test = 1000 * dimension
    
    number_of_repetition = 10
    
    # Specify the path of the Result folder
    result_path = 'Result'
    os.makedirs(result_path, exist_ok=True)

    save_original_model = result_path + '/' + str(dimension) + '/' + f"{problem_selection}" + '/original_model'
    save_path = result_path + '/' + str(dimension) + '/' + f"{problem_selection}" + '/' + str(number_of_data_TL_training)
    save_transfer_model = save_path + '/transfer_model'
    save_without_transfer = save_path + '/without_transfer'
    save_loss = save_path + '/loss_value'
    
    isExist = os.path.exists(save_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(save_path, exist_ok=True)
        print("The new directory is created!")

    complete_name = os.path.join(save_path, f"{problem_selection}_output"'.log')
    logger = Logger(complete_name)
    
    print("This is the BBOB function: ", problem_selection)
    print("Number of dimension: ", dimension)
    print("Number of data for training origianl RF: ", number_of_data_RF_training)
    print("Number of data for testing origianl RF: ", number_of_data_RF_test)
    print("Number of data for training Transferred RF: ", number_of_data_TL_training)
    print("Number of data for testing Transferred RF: ", number_of_data_TL_test)
    print("The setting of number of repetition for experiments: ", number_of_repetition)

    store_mean_absolute_percentage_error_RF = []
    store_SMAPE_RF = []
    store_R_square_score_RF = []
    store_log_ratio_RF = []
    store_square_sum_log_ratio_RF = []

    store_mean_absolute_percentage_error_RF_TL_test = []
    store_SMAPE_RF_TL_test = []
    store_R_square_score_RF_TL_test = []
    store_log_ratio_RF_TL_test = []
    store_square_sum_log_ratio_RF_TL_test = []
    
    store_mean_absolute_percentage_error_RF_without_transfer = []
    store_SMAPE_RF_without_transfer = []
    store_R_square_score_RF_without_transfer = []
    store_log_ratio_RF_without_transfer = []
    store_square_sum_log_ratio_RF_without_transfer = []

    store_mean_absolute_percentage_error_TL = []
    store_SMAPE_TL = []
    store_R_square_score_TL = []
    store_log_ratio_TL = []
    store_square_sum_log_ratio_TL = []
    
    frobenius_value_list = []
    inner_product_list = []
    new_metric_value_list = []

    for k in range(1, number_of_repetition+1):

        np.random.seed(k)
        print('-------------------------------------------------------------------------------------------------')
        print("Number of repetition: ", k)
        print('-------------------------------------------------------------------------------------------------')

        # We generate the datasets accordingly.
        X_RF_training, Y_RF_training, X_RF_test, Y_RF_test, X_TL_training, Y_TL_training, \
            X_TL_test, Y_TL_test, TL_training_data, R, beta, problem1 = generate_data(problem_selection,
                                                                                            dimension,
                                                                                            number_of_data_RF_training,
                                                                                            number_of_data_RF_test,
                                                                                            max_number_of_data_TL_training,
                                                                                            number_of_data_TL_test, k, save_path)

        train_data_generation = TL_training_data[np.random.choice(TL_training_data.shape[0], number_of_data_TL_training, replace=False),:]
        X_TL_training = train_data_generation[:, :dimension]
        Y_TL_training = train_data_generation[:, [-1]]

        start_build_RF_model = time.time()
        isExist = os.path.exists(save_original_model)
        if not isExist:
            os.makedirs(save_original_model, exist_ok=True)
            print("The new directory is created!")
        
        complete_name = os.path.join(save_original_model, f"{problem_selection}_{k}"'.sav')
        # If the model exists, load it
        if os.path.isfile(complete_name):
            print("Loading existing model...")
            m = pickle.load(open(complete_name, 'rb'))
        else:
            print("Building new model...")
            m = RF_regression(X_RF_training, Y_RF_training, k, save_original_model)
            pickle.dump(m, open(complete_name, 'wb'))
        
        finish_build_RF_model = time.time()
        print("The time for training the origianl RF model: ", finish_build_RF_model - start_build_RF_model, "seconds.")
        
        print("Test the original RF test data on original RF Model: ")
        mean_RF = m.predict(X_RF_test)
        print(mean_RF.shape)
        mean_RF = mean_RF.reshape(-1, 1)
        print(mean_RF.shape)
        
        MAPE_RF, SMAPE_RF, R_square_RF, log_ratio_RF, square_sum_log_ratio_RF = prediction(mean_RF, Y_RF_test)

        store_mean_absolute_percentage_error_RF.append(MAPE_RF)
        store_SMAPE_RF.append(SMAPE_RF)
        store_R_square_score_RF.append(R_square_RF)
        store_log_ratio_RF.append(log_ratio_RF)
        store_square_sum_log_ratio_RF.append(square_sum_log_ratio_RF)
        
        print("Test the original trainined RF Model on transfer learning test data: ")
        # Compute posterior means and variance on the test dataset of TL
        mean_RF_test = m.predict(X_TL_test)
        print(mean_RF_test.shape)
        mean_RF_test = mean_RF_test.reshape(-1, 1)
        print(mean_RF_test.shape)
        MAPE_RF_test, SMAPE_RF_test, R_square_RF_test, log_ratio_RF_test, square_sum_log_ratio_RF_test = prediction(mean_RF_test, Y_TL_test)

        store_mean_absolute_percentage_error_RF_TL_test.append(MAPE_RF_test)
        store_SMAPE_RF_TL_test.append(SMAPE_RF_test)
        store_R_square_score_RF_TL_test.append(R_square_RF_test)
        store_log_ratio_RF_TL_test.append(log_ratio_RF_test)
        store_square_sum_log_ratio_RF_TL_test.append(square_sum_log_ratio_RF_test)
        
        isExist = os.path.exists(save_without_transfer)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_without_transfer, exist_ok=True)
            print("The new directory is created!")
        
        complete_name = os.path.join(save_without_transfer, f"{problem_selection}_{k}"'.sav')
        # If the model exists, load it
        if os.path.isfile(complete_name):
            print("Loading existing model...")
            m_without_transfer = pickle.load(open(complete_name, 'rb'))
        else:
            print("Building new model...")
            m_without_transfer = RF_regression(X_TL_training, Y_TL_training, k, save_without_transfer)
            pickle.dump(m_without_transfer, open(complete_name, 'wb'))
        
        print("Build RF model with all the transfer learning data: ")
        mean_RF_without_transfer = m_without_transfer.predict(X_TL_test)
        print(mean_RF_without_transfer.shape)
        mean_RF_without_transfer = mean_RF_without_transfer.reshape(-1, 1)
        print(mean_RF_without_transfer.shape)
        MAPE_RF_without_transfer, SMAPE_RF_without_transfer, R_square_RF_without_transfer, log_ratio_RF_without_transfer, square_sum_log_ratio_RF_without_transfer = prediction(mean_RF_without_transfer, Y_TL_test)

        store_mean_absolute_percentage_error_RF_without_transfer.append(MAPE_RF_without_transfer)
        store_SMAPE_RF_without_transfer.append(SMAPE_RF_without_transfer)
        store_R_square_score_RF_without_transfer.append(R_square_RF_without_transfer)
        store_log_ratio_RF_without_transfer.append(log_ratio_RF_without_transfer)
        store_square_sum_log_ratio_RF_without_transfer.append(square_sum_log_ratio_RF_without_transfer)
        
        isExist = os.path.exists(save_transfer_model)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_transfer_model, exist_ok=True)
            print("The new directory is created!")
        
        '''
        Save the loss values
        '''
        start_training_each_epoch = time.time()
        isExist = os.path.exists(save_loss)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_loss, exist_ok=True)
            print("The new directory is created!")

        final_rotation_matrix_complete_name = os.path.join(save_transfer_model, f"final_rotation_{problem_selection}_{k}"'.npy')
        final_translation_matrix_complete_name = os.path.join(save_transfer_model, f"final_translation_{problem_selection}_{k}"'.npy')
            
        # If the model exists, load it
        if os.path.isfile(final_rotation_matrix_complete_name):
            print("Loading existing model...")
            final_rotation_matrix = np.load(final_rotation_matrix_complete_name)
            final_translation_matrix = np.load(final_translation_matrix_complete_name)
        else:
            print("Building new model with CMA-ES...")
            
            final_cost, final_rotation_matrix, final_translation_matrix, loss_record = run_optimization_cma(dimension, X_TL_training, Y_TL_training, m, problem_selection, save_loss, k)

            final_loss_path = os.path.join(save_loss, f"loss_history_{problem_selection}_{k}.json")
            with open(final_loss_path, "w") as file:
                json.dump(loss_record, file)

            np.save(final_rotation_matrix_complete_name, final_rotation_matrix)
            np.save(final_translation_matrix_complete_name, final_translation_matrix)
        
        end_training_each_epoch = time.time()
        print("The training time for the whole iterations with CMA-ES : ", end_training_each_epoch - start_training_each_epoch, "seconds.")
        
        print(final_rotation_matrix)
        print(final_translation_matrix)

        print("The determinant of the generated rotation matrix: ", np.linalg.det(final_rotation_matrix))

        # Frobenius norm value between the generated rotation matrix and the origianl matrix
        new_metric_value = rotation_metric(final_rotation_matrix, R)
        print("New metric: ", new_metric_value)
        
        frobenius_value = frobenius_norm(final_rotation_matrix, R)
        print("Frobenius norm: ", frobenius_value)
        
        normalized_frobenius_value = frobenius_norm(final_rotation_matrix, R) / dimension
        print("Frobenius norm (Normalized): ", normalized_frobenius_value)

        # Inner product between the generated rotation matrix and the origianl matrix (another metric)
        inner_product = np.trace(final_rotation_matrix.T @ R)
        print("Inner product: ", inner_product)
        
        normalized_inner_product = np.trace(final_rotation_matrix.T @ R) / dimension
        print("Inner product (Normalized): ", normalized_inner_product)
        
        new_metric_value_list.append(new_metric_value)
        frobenius_value_list.append(normalized_frobenius_value)
        inner_product_list.append(normalized_inner_product)
            
        print("Test the transferred RF on transfer learning test data: ")
        x_affine_test = (final_rotation_matrix @ X_TL_test.T).T + final_translation_matrix
        mean_TL = m.predict(x_affine_test)
        print(mean_TL.shape)
        mean_TL = mean_TL.reshape(-1, 1)
        print(mean_TL.shape)
        mean_absolute_percentage_error_TL, SMAPE_TL, R_square_TL, log_ratio_TL, square_sum_log_ratio_TL = prediction(mean_TL, Y_TL_test)
                
        store_mean_absolute_percentage_error_TL.append(mean_absolute_percentage_error_TL)
        store_SMAPE_TL.append(SMAPE_TL)
        store_R_square_score_TL.append(R_square_TL)
        store_log_ratio_TL.append(log_ratio_TL)
        store_square_sum_log_ratio_TL.append(square_sum_log_ratio_TL)

        # Delete the model and free up memory
        del m, m_without_transfer, final_rotation_matrix, final_translation_matrix

        # At the end of each repetition, delete the datasets
        del X_RF_training, Y_RF_training, X_RF_test, Y_RF_test
        del X_TL_training, Y_TL_training, X_TL_test, Y_TL_test, TL_training_data

        gc.collect()
    
    print("The Original Random forest regression Model: ")
    evaluation(store_mean_absolute_percentage_error_RF, store_SMAPE_RF, store_R_square_score_RF, store_log_ratio_RF, store_square_sum_log_ratio_RF)

    print("The Original Random forest regression Model on transfer learning test data: ")
    evaluation(store_mean_absolute_percentage_error_RF_TL_test, store_SMAPE_RF_TL_test, store_R_square_score_RF_TL_test, store_log_ratio_RF_TL_test, store_square_sum_log_ratio_RF_TL_test)

    print("The model built with transfer learning data: ")
    evaluation(store_mean_absolute_percentage_error_RF_without_transfer, store_SMAPE_RF_without_transfer, store_R_square_score_RF_without_transfer, store_log_ratio_RF_without_transfer, store_square_sum_log_ratio_RF_without_transfer)
    
    print("The transferred Random forest regression Model on transfer learning test data: ")
    evaluation(store_mean_absolute_percentage_error_TL, store_SMAPE_TL, store_R_square_score_TL, store_log_ratio_TL, store_square_sum_log_ratio_TL)

    print("1. The p-value of U-test on MAPE")
    p_value_MAPE = mannwhitneyu_test(store_mean_absolute_percentage_error_RF_TL_test, store_mean_absolute_percentage_error_TL)
    
    print("2. The p-value of U-test on SMAPE")
    p_value_SMAPE = mannwhitneyu_test(store_SMAPE_RF_TL_test, store_SMAPE_TL)
    
    print("3. The p-value of U-test on R squared score")
    p_value_R2 = mannwhitneyu_test(store_R_square_score_RF_TL_test, store_R_square_score_TL)
    
    print("4. The p-value of U-test on log ratio score")
    p_value_log_ratio = mannwhitneyu_test(store_log_ratio_RF_TL_test, store_log_ratio_TL)
    
    print("5. The p-value of U-test on square sum log ratio")
    p_value_square_sum_log_ratio = mannwhitneyu_test(store_square_sum_log_ratio_RF_TL_test, store_square_sum_log_ratio_TL)

    print("6. The average of frobenius value is: ", np.mean(frobenius_value_list))
    print("7. The average of inner product value is: ", np.mean(inner_product_list))
    print("8. The average of new metric value is: ", np.mean(new_metric_value_list))
    
    frobenius_box_plot(frobenius_value_list, inner_product_list, problem_selection, dimension, save_path)
    
    # Kruskal-Wallis Test
    group1 = store_SMAPE_RF_TL_test
    group2 = store_SMAPE_RF_without_transfer
    group3 = store_SMAPE_TL
    h_stat, p_val = kruskal(group1, group2, group3)

    # If p_val is significant, proceed with post-hoc analysis
    if p_val < 0.05:
        data = pd.DataFrame({
            'values': group1 + group2 + group3,
            'groups': ['group1']*len(group1) + ['group2']*len(group2) + ['group3']*len(group3)
        })
        
        # Dunn's test with Bonferroni correction
        post_hoc_res = sp.posthoc_dunn(data, val_col='values', group_col='groups', p_adjust='bonferroni')
        print(post_hoc_res)

    logger.reset()

if __name__ == '__main__':
    BBOB_function_list = ['Sphere', 'Ellipsoid', 'Rastrigin', 'BuecheRastrigin', 'LinearSlope', 'AttractiveSector', 'StepEllipsoid', 
                          'Rosenbrock', 'RosenbrockRotated', 'EllipsoidRotated', 'Discus', 'BentCigar', 'SharpRidge', 'DifferentPowers', 
                          'RastriginRotated', 'Weierstrass', 'Schaffers10', 'Schaffers1000', 'GriewankRosenBrock', 'Schwefel', 'Gallagher101', 
                          'Gallagher21', 'Katsuura', 'LunacekBiRastrigin']
    pool = Pool()
    result = pool.map(main, BBOB_function_list)
    pool.close()
    pool.join()