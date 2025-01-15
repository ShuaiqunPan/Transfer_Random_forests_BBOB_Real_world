#!/usr/bin/env python
# coding: utf-8
'''
Script Name: surrogate.py
Description: This script is the program of implementing the RF surrogate model.
'''
from sklearn.ensemble import RandomForestRegressor


def RF_regression(X, Y, k, save_path):
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X, Y.ravel())
    return rf
