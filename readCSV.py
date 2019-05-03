# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:31:28 2019

@author: lclea
"""

import csv
import pandas as pd

# was not reading all the rows, so there is a 'dummy' row in the csv as the first row. This row is not read by pd
df = pd.read_csv('C:/Users/lclea/Documents/machine_learning/Project/ssim_matrix.csv', skiprows=0)

# we have a 92x92 matrix 
print(df)
