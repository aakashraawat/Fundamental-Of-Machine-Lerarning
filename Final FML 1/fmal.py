# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:32:23 2024

@author: Aditya
"""
import numpy as np
from scipy.io import loadmat  # Use scipy.io to import the data
from sklearn.preprocessing import PolynomialFeatures

def my_linear_regression(phi, ys):
    w=np.linalg.solve(phi.T @ phi, phi.T @ ys)
    return w

def standardize(x,y):
    x_mean=np.mean(x,axis=0)
    x_std_dev=np.std(x,axis=0,ddof=1)
    print("\n\n",x_mean,"\n\n",x_std_dev)
    x_std=(x-x_mean)/(x_std_dev)
    
    y_mean=np.mean(y)
    y_std=y-y_mean
    return x_std, y_std

def my_quadratic_features(xs):
    xs = np.array(xs)
    N, D = xs.shape
    # Initialize a list of features, starting with the bias term (1)
    features= [np.ones((N, 1))]  # Bias term
    features.append(xs)
    
    # Add interaction terms (x_i * x_j), avoiding duplicates
    cross_terms = []
    for i in range(D):
        for j in range(i + 1, D):  # j starts from i+1 to avoid repetition
            cross_terms.append((xs[:, i] * xs[:, j]).reshape(-1, 1))
    # Add the interaction terms first
    features.extend(cross_terms)
    # Add quadratic terms (x_i^2)
    squared_terms = []
    for i in range(D):
        squared_terms.append((xs[:, i] ** 2).reshape(-1, 1))
    # Add quadratic terms at the end
    features.extend(squared_terms)
    # Concatenate all the features horizontally
    return np.hstack(features)
    



data=loadmat(r"C:\Users\Aditya\Desktop\a2\FML-WS24-BPA1_v1\notebooks\data\sarcos_inv.mat")

#array1 is for total train+validation data
#since loadmat gives dictionary we have to get the numerical part using key 
array1=data['sarcos_inv']
data=loadmat(r"C:\Users\Aditya\Desktop\a2\FML-WS24-BPA1_v1\notebooks\data\sarcos_inv_test.mat")

#array2 is for total test data
array2=data['sarcos_inv_test']
print(len(array1),"",len(array2))

#code for random 80% to 20% partition
train_size = 0.8  # 80% for training
val_size = 0.2 # 20% for validation
num_samples = array1.shape[0]
train_indices = np.random.choice(num_samples, size=round(train_size * num_samples), replace=False)
val_indices = np.setdiff1d(np.arange(num_samples), train_indices)
train_data = array1[train_indices]
val_data = array1[val_indices]

#print("\nTraining data:\n", train_data,"\nlenght=",len(train_data))
#print("\nValidation data:\n", val_data,"\nlenght=",len(val_data))

#code for splitting total train data into input (1st 21 columns) and output(22nd column)
xs_train = train_data[:,:21]
#print("\nxs train data\n",xs_train,"\nlenght=",len(xs_train))
ys_train = train_data[:,21:22]
#print("\nys_train data only torque column (i.e column no. 22) \n",ys_train,"\nlenght=",len(ys_train))

# same method for Input and output validation data
xs_valid = val_data[:,:21]
#print("\nxs_valid\n",xs_valid,"\nlenght=",len(xs_valid))
ys_valid =val_data[:,21:22]
#print("\nys_valid i.e torque column\n",ys_valid,"\nlenght=",len(ys_valid))

# Input and output test data
xs_test= array2[:,:21]
#print("\nxs_test\n",xs_test,"\nlenght=",len(xs_test))
ys_test =array2[:,21:22]
#print("\nys_test i.e torque column\n",ys_test,"\nlenght=",len(ys_test))

xs_train_std,ys_train_std=standardize(xs_train,ys_train)
xs_valid_std,ys_valid_std=standardize(xs_valid, ys_valid)
xs_test_std,ys_test_std=standardize(xs_test, ys_test)

#to standardize data
#print("\nxs_train_std\n",xs_train_std)
#print("\nys_train_std\n",ys_train_std)
#print("\nxs_valid_std\n",xs_valid_std)
#print("\nys_valid_std\n",ys_valid_std)
#print("\nxs_test_std\n",xs_train_std)
#print("\nys_test_std\n",ys_test_std)
#print(xs_train.shape,"\n",ys_train.shape)

#weight calculation for train data
my_weights=my_linear_regression(xs_train_std,ys_train_std)
#print("\nweights",my_weights,"\nshape",my_weights.shape)
my_valid_pred=(xs_train_std @ my_weights)
#print(my_valid_pred,my_valid_pred.shape)

my_mse=np.mean((ys_train_std-my_valid_pred)**2) 
print("\nMSE of train data after standardizing=",my_mse) 
#print(my_quadratic_features(xs_train))

xs_train_polynomial=PolynomialFeatures(degree=3, include_bias=True).fit_transform(xs_train_std)
xs_valid_polynomial=PolynomialFeatures(degree=3,include_bias=True).fit_transform(xs_valid_std)

my_weights=my_linear_regression(xs_train_polynomial, ys_train_std)
print("\nweights after for 3rd degree polynomial\n",my_weights,"\nshape=",my_weights.shape)
ys_train_pred= xs_train_polynomial @ my_weights
my_mse=np.mean((ys_train_std-ys_train_pred)**2) 
print("\n MSE for 3rd degree polynomial function=",my_mse)
# YOUR CODE HERE
#raise NotImplementedError()