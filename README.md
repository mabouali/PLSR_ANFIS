# PLSR-ANFIS
## Introduction
**_PLSR_ANFIS* is a MATLAB software package that uses Partial Least-Square Regression (PLSR) and Adaptive Network-based Fuzzy Inference System (ANFIS) at the same time to establish a preditive model between a set of independent variables (X) and a dependent variable (Y). The model harnesses a two phase approach as follows:

1. **Phase 1:**
  1. **Part 1:** Using PLSR, establishes a predictive model to estimate Y. This is denoted as YP.
  2. **Part 2:** Using PLSR, establishes a predictive model to estimate the error of YP. This is denoted as ErrP, or predicted error.
2. **Phase 2:** uses ANFIS to establish a predictive model to estimate Y, using YP and ErrP as the model input.

## How to use the code?
there are couple of MATLAB codes available under MCode directory. The two main functions are:

'''
[result, detail]=PLSR_ANFIS(X,Y,inputMFTypes,mfNum)
'''

and 

'''
[modYP,YP,ErrP]=apply_PLSR_ANFIS(X,PLSR_ANFIS_Result)
'''

As their name implies the first one is used to train and establish the proposed two phase model and the second one is used to apply the model on a new data set.

There are two test files also available, showing how to use these two functions.

## How the Input Data Set should be formatted?
X is the independent variable and Y is the dependent variable. Y must be a double precision vector of nx1 where n is the number of samples or ovservations. X should be a double precision matrix of size nxm where m is the number of independent variables. Each row in X should correspond to the same row of Y.



