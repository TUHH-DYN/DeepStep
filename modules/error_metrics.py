import numpy as np
from math import sqrt

def rmse_numpy(y_true, y_pred):
    error = (y_true - y_pred)
    sq_error = np.square(error)
    error_sum = np.sum(sq_error)
    return sqrt(error_sum/y_pred.size)

def mse_numpy(y_true, y_pred):
    error = (y_true - y_pred)
    sq_error = np.square(error)
    error_sum = np.sum(sq_error)
    return error_sum/y_pred.size

def mae_numpy(y_true, y_pred):
    error = np.abs(y_true - y_pred)
    error_sum = np.sum(error)
    return error_sum/y_pred.size

def me_numpy(y_true, y_pred):
    error = y_true - y_pred
    error_sum = np.sum(error)
    return error_sum/y_pred.size

def mes_numpy(y_true, y_pred):
    error = y_true - y_pred
    error_sum = np.sum(error)
    return (error_sum/y_pred.size)**2

def domain_rmse_numpy(y_true, y_pred, mask):
    error = (y_true - y_pred)*mask
    sq_error = np.square(error)
    mask_sum = np.sum(mask)
    error_sum = np.sum(sq_error)
    return sqrt(error_sum/mask_sum)

def domain_mse_numpy(y_true, y_pred, mask):
    error = (y_true - y_pred)*mask
    sq_error = np.square(error)
    mask_sum = np.sum(mask)
    error_sum = np.sum(sq_error)
    return error_sum/mask_sum

def domain_mae_numpy(y_true, y_pred, mask):
    error = (y_true - y_pred)*mask
    abs_error = np.abs(error)
    mask_sum = np.sum(mask)
    error_sum = np.sum(abs_error)
    return error_sum/mask_sum

def domain_me_numpy(y_true, y_pred, mask):
    error = (y_true - y_pred)*mask
    mask_sum = np.sum(mask)
    error_sum = np.sum(error)
    return error_sum/mask_sum

def domain_mes_numpy(y_true, y_pred, mask):
    error = (y_true - y_pred)*mask
    mask_sum = np.sum(mask)
    error_sum = np.sum(error)
    return (error_sum/mask_sum)**2

def integral_trapz_2d_numpy(y):    
    tr1 = (y[0,0] + y[0,-1] + y[-1,0] + y[-1,-1] )/4.0 # sum of corner points
    tr2 = (np.sum(y[0,1:-1],axis=0) + np.sum(y[-1,1:-1],axis=0) + np.sum(y[1:-1,0],axis=0) + np.sum(y[1:-1,-1],axis=0))/2.0  # along the edges twice in total
    tr3 = np.sum(y[1:-1,1:-1],axis=(0,1)) # everything else is used four times in total sum
    return tr1 + tr2 + tr3

def ssp_numpy(y_true, y_pred):       
    spec1 = np.fft.fft2(y_true.astype(complex))                   
    spec2 = np.fft.fft2(y_pred.astype(complex))
    nominator = np.sqrt(integral_trapz_2d_numpy(np.square(np.abs(spec1- spec2))))
    denominator = np.sqrt(integral_trapz_2d_numpy(np.square(np.abs(spec1)))) + np.sqrt(integral_trapz_2d_numpy(np.square(np.abs(spec2))))
    ssp = np.divide(nominator, denominator)
    return ssp

def domain_ssp_numpy(y_true, y_pred, mask):       
    y_true_valid = np.multiply(y_true,mask)                                       
    y_pred_valid = np.multiply(y_pred,mask) 
    spec1 = np.fft.fft2(y_true_valid.astype(complex))                   
    spec2 = np.fft.fft2(y_pred_valid.astype(complex))
    nominator = np.sqrt(integral_trapz_2d_numpy(np.square(np.abs(spec1- spec2))))
    denominator = np.sqrt(integral_trapz_2d_numpy(np.square(np.abs(spec1)))) + np.sqrt(integral_trapz_2d_numpy(np.square(np.abs(spec2))))
    ssp = np.divide(nominator, denominator)
    return ssp