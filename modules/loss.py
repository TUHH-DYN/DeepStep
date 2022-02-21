import numpy as np
# from math import sqrt
import tensorflow as tf


def mse(y_true, y_pred):
    error = tf.math.square(tf.math.subtract(y_true, y_pred))
    return tf.math.reduce_mean(error)

def mae(y_true, y_pred):
    error = tf.math.abs(tf.math.subtract(y_true, y_pred))
    return tf.math.reduce_mean(error)

def rmse(y_true, y_pred):
    error = tf.math.reduce_mean(tf.math.square(tf.math.subtract(y_true, y_pred)), axis=(1,2,3))
    return tf.math.reduce_mean(tf.math.sqrt(error))

def me(y_true, y_pred):
    mask = tf.ones_like(y_true)
    mask_sum = tf.math.reduce_sum(mask)
    y_true_mean = tf.math.divide(tf.math.reduce_sum(y_true), mask_sum)
    y_pred_mean = tf.math.divide(tf.math.reduce_sum(y_pred), mask_sum)
    me = tf.math.abs(tf.math.subtract(y_true_mean, y_pred_mean))
    return me

def mes(y_true, y_pred):
    mask = tf.ones_like(y_true)
    mask_sum = tf.math.reduce_sum(mask)
    y_true_mean = tf.math.divide(tf.math.reduce_sum(y_true), mask_sum)
    y_pred_mean = tf.math.divide(tf.math.reduce_sum(y_pred), mask_sum)
    mes = tf.math.square(tf.math.subtract(y_true_mean, y_pred_mean))
    return mes

def domain_mse(y_true, y_pred, mask, mask_crop = 0):
    if mask_crop > 0:
        mask = mask[:,mask_crop:-mask_crop,mask_crop:-mask_crop,:]
    error = tf.math.multiply(tf.math.subtract(y_true, y_pred), mask)
    sq_error = tf.math.square(error)
    mask_sum = tf.math.reduce_sum(mask)
    error_sum = tf.math.reduce_sum(sq_error)
    return tf.math.divide(error_sum, mask_sum)

def domain_mae(y_true, y_pred, mask, mask_crop = 0):
    if mask_crop > 0:
        mask = mask[:,mask_crop:-mask_crop,mask_crop:-mask_crop,:]
    error = tf.math.multiply(tf.math.subtract(y_true, y_pred), mask)
    error_abs = tf.math.abs(error)
    mask_sum = tf.math.reduce_sum(mask)
    error_sum = tf.math.reduce_sum(error_abs)
    return tf.math.divide(error_sum, mask_sum)

def domain_rmse(y_true, y_pred, mask, mask_crop = 0):
    if mask_crop > 0:
        mask = mask[:,mask_crop:-mask_crop,mask_crop:-mask_crop,:]
    error = tf.math.multiply(tf.math.subtract(y_true, y_pred), mask)
    sq_error = tf.math.square(error)
    mask_sum = tf.math.reduce_sum(mask, axis=(1,2,3))
    error_sum = tf.math.reduce_sum(sq_error, axis=(1,2,3))
    return tf.math.reduce_mean(tf.math.sqrt(tf.math.divide(error_sum, mask_sum)))

def integral_trapz_2d(y):    
    tr1 = (y[:,0,0] + y[:,0,-1] + y[:,-1,0] + y[:,-1,-1] )/4.0
    tr2 = (tf.reduce_sum(y[:,0,1:-1],axis=1) + tf.reduce_sum(y[:,-1,1:-1],axis=1) + tf.reduce_sum(y[:,1:-1,0],axis=1) + tf.reduce_sum(y[:,1:-1,-1],axis=1))/2.0   
    tr3 = tf.reduce_sum(y[:,1:-1,1:-1],axis=[1,2])
    return tr1 + tr2 + tr3

def ssp(y_true, y_pred):
    spec1 = tf.signal.fft2d(tf.cast(y_true, dtype=tf.complex64))                 
    spec2 = tf.signal.fft2d(tf.cast(y_pred, dtype=tf.complex64))
    nominator = tf.math.sqrt(integral_trapz_2d(tf.math.square(tf.math.abs(spec1- spec2))))
    denominator = tf.math.sqrt(integral_trapz_2d(tf.math.square(tf.math.abs(spec1)))) + tf.math.sqrt(integral_trapz_2d(tf.math.square(tf.math.abs(spec2))))
    SSP = tf.math.divide(nominator, denominator)
    return SSP

def domain_ssp(y_true, y_pred, mask, mask_crop = 0):
    if mask_crop > 0:
        mask = mask[:,mask_crop:-mask_crop,mask_crop:-mask_crop,:]
    y_true_valid = tf.math.multiply(y_true, mask)                                       
    y_pred_valid = tf.math.multiply(y_pred, mask) 
    spec1 = tf.signal.fft2d(tf.cast(y_true_valid, dtype=tf.complex64))                   
    spec2 = tf.signal.fft2d(tf.cast(y_pred_valid, dtype=tf.complex64))
    nominator = tf.math.sqrt(integral_trapz_2d(tf.math.square(tf.math.abs(spec1- spec2))))
    denominator = tf.math.sqrt(integral_trapz_2d(tf.math.square(tf.math.abs(spec1)))) + tf.math.sqrt(integral_trapz_2d(tf.math.square(tf.math.abs(spec2))))
    SSP = tf.math.divide(nominator, denominator)
    return SSP

def domain_me(y_true, y_pred, mask, mask_crop = 0):
    if mask_crop > 0:
        mask = mask[:,mask_crop:-mask_crop,mask_crop:-mask_crop,:]
    mask_sum = tf.math.reduce_sum(mask)
        
    y_true_mean = tf.math.divide(tf.math.reduce_sum(tf.math.multiply(y_true, mask)), mask_sum)
    y_pred_mean = tf.math.divide(tf.math.reduce_sum(tf.math.multiply(y_pred, mask)), mask_sum)
    me = tf.math.abs(tf.math.subtract(y_true_mean, y_pred_mean))
    return me

def domain_mes(y_true, y_pred, mask, mask_crop = 0):
    if mask_crop > 0:
        mask = mask[:,mask_crop:-mask_crop,mask_crop:-mask_crop,:]
    mask_sum = tf.math.reduce_sum(mask)
        
    y_true_mean = tf.math.divide(tf.math.reduce_sum(tf.math.multiply(y_true, mask)), mask_sum)
    y_pred_mean = tf.math.divide(tf.math.reduce_sum(tf.math.multiply(y_pred, mask)), mask_sum)
    mes = tf.math.square(tf.math.subtract(y_true_mean, y_pred_mean))
    return mes