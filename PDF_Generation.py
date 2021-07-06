'''
This script builds a set of Probability Density Functions (PDF's) out of a set of base_functions. 
The implemented set of base_functions is chosen to ensure a as random as possible shape of the PDF, 
while also ensuring a relatively uniform distribution regarding the monotony (if falling or rising). 
The base functions are selected in a tree manner and are connected via a randomly chosen operator at each step. 
The set of operators is defined in 'tokens' and contains only '+' and '*' to ensure positive definiteness and finiteness. 
After initialization we select a random number on the scale of the respective PDF 
for each proposed sample point and only select those samples higher than this rn. 
This ensures the probability distribution. 
Afterwards we integrate the function and divide by the integral to get a normalized PDF.

TODO: add more weights for function selection. (Such that global monotonicity is evenly distributed)
TODO: implement support of draw from saved
TODO: MAYBE make complexity and scale also per dim, requires a bit more of index magic.
TODO: add skew sampling of scale parameter
TODO: implement maximum threshold
TODO: implement dimension swapping and mirroring, generates more data without much work
TODO: Think about a set of multidimensional base functions, also gaussian only
'''

import random
import numpy as np
import copy
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import math
import pandas as pd
from PIL import Image
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator as interpol
from scipy.integrate import simps
from scipy.spatial import KDTree
import scipy.optimize as opt
from datetime import datetime as dt
import time
import glob
import itertools

pi = np.float32(math.pi)
two = np.float32(2)
eps = 1.e-8
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

class base_functions():
    def __init__(self, scale=1, rn=0.5):
        self.scale = scale
        self.rn = rn

    def sigmoidtf(self, x):
        '''
        Input: float x
        Calculates the sigmoid function for a given x in tensorflow
        '''
        return 1.0/(1.0+tf.math.exp(-self.rn*x))

    def gauss0tf(self, x):
        sig = tf.cast(tf.math.maximum(4 * self.rn, 1), dtype=tf.float32)
        mu = np.float32(self.scale * 0.75 * self.rn)
        return tf.math.exp(-(x-mu)**2/2/sig**2)/tf.math.sqrt(2*pi*sig**2)*2*self.rn

    def gausstf(self, x):
        '''
        Input: float x
        Calculates a Gauss with mean 0.5*self.rn und std max(0.4*self.rn, 0.1) in tensorflow
        '''
        sig = tf.cast(tf.math.maximum(0.4 * self.rn, 0.1), dtype=tf.float32)
        mu = np.float32(self.scale*0.5 * self.rn)
        return tf.math.exp(-(x-mu)**2/2/sig**2)/tf.math.sqrt(2*pi*sig**2)*2*self.rn

    def gauss2tf(self, x):
        '''
        Input: float x
        Calculates a Gauss with mean 0.25*self.rn und std max(0.1*self.rn, 0.03) in tensorflow
        '''
        sig = tf.cast(tf.math.maximum(0.1 * self.rn,0.03), dtype=tf.float32)
        mu = np.float32(self.scale*0.25 * self.rn)
        return tf.math.exp(-(x-mu)**2/2/sig**2)/tf.math.sqrt(2*pi*sig**2)*2*self.rn

    def gauss3tf(self, x):
        '''
        Input: float x
        Calculates a Gauss with mean 0.75*self.rn und std max(0.1*self.rn, 0.03) in tensorflow
        '''
        sig = tf.cast(tf.math.maximum(0.1 * self.rn,0.03), dtype=tf.float32)
        mu = np.float32(self.scale*0.75 * self.rn)
        return tf.math.exp(-(x-mu)**2/2/sig**2)/tf.math.sqrt(2*pi*sig**2)*2*self.rn

    def gauss4tf(self, x):
        '''
        Input: float x
        Calculates a Gauss with mean 1*self.rn und std max(0.4*self.rn, 0.1) in tensorflow
        '''
        sig = tf.cast(tf.math.maximum(0.4 * self.rn, 0.1), dtype=tf.float32)
        mu = self.scale*self.rn
        return tf.math.exp(-(x-mu)**2/2/sig**2)/tf.math.sqrt(2*pi*sig**2)*2*self.rn

    def gauss5tf(self, x):
        sig = tf.cast(tf.math.maximum(4 * self.rn, 1), dtype=tf.float32)
        mu = np.float32(self.scale*0.5 * self.rn)
        return tf.math.exp(-(x-mu)**2/2/sig**2)/tf.math.sqrt(2*pi*sig**2)*2*self.rn

    def gauss6tf(self, x):
        sig = tf.cast(tf.math.maximum(1 * self.rn,0.3), dtype=tf.float32)
        mu = np.float32(self.scale*0.25 * self.rn)
        return tf.math.exp(-(x-mu)**2/2/sig**2)/tf.math.sqrt(2*pi*sig**2)*2*self.rn

    def gauss7tf(self, x):
        sig = tf.cast(tf.math.maximum(4 * self.rn, 1), dtype=tf.float32)
        mu = np.float32(self.scale * self.rn)
        return tf.math.exp(-(x-mu)**2/2/sig**2)/tf.math.sqrt(2*pi*sig**2)*2*self.rn

    def neg(self, x):
        '''
        Input: float x
        Calculates 1-x
        '''
        return(self.scale-x)

    def inverse(self, x):
        '''
        Input: float x
        Calculates 1/(4*x+eps)
        '''
        return np.minimum(1/(4*x+eps), 1000.0)
    
    def inversetf(self, x):
        '''
        Input: float x
        Calculates 1/(4*x+eps)
        '''
        return tf.math.minimum(1/(4*x+eps), 1000.0)

    def inverse_cuttf(self, x):
        '''
        Input: float x
        Calculates min(2*self.rn, 1/(50*x+eps)) in tensorflow
        '''
        return tf.math.minimum(2*self.rn,1/(50*x+eps))

    def inverse_cut2tf(self, x):
        '''
        Input: float x
        Calculates min(4*self.rn, 1/(50*x+eps)) in tensorflow
        '''
        return tf.math.minimum(4*self.rn,1/(50*x+eps))

    def inverse_cut3tf(self, x):
        '''
        Input: float x
        Calculates min(0.5*self.rn, 1/(50*x+eps)) in tensorflow
        '''
        return tf.math.minimum(np.float(0.5)*self.rn,1/(50*x+eps))

    def cuttf(self, x):
        '''
        Input: float x
        Calculates max(0.4*self.rn, x)  in tensorflow
        '''
        return tf.math.maximum(np.float(self.scale*0.4)*self.rn,x)

    def cut2tf(self, x):
        '''
        Input: float x
        Calculates max(0.8*self.rn, x)  in tensorflow
        '''
        return tf.math.maximum(np.float(self.scale*0.8)*self.rn,x)

    def twice(self, x):
        '''
        Input: float x
        Calculates 2*self.rn*x
        '''
        return 2*self.rn*x

    def thrice(self, x):
        '''
        Input: float x
        Calculates 3*self.rn*x
        '''
        return 3*self.rn*x

    def halftf(self, x):
        '''
        Input: float x
        Calculates x/(4*max(0.2, self.rn)) in tensorflow
        '''
        return x/(4*tf.math.maximum(np.float32(0.2),self.rn))

    def neg_square(self, x):
        '''
        Input: float x
        Calculates 1-x**2
        '''
        return self.scale*self.scale*1-x*x

    def neg_square2(self, x):
        '''
        Input: float x
        Calculates (1-x)**2
        '''
        return (self.scale-x)**2

    def potencetf(self, x):
        '''
        Input: float x
        Calculates x**self.rn in tensorflow
        '''
        return  tf.math.pow(x,self.rn)

    def potence2tf(self, x):
        '''
        Input: float x
        Calculates x**2rn in tensorflow
        '''
        return  tf.math.pow(x, 2*self.rn)

    def neg_potencetf(self, x):
        '''
        Input: float x
        Calculates 1-x**max(self.rn, 0.05) in tensorflow
        '''
        return tf.math.abs(self.scale - tf.math.pow(x,tf.math.maximum(self.rn,np.float32(0.05))))

    def neg_potence2tf(self, x):
        '''
        Input: float x
        Calculates 1-x**max(2rn, 0.05) in tensorflow
        '''
        return self.scale*self.scale - tf.math.pow(x,tf.math.maximum(np.float32(0.05),2*self.rn))

    def steptf(self, x):
        '''
        Input: float x
        Returns Step function (Output in {0,1}) with boundaries depending on self.rn in tensorflow
        '''
        set1=['0','1','2']
        set2=['3','4','5']
        set3=['6','7']
        return tf.cond(str(self.rn)[-1] in set1,
            lambda:(tf.math.maximum(x, tf.math.maximum(self.scale*self.rn, np.float32(self.scale*0.6))) - np.float32(tf.math.maximum(self.scale*self.rn, np.float32(self.scale*0.6)))) / (x-np.float(tf.math.maximum(self.scale*self.rn, np.float32(self.scale*0.6)))),
        lambda:tf.cond(str(self.rn)[-1] in set2,
            lambda:(tf.math.minimum(x, tf.math.minimum(self.scale*self.rn, np.float32(self.scale*0.4))) - np.float32(tf.math.minimum(self.scale*self.rn, np.float32(self.scale*0.4)))) / (x-np.float(tf.math.minimum(self.scale*self.rn, np.float32(self.scale*0.4)))),
        lambda:tf.cond(str(self.rn)[-1] in set3,
            lambda:tf.math.abs(((tf.math.maximum(x, np.float32(self.scale*0.25)*self.rn)-np.float32(self.scale*0.25)*self.rn)*(tf.math.minimum(x,np.float32(self.scale*0.75)*self.rn)-np.float32(self.scale*0.75)*self.rn)) / ((x-np.float32(self.scale*0.25)*self.rn)*(x-np.float32(self.scale*0.75)*self.rn))-1),
        lambda:((tf.math.maximum(x, tf.math.maximum(np.float32(self.scale*0.1),np.float32(self.scale*0.25)*self.rn))-tf.math.maximum(np.float32(self.scale*0.1),np.float32(self.scale*0.25)*self.rn))*(tf.math.minimum(x,tf.math.maximum(np.float32(self.scale*0.4),np.float32(self.scale*0.75)*self.rn))-tf.math.maximum(np.float32(self.scale*0.4),np.float32(self.scale*0.75)*self.rn)))/((x-tf.math.maximum(np.float32(self.scale*0.1),np.float32(self.scale*0.25)*self.rn))*(x-tf.math.maximum(np.float32(self.scale*0.4),np.float32(self.scale*0.75)*self.rn))))))

    def step2tf(self, x):
        '''
        Input: float x
        Returns Step function (Output in {0,1}) with boundaries depending on self.rn in tensorflow
        '''
        set1=['0','1','2']
        set2=['3','4','5']
        set3=['6','7']
        return tf.cond(str(self.rn)[-1] in set1,
            lambda:(tf.math.maximum(x, tf.math.maximum(self.scale*self.rn, np.float32(self.scale*0.5))) - np.float32(tf.math.maximum(self.scale*self.rn, np.float32(self.scale*0.5)))) / (x-np.float(tf.math.maximum(self.scale*self.rn, np.float32(self.scale*0.5)))),
        lambda:tf.cond(str(self.rn)[-1] in set2,
            lambda:(tf.math.minimum(x, tf.math.minimum(self.scale*self.rn, np.float32(self.scale*0.5))) - np.float32(tf.math.minimum(self.scale*self.rn, np.float32(self.scale*0.5)))) / (x-np.float(tf.math.minimum(self.scale*self.rn, np.float32(self.scale*0.5)))),
        lambda:tf.cond(str(self.rn)[-1] in set3,
            lambda:tf.math.abs(((tf.math.maximum(x, np.float32(self.scale*0.3)*self.rn)-np.float32(self.scale*0.3)*self.rn)*(tf.math.minimum(x,np.float32(self.scale*0.7)*self.rn)-np.float32(self.scale*0.7)*self.rn)) / ((x-np.float32(self.scale*0.3)*self.rn)*(x-np.float32(self.scale*0.7)*self.rn))-1),
        lambda:((tf.math.maximum(x, tf.math.maximum(np.float32(self.scale*0.1),np.float32(self.scale*0.2)*self.rn))-tf.math.maximum(np.float32(self.scale*0.1),np.float32(self.scale*0.2)*self.rn))*(tf.math.minimum(x,tf.math.maximum(np.float32(self.scale*0.6),np.float32(self.scale*0.75)*self.rn))-tf.math.maximum(np.float32(self.scale*0.6),np.float32(self.scale*0.75)*self.rn)))/((x-tf.math.maximum(np.float32(self.scale*0.1),np.float32(self.scale*0.2)*self.rn))*(x-tf.math.maximum(np.float32(self.scale*0.6),np.float32(self.scale*0.75)*self.rn))))))

    def sintf(self, x):
        return tf.math.sin(x) + 1

    def costf(self, x):
        return tf.math.cos(x) + 1

    def sinabstf(self, x):
        return tf.math.abs(tf.math.sin(x))

    def cosabstf(self, x):
        return tf.math.abs(tf.math.cos(x))

    def sincabstf(self, x):
        return tf.math.abs(tf.math.sin(x)/(x+eps))

    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-self.rn*x))

    def gauss0(self, x):
        sig = np.maximum(4 * self.rn, 1)
        mu = self.scale * 0.75 * self.rn
        return np.exp(-(x-mu)**2/2/sig**2)/np.sqrt(2*np.pi*sig**2)*2*self.rn   
        
    def gauss(self, x):
        sig = np.minimum(0.4 * self.rn, 0.1)
        mu = self.scale*0.5 * self.rn
        return np.exp(-(x-mu)**2/2/sig**2)/np.sqrt(2*np.pi*sig**2)*2*self.rn

    def gauss2(self, x):
        sig = np.minimum(0.1 * self.rn,0.03)
        mu = self.scale*0.25 * self.rn
        return np.exp(-(x-mu)**2/2/sig**2)/np.sqrt(2*np.pi*sig**2)*2*self.rn
        
    def gauss3(self, x):
        sig = np.minimum(0.1 * self.rn,0.03)
        mu = self.scale*0.75 * self.rn
        return np.exp(-(x-mu)**2/2/sig**2)/np.sqrt(2*np.pi*sig**2)*2*self.rn

    def gauss4(self, x):
        sig = np.minimum(0.4 * self.rn, 0.1)
        mu = self.scale*self.rn
        return np.exp(-(x-mu)**2/2/sig**2)/np.sqrt(2*np.pi*sig**2)*2*self.rn

    def gauss5(self, x):
        sig = np.minimum(4 * self.rn, 1)
        mu = self.scale*0.5 * self.rn
        return np.exp(-(x-mu)**2/2/sig**2)/np.sqrt(2*np.pi*sig**2)*2*self.rn

    def gauss6(self, x):
        sig = np.minimum(1 * self.rn,0.3)
        mu = self.scale*0.25 * self.rn
        return np.exp(-(x-mu)**2/2/sig**2)/np.sqrt(2*np.pi*sig**2)*2*self.rn

    def gauss7(self, x):
        sig = np.minimum(4 * self.rn, 1)
        mu = self.scale * self.rn
        return np.exp(-(x-mu)**2/2/sig**2)/np.sqrt(2*np.pi*sig**2)*2*self.rn
        
    def inverse_cut(self, x):
        return np.minimum(2*self.rn,1./(50*x+eps))

    def inverse_cut2(self, x):
        return np.minimum(4*self.rn,1./(50*x+eps))

    def inverse_cut3(self, x):
        return np.minimum(0.5*self.rn,1./(50*x+eps))

    def cut(self, x):
        return np.maximum(self.scale*0.4*self.rn,x)

    def cut2(self, x):
        return np.maximum(self.scale*0.8*self.rn,x)
        
    def cuttf(self, x):
        return tf.math.maximum(self.scale*0.4*self.rn,x)

    def cut2tf(self, x):
        return tf.math.maximum(self.scale*0.8*self.rn,x)

    def half(self, x):
        return x/(4.*np.maximum(0.2,self.rn))
    
    def potence(self, x):
        return np.power(x,self.rn)
        
    def potence2(self, x):
        return np.power(x,2*self.rn)
        
    def neg_potence(self, x):
        return np.absolute(self.scale-np.power(x,np.maximum(self.rn,0.05)))

    def neg_potence2(self, x):
        return self.scale*self.scale - np.power(x,np.maximum(0.05,2*self.rn))
         
    def step(self, x):
        set1=['0','1','2']
        set2=['3','4','5']
        set3=['6','7']
        if str(self.rn)[-1] in set1:
            return 1 if x > np.maximum(self.scale*self.rn,self.scale*0.5) else 0
        elif str(self.rn)[-1] in set2:
            return 1 if x < np.minimum(self.scale*self.rn, self.scale*0.5) else 0
        elif str(self.rn)[-1] in set3:
            return 1 if x < self.scale*0.3*self.rn or x > self.scale*0.7*self.rn else 0
        else:
            return 1 if x > np.maximum(self.scale*0.1,self.scale*0.2*self.rn) and x < np.maximum(self.scale*0.6,self.scale*0.75*self.rn) else 0

    def step2(self, x):
        set1=['0','1','2']
        set2=['3','4','5']
        set3=['6','7']
        if str(self.rn)[-1] in set1:
            return 1 if x > np.maximum(self.scale*self.rn,self.scale*0.6) else 0
        elif str(self.rn)[-1] in set2:
            return 1 if x < np.minimum(self.scale*self.rn, self.scale*0.4) else 0
        elif str(self.rn)[-1] in set3:
            return 1 if x < self.scale*0.25*self.rn or x > self.scale*0.75*self.rn else 0
        else:
            return 1 if x > np.maximum(self.scale*0.1,self.scale*0.25*self.rn) and x < np.maximum(self.scale*0.4,self.scale*0.75*self.rn) else 0

    def sin(self, x):
        return np.sin(x) + 1

    def cos(self, x):
        return np.cos(x) + 1
        
    def sinabs(self, x):
        return np.abs(np.sin(x))    

    def cosabs(self, x):
        return np.abs(np.cos(x))

    def sincabs(self, x):
        return np.abs(np.sin(x)/(x+eps))

class locality_test_functions():
    def linear1(self, x):
        '''
        returns linear function x/2, scale must be 2, f(2)=1
        '''
        return x/2

    def linear2(self, x):
        '''
        returns linear function 2x, scale must be 1, f(0.5)=1
        '''
        return 2*x

    def sin1(self, x):
        '''
        returns sine which is 0 for x>pi/2, scale should be 2
        '''
        if x < pi/2:
            return np.sin(x)
        else:
            return 0

    def sin1tf (self, x):
        '''
        returns sine which is 0 for x>pi/2, scale should be 2
        '''
        #return tf.cond(x < tf.cast(pi/2, tf.float32),
        #                lambda x: tf.math.sin(x),
        #                lambda: tf.cast(0, tf.float32))
        return tf.math.sin(tf.minimum(x, tf.cast(pi/2, tf.float32))) - (0.5 + 0.5*tf.math.sign(tf.minimum(x, tf.cast(pi/2, tf.float32))-tf.cast(pi/2, tf.float32)+tf.cast(1e-8, tf.float32)))

    def sin2(self, x):
        '''
        returns sine which is 0 for 2pi/3 < x and x < pi/3, scale should be 3
        '''
        if x > pi/3 and x < 2*pi/3:
            return np.sin(x)
        else:
            return 0

    def sin2tf (self, x):
        '''
        returns sine which is 0 for 2pi/3 < x and x < pi/3, scale should be 3
        '''
        low = tf.cast(pi/3, tf.float32)
        hig = tf.cast(2*pi/3, tf.float32)
        #return tf.cond(x < tf.cast(pi/3, tf.float32),
        #                lambda: tf.cast(0, tf.float32),
        #                tf.cond(x > tf.cast(2*pi/3, tf.float32),
        #                        lambda: tf.cast(0, tf.float32),
        #                        tf.math.sin(x)))
        #                        
        #return tf.math.sin(tf.minimum(tf.maximum(pi/3, x), 2*pi/3)) - (0.5-0.5*tf.math.sign(tf.maximum(pi/3, x)-pi/3-1e-8))*tf.math.sin(pi/3)-(0.5+0.5*tf.math.sign(tf.minimum(2*pi/3, x)-2*pi/3+1e-8))*tf.math.sin(2*pi/3)
        return tf.math.sin(tf.minimum(tf.maximum(low, x), hig)) - (0.5-0.5*tf.math.sign(tf.maximum(low, x)-low-1e-8))*tf.math.sin(low)-(0.5+0.5*tf.math.sign(tf.minimum(hig, x)-hig+1e-8))*tf.math.sin(hig)

    def sin3(self, x):
        '''
        returns sine+2 for 3*pi/2 - y < x < 3*pi/2 + y, y=pi/6.52326761054738, scale should be 6
        '''
        low = 3*pi/2 - pi/6.52326761054738
        hig = 3*pi/2 + pi/6.52326761054738

        if x > low and x < hig:
            return np.sin(x)+2
        else:
            return 0

    def sin3tf (self, x):
        '''
        returns sine+2 for 3*pi/2 - y < x < 3*pi/2 + y, y=pi/6.52326761054738, scale should be 6
        '''
        low = tf.cast(3*pi/2 - pi/6.52326761054738, tf.float32)
        hig = tf.cast(3*pi/2 + pi/6.52326761054738, tf.float32)

        def func_(x):
            return tf.math.sin(x) + 2
        #zero = tf.cast(0, tf.float32)
        #return tf.cond(x < low,
        #                lambda: zero,
        #                tf.cond(x > hig,
        #                        lambda: zero,
        #                        tf.math.sin(x) + tf.cast(2, tf.float32)))

        return func_(tf.minimum(tf.maximum(low, x), hig)) - (0.5-0.5*tf.math.sign(tf.maximum(low, x)-low-1e-8))*func_(low)-(0.5+0.5*tf.math.sign(tf.minimum(hig, x)-hig+1e-8))*func_(hig)
         

    def step(self, x):
        return 1 if x > 0.5 and x < 1.5 else 0

    def steptf(self, x):
        '''
        Input: float x, scale should be 2
        Returns Step function (Output in {0,1}), 1 ix 0.5 < x < 1.5
        '''
        a = tf.cast(0.5, tf.float32)
        b = tf.cast(1.5, tf.float32)
        return ((tf.math.maximum(x, a) - a) * (tf.math.minimum(x, b) - b)) / ((x - a) * (x - b))
     
    def gauss(self, x):
        '''
        Input: float x
        Calculates a Gauss on scale 30 with mean 15 und std 1/sqrt(2pi)
        '''
        sig = 1 / np.sqrt(2*pi)
        mu = 15
        return np.exp(-(x-mu)**2/2/sig**2)

    def gausstf(self, x):
        '''
        Input: float x
        Calculates a Gauss on scale 30 with mean 15 und std 1/sqrt(2pi)
        '''
        sig = tf.cast(1/tf.math.sqrt(2*pi), tf.float32)
        mu = tf.cast(15, tf.float32)
        return tf.math.exp(-(x-mu)**2/2/sig**2)

    def potence1(self, x):
        '''
        Input: float x
        Calculates x**2 if x < 3**1/3, 0 else
        '''
        if x <= 3**(1./3):
            return x**2
        else:
            return np.float32(0)
   
    def potence1tf(self, x):
        '''
        Input: float x
        Calculates x**2 if x < 3**1/3, 0 else
        '''
        hig = tf.cast(3**(1./3), tf.float32)

        #return  tf.cond(x <= tf.cast(3**(1./3), tf.float32), 
        #                lambda x: tf.math.pow(x, 2),
        #                lambda: tf.cast(0, tf.float32))
        return tf.math.square(tf.minimum(x, hig)) - (0.5 + 0.5*tf.math.sign(tf.minimum(x, hig)-hig+1e-8))*tf.math.square(hig)
        
    
    def potence2(self, x):
        '''
        Input: float x
        Calculates x**2 / 3 if x < 9**1/3, 0 else
        '''
        if x <= 9**(1./3):
            return x**2 / 3
        else:
            return np.float32(0)
   
    def potence2tf(self, x):
        '''
        Input: float x
        Calculates x**2 / 3 if x < 9**1/3, 0 else
        '''
        def func_(x):
            return tf.math.square(x)/tf.cast(3, tf.float32)

        hig = tf.cast(9**(1./3), tf.float32)

        #return  tf.cond(x <= tf.cast(9**(1./3), tf.float32), 
        #                tf.math.pow(x, 2) / 3,
        #                lambda: tf.cast(0, tf.float32))
        return func_(tf.minimum(x, hig)) - (0.5 + 0.5*tf.math.sign(tf.minimum(x, hig)-hig+1e-8))*func_(hig)

def process_bar(point, number_of_items, carriage_return=True):
    '''
    Prints a progress bar along percentage and number of items.

    Args: 
            point:           INT, current iteration
            number_of_items: INT, the total number of items ofǘer which is iterated.
            carriage_return: BOOL, whether or not to use carriage_return.
    '''
    point = point + 1
    relative_progress = point/number_of_items
    if point == number_of_items:
        carriage_return = False
    increment = int(relative_progress * 20)
    process_string = str(int(relative_progress*100)) + '% [' + '=' * increment + ' ' * (20 - increment) + ']' + ' ' + str(point) + '/' + str(number_of_items)
    if carriage_return:
        print(process_string, end='\r', flush=True)
    else:
        print(process_string, flush=True)

class utils():
    def __init__(self, scale=1.0, rn=0.5):
        self.scale = scale
        self.rn = rn

    def get_operator(self, op):
            return {
            '+' : np.add,
            '*' : np.multiply
            }[op]

    def get_operatortf(self, op):
        '''
        Input: String indicator of the operator, one of ['+', '*']
        Returns: respective tensorflow operator
        '''
        return {
        '+' : tf.math.add,
        '*' : tf.math.multiply
        }[op]   

    def get_base_functiontf(self, func):
        return{
            'x'     	        : tf.math.abs,                                          
            'x**2'  	        : tf.math.square,                                          
            'sqrt(x)'	        : tf.math.sqrt,                                             
            'sin(x)'   	        : base_functions(scale=self.scale, rn=self.rn).sintf,          
            'cos(x)'   	        : base_functions(scale=self.scale, rn=self.rn).costf,          
            'sinabs(x)'         : base_functions(scale=self.scale, rn=self.rn).sinabstf,       
            'sincabs(x)'        : base_functions(scale=self.scale, rn=self.rn).sincabstf,      
            'cosabs(x)'         : base_functions(scale=self.scale, rn=self.rn).cosabstf,       
            'sigmoid(x)'	    : base_functions(scale=self.scale, rn=self.rn).sigmoidtf,      
            'gauss(x)'	        : base_functions(scale=self.scale, rn=self.rn).gausstf,        
            'inverse(x)'        : base_functions(scale=self.scale, rn=self.rn).inversetf,        
            'gauss0(x)'	        : base_functions(scale=self.scale, rn=self.rn).gauss0tf,       
            'gauss2(x)'	        : base_functions(scale=self.scale, rn=self.rn).gauss2tf,       
            'gauss3(x)'	        : base_functions(scale=self.scale, rn=self.rn).gauss3tf,       
            'gauss4(x)'	        : base_functions(scale=self.scale, rn=self.rn).gauss4tf,       
            'gauss5(x)'	        : base_functions(scale=self.scale, rn=self.rn).gauss5tf,       
            'gauss6(x)'	        : base_functions(scale=self.scale, rn=self.rn).gauss6tf,       
            'gauss7(x)'	        : base_functions(scale=self.scale, rn=self.rn).gauss7tf,       
            'neg(x)'	        : base_functions(scale=self.scale, rn=self.rn).neg,            
            'inverse_cut(x)'    : base_functions(scale=self.scale, rn=self.rn).inverse_cuttf,  
            'inverse_cut2(x)'	: base_functions(scale=self.scale, rn=self.rn).inverse_cut2tf, 
            'inverse_cut3(x)'	: base_functions(scale=self.scale, rn=self.rn).inverse_cut3tf, 
            'cut(x)'	        : base_functions(scale=self.scale, rn=self.rn).cuttf,          
            'cut2(x)'           : base_functions(scale=self.scale, rn=self.rn).cut2tf,         
            'twice(x)'	        : base_functions(scale=self.scale, rn=self.rn).twice,          
            'thrice(x)'	        : base_functions(scale=self.scale, rn=self.rn).thrice,         
            'half(x)'	        : base_functions(scale=self.scale, rn=self.rn).halftf,         
            'neg_square(x)'     : base_functions(scale=self.scale, rn=self.rn).neg_square,     
            'neg_square2(x)'    : base_functions(scale=self.scale, rn=self.rn).neg_square2,    
            'potence(x)'	    : base_functions(scale=self.scale, rn=self.rn).potencetf,      
            'potence2(x)'	    : base_functions(scale=self.scale, rn=self.rn).potence2tf,     
            'neg_potence(x)'    : base_functions(scale=self.scale, rn=self.rn).neg_potencetf,             
            'neg_potence2(x)'   : base_functions(scale=self.scale, rn=self.rn).neg_potence2tf, 
            'step(x)'           : base_functions(scale=self.scale, rn=self.rn).steptf,         
            'step2(x)'          : base_functions(scale=self.scale, rn=self.rn).step2tf         
            }[func]
       
    def get_base_function(self, func):
        return{
            'x'                 : np.nan_to_num,
            'x**2'              : np.square,
            'sqrt(x)'           : np.sqrt,
            'sin(x)'   	        : base_functions(scale=self.scale, rn=self.rn).sin,
            'cos(x)'   	        : base_functions(scale=self.scale, rn=self.rn).cos,
            'sinabs(x)'         : base_functions(scale=self.scale, rn=self.rn).sinabs,
            'sincabs(x)'        : base_functions(scale=self.scale, rn=self.rn).sincabs,
            'cosabs(x)'         : base_functions(scale=self.scale, rn=self.rn).cosabs,
            'sigmoid(x)'        : base_functions(scale=self.scale, rn=self.rn).sigmoid,
            'gauss(x)'          : base_functions(scale=self.scale, rn=self.rn).gauss,
            'inverse(x)'        : base_functions(scale=self.scale, rn=self.rn).inverse,
            'gauss0(x)'         : base_functions(scale=self.scale, rn=self.rn).gauss0,
            'gauss2(x)'         : base_functions(scale=self.scale, rn=self.rn).gauss2,
            'gauss3(x)'         : base_functions(scale=self.scale, rn=self.rn).gauss3,
            'gauss4(x)'         : base_functions(scale=self.scale, rn=self.rn).gauss4,
            'gauss5(x)'         : base_functions(scale=self.scale, rn=self.rn).gauss5,
            'gauss6(x)'         : base_functions(scale=self.scale, rn=self.rn).gauss6,
            'gauss7(x)'         : base_functions(scale=self.scale, rn=self.rn).gauss7,
            'neg(x)'            : base_functions(scale=self.scale, rn=self.rn).neg,
            'inverse_cut(x)'    : base_functions(scale=self.scale, rn=self.rn).inverse_cut,
            'inverse_cut2(x)'   : base_functions(scale=self.scale, rn=self.rn).inverse_cut2,
            'inverse_cut3(x)'   : base_functions(scale=self.scale, rn=self.rn).inverse_cut3,
            'cut(x)'            : base_functions(scale=self.scale, rn=self.rn).cut,
            'cut2(x)'           : base_functions(scale=self.scale, rn=self.rn).cut2,
            'twice(x)'          : base_functions(scale=self.scale, rn=self.rn).twice,
            'thrice(x)'         : base_functions(scale=self.scale, rn=self.rn).thrice,
            'half(x)'           : base_functions(scale=self.scale, rn=self.rn).half,
            'neg_square(x)'     : base_functions(scale=self.scale, rn=self.rn).neg_square,
            'neg_square2(x)'    : base_functions(scale=self.scale, rn=self.rn).neg_square2,
            'potence(x)'        : base_functions(scale=self.scale, rn=self.rn).potence,
            'potence2(x)'       : base_functions(scale=self.scale, rn=self.rn).potence2,
            'neg_potence(x)'    : base_functions(scale=self.scale, rn=self.rn).neg_potence,
            'neg_potence2(x)'   : base_functions(scale=self.scale, rn=self.rn).neg_potence2,
            'step(x)'           : base_functions(scale=self.scale, rn=self.rn).step,
            'step2(x)'          : base_functions(scale=self.scale, rn=self.rn).step2
            }[func]

    def get_single_max(self, func):
        return{
            'x'                 : self.scale,
            'x**2'              : self.scale**2,
            'sqrt(x)'           : np.sqrt(self.scale),
            'sin(x)'   	        : 2.0,
            'cos(x)'   	        : 2.0,
            'sinabs(x)'         : 1.0,
            'sincabs(x)'        : 1.0,
            'cosabs(x)'         : 1.0,
            'sigmoid(x)'        : 1.0,
            'gauss(x)'          : 1.0/(max(0.4 * self.rn, 0.1)*np.sqrt(2*pi)) * 2 * self.rn, # max of gauss is 1/(sigma*sqrt(pi))
            'inverse(x)'        : 1000.0,
            'gauss0(x)'         : 1.0/(max(4 * self.rn, 1)*np.sqrt(2*pi)) * 2 * self.rn,
            'gauss2(x)'         : 1.0/(max(0.1 * self.rn,0.03)*np.sqrt(2*pi)) * 2 * self.rn,
            'gauss3(x)'         : 1.0/(max(0.1 * self.rn,0.03)*np.sqrt(2*pi)) * 2 * self.rn,
            'gauss4(x)'         : 1.0/(max(0.4 * self.rn, 0.1)*np.sqrt(2*pi)) * 2 * self.rn,
            'gauss5(x)'         : 1.0/(max(4 * self.rn, 1)*np.sqrt(2*pi)) * 2 * self.rn,
            'gauss6(x)'         : 1.0/(max(1 * self.rn,0.3)*np.sqrt(2*pi)) * 2 * self.rn,
            'gauss7(x)'         : 1.0/(max(4 * self.rn, 1)*np.sqrt(2*pi)) * 2 * self.rn,
            'neg(x)'            : self.scale,
            'inverse_cut(x)'    : 2 * self.rn,
            'inverse_cut2(x)'   : 4 * self.rn,
            'inverse_cut3(x)'   : 0.5 * self.rn,
            'cut(x)'            : self.scale,
            'cut2(x)'           : self.scale,
            'twice(x)'          : 2 * self.rn * self.scale,
            'thrice(x)'         : 3 * self.rn * self.scale,
            'half(x)'           : self.scale / (4. * max(0.2, self.rn)),
            'neg_square(x)'     : self.scale**2,
            'neg_square2(x)'    : self.scale**2,
            'potence(x)'        : np.power(self.scale, self.rn),
            'potence2(x)'       : np.power(self.scale, 2 * self.rn),
            'neg_potence(x)'    : self.scale,
            'neg_potence2(x)'   : self.scale**2,
            'step(x)'           : 1.0,
            'step2(x)'          : 1.0
            }[func]

    def get_locality_base_functiontf(self, func):
        return{                                            
            'sin1(x)'   	        : locality_test_functions().sin1tf,          
            'sin2(x)'   	        : locality_test_functions().sin2tf,          
            'sin3(x)'   	        : locality_test_functions().sin3tf,          
            'linear1(x)'   	        : locality_test_functions().linear1,          
            'linear2(x)'   	        : locality_test_functions().linear2,          
            'step(x)'   	        : locality_test_functions().steptf,          
            'gauss(x)'   	        : locality_test_functions().gausstf,          
            'potence1(x)'   	    : locality_test_functions().potence1tf,          
            'potence2(x)'   	    : locality_test_functions().potence2tf         
            }[func]
       
    def get_locality_base_function(self, func):
        return{
            'sin1(x)'   	        : locality_test_functions().sin1,          
            'sin2(x)'   	        : locality_test_functions().sin2,          
            'sin3(x)'   	        : locality_test_functions().sin3,          
            'linear1(x)'   	        : locality_test_functions().linear1,          
            'linear2(x)'   	        : locality_test_functions().linear2,          
            'step(x)'   	        : locality_test_functions().step,          
            'gauss(x)'   	        : locality_test_functions().gauss,          
            'potence1(x)'   	    : locality_test_functions().potence1,          
            'potence2(x)'   	    : locality_test_functions().potence2      
            }[func]

    def get_locality_single_max(self, func):
        return{
            'sin1(x)'   	        : 1.0,      
            'sin2(x)'   	        : 1.0,      
            'sin3(x)'   	        : 1.2,      
            'linear1(x)'   	        : 1.0,         
            'linear2(x)'   	        : 2.0,         
            'step(x)'   	        : 1.0,      
            'gauss(x)'   	        : 1.0,    
            'potence1(x)'   	    : 2.1,          
            'potence2(x)'   	    : 1.5 
            }[func]

class image_to_pdf():
    '''
    Takes a single image and turns it into a pdf for which it creates a sample with size.

    Args:
        size:       INT, number of samples drawn
        image_path: STRING, path to the image to process
        verbose:    BOOL, adds some progress output
    '''
    def __init__(self, size=1000, image_path='data/image.png', verbose=False):
        self.size = size
        self.path = image_path
        self.verbose = verbose

    def find_value(self, test, data_, raw_data, tree, height, width):
        test = np.asarray(test)
        testn = np.zeros(np.shape(test))
        testn[:, 0] = test[:, 0] * (height - 1)
        testn[:, 1] = test[:, 1] * (width - 1)
        # _, idx = tree.query(test)
        idx = tree.query_ball_point(testn, 1)
        values = []
        for i in range(len(idx)):
            if self.verbose: process_bar(i, len(idx), carriage_return=False) 
            positions = np.asarray(data_[idx[i]])
            positions = np.transpose(positions)
            positions = tuple(positions)
            values.append(np.mean(raw_data[positions]))
        return values

    def get_value(self, xpos, ypos, tree, data_, raw_data, height, width):
        xpos, ypos = np.asarray(
            xpos) * (height - 1), np.asarray(ypos) * (width - 1)
        # _, idx = tree.query(test)
        test = np.squeeze(np.dstack((xpos, ypos)))
        idx = tree.query_ball_point(test, 1)
        values = []
        for i in range(len(idx)):
            positions = np.asarray(data_[idx[i]])
            positions = np.transpose(positions)
            positions = tuple(positions)

            values.append(np.mean(raw_data[positions]))
        return np.asarray(values)

    def integration(self, x, y, z, tree, data_, raw_data, height, width, intits=int(1e5)):
        xmin = min(x)
        ymin = min(y)
        zmin = 0.
        xmax = max(x)
        ymax = max(y)
        zmax = max(z)
        xrange_ = (xmax - xmin)
        yrange_ = (ymax - ymin)
        zrange_ = (zmax - zmin)
        volume = xrange_ * yrange_ * zrange_

        def compute_integral(n_samples):
            xsamp = np.random.uniform(xmin, xmax, (n_samples,))
            ysamp = np.random.uniform(ymin, ymax, (n_samples,))
            zsamp = np.random.uniform(zmin, zmax, (n_samples,))
            value = self.get_value(xsamp, ysamp, tree, data_, raw_data, height, width)
            c_in_region = zsamp < value
            c_in_region = np.squeeze(c_in_region) * 1
            count = sum(c_in_region)
            return float(count) / n_samples

        result = compute_integral(intits) * volume
        return result

    def generation(self):     
        img = Image.open(self.path).convert('L')
        raw_data = img.getdata()
        width, height = img.size
        if self.verbose: print('image height and width', height, width)
        raw_data = np.reshape(raw_data, (height, width))
        if self.verbose: print('pixel range goes from {} to {}'.format(np.min(raw_data), np.max(raw_data)))
        max_value = np.max(raw_data)

        filename = os.path.basename(self.path)
        name = filename[:-4]

        # build a tree to find pixel_values at random positions
        x_, y_ = np.meshgrid(np.arange(height), np.arange(width))
        data_ = np.asarray(list(zip(x_.ravel(), y_.ravel())))
        if self.verbose: print('shape of the data to prcess in KDTree', np.shape(data_))
        tree = KDTree(data_)

        x = []
        y = []
        z = []
        # importance sample of 'size' points for the given sample
        points = []
        sample_size = self.size * 6
        while len(points) < self.size:
            test = np.random.rand(sample_size, 2)
            values = self.find_value(test, data_, raw_data, tree, height, width)
            ran = np.random.rand(sample_size) * max_value
            truth = ran < values
            points = test[truth]
            sample_size *= 2
        values = np.asarray(values)[truth]
        points, values = points[:self.size], values[:self.size]
        x.append(points[:, 0])
        y.append(points[:, 1])
        z.append(values)

        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        if self.verbose: print(f'resulting shapes: shape(x)={np.shape(x)}, shape(y)={np.shape(y)}, shape(z)={np.shape(z)}')

        if self.verbose: print('integration')
        integral = self.integration(x[0], y[0], z[0], tree, data_, raw_data, height, width)
        z[0] = np.true_divide(z[0], integral)


        save_file = {'x': x, 'y': y, 'z': z}
        dirname = f'new_data/images/{name}_size_{self.size}'
        try:
            os.makedirs(dirname)
        except:
            pass
        pickle.dump(save_file, open(f'{dirname}/{name}_size_{self.size}.p', 'wb'))
        if self.verbose: print('saved to: ', f'{dirname}/{name}_size_{self.size}.p')

class Prob_dist_from_1D():
    '''
    Generates a probability density out of arbitrary 1-dimensional data and draws a sample distribution with size SIZE out of it.
    The probability values at arbitrary points are interpolated from the surrounding values by linear interpolation.
    Normalizes the domain to range [0,1]
    Dumps a list per sample in a directory, containing samples or [samples, grid_samples], where each is of size [n_dim, n_samples], n_dim = 2

    Args:
        size: INT, size of the drawn sample distribution
        data: array-like, list or array of data points (float or int), structures as [n_functions, n_points, dim], where dim = 2 for x and f(x). Required if readtxt==False
        with_grid: BOOL, if True a grid with the corresponding true probability values is also generated (for nicer plots)
        grid_number: INT, number of samples in the grid
        readtxt: BOOL, if True the dat is read from txt files in data_dir. Then data is expected to be tabulated in txt/csv files with 1 sample point per row structured x,y (float or int).  
        data_dir: STRING, Directory with the data, only required if readtxt == True. data_dir is directly searched with glob.glob, thus '/*' is appended to the directory, if not already there. 
                          If subdirectories have to be searched, give path as 'path/to/data/*/*', where the last asterisk is the directory of readable datafiles.
        exclude_short: BOOL, excludes files shorter than 100 entries.
        savedir: STRING, directory for the generated pdfs. will be appended by /"size" and _with_grid if with_grid==True

    '''
    def __init__(self, size, data=None, with_grid=False, grid_number=10000, verbose=False, readtxt=False, exclude_short=False, data_dir=None, savedir='new_data/real_1d'):
        self.data = data
        self.data_dir = data_dir
        self.size = size
        self.with_grid = with_grid
        self.grid_number = grid_number
        self.verbose = verbose
        self.readtxt = readtxt
        self.exclude_short = exclude_short
        self.savedir = os.path.join(savedir, '{}'.format(size))
        if self.with_grid:
            self.savedir = self.savedir + '_with_grid'
        try: os.makedirs(self.savedir)
        except: pass


    def find_nearest(self, array,value):
        idx = np.searchsorted(array, value, side="left")
        return array[idx-1], array[idx], idx-1, idx

    def get_values(self, x, y, point):
        lower, upper, lower_idx, upper_idx = self.find_nearest(x, point)
        value = y[lower_idx] + (y[upper_idx]-y[lower_idx])*(point-lower)/(upper-lower)
        return value

    def integration(self, x, y):    
        return integrate.simps(y,x)

    def get_pdf(self):

        count = 0
        if self.readtxt:
            if not self.data_dir.endswith('*'):
                os.path.join(self.data_dir, '*')
            file_list = glob.glob(self.data_dir)
            iteration = len(file_list)
        else:                
            if len(np.shape(self.data)) == 2:
                self.data = np.expand_dims(self.data, 0)
            iteration = len(self.data)

        for i in range(iteration):
            if self.verbose: process_bar(i, iteration, carriage_return=False) 
            
            if self.readtxt:
                fname = file_list[i]
                try:
                    raw_data = pd.read_csv(fname, sep=',', header=0).values
                except: continue
            else:
                raw_data = self.data[i]

            raw_data = np.array(raw_data)
            x = np.float32(raw_data[:,0])
            y = np.float32(raw_data[:,1])

            if len(x) < 100: continue


            # normalize the range to 0,1
            x = x - min(x) 
            x = x / max(x)

            # normalize the function to give it pdf nature
            integral = self.integration(x,y)
            y = y / integral

            save_name = f'{count}.p'
            count += 1

            y_max = max(y)
            sample_size = self.size * 6
            points = []
            values = []
            while len(points) < self.size:
                test = np.random.rand(sample_size)
                vals = self.get_values(x, y, test)
                ran = np.random.rand(sample_size) * y_max
                truth = ran < vals
                test = test[truth]
                vals = vals[truth]
                [points.append(t) for t in test]
                [values.append(t) for t in vals]
                if sample_size < 2500000:
                    sample_size *= 2
                
            points, values = np.asarray(points), np.asarray(values)
            points, values = points[:self.size], values[:self.size]
            
            sample = [points, values]

            if self.with_grid:
                x_grid = np.linspace(0, 1, self.grid_number)
                y_grid = self.get_values(x, y, x_grid)
                sample_grid = [x_grid, y_grid]

            if self.with_grid:
                save_file = [sample, sample_grid]
            else:
                save_file = sample

            pickle.dump(save_file, open(os.path.join(self.savedir, save_name), 'wb'))

class Prob_dist_from_2D():
    '''
    Generates a probability density out of arbitrary 2-dimensional data and draws a sample distribution with size SIZE out of it.
    The probability values at arbitrary points are interpolated from the surrounding values by linear interpolation.
    Normalizes the domain to range [0,1]^2
    Dumps a list per sample in a directory, containing samples or [samples, grid_samples], where each is of size [n_dim, n_samples], n_dim = 3

    Args:
        size: INT, size of the drawn sample distribution
        data: array-like, list or array of data points (float or int), structures as [n_functions, height, width]. Required if readimg==False
        with_grid: BOOL, if True a grid with the corresponding true probability values is also generated (for nicer plots)
        grid_number: INT, number of samples in the grid
        imagenorm: INT, normalization constant for images. Has to be adapted if image files or data aer not 8bit (maxvalue of 255)
        readimg: BOOL, if True the dat is read from txt files in data_dir. Then data is expected to be imagefiles.  
        data_dir: STRING, Directory with the data, only required if readtxt == True. data_dir is directly searched with glob.glob, thus '/*' is appended to the directory, if not already there. 
                          If subdirectories have to be searched, give path as 'path/to/data/*/*', where the last asterisk is the directory of readable datafiles.
        savedir: STRING, directory for the generated pdfs. will be appended by /"size" and _with_grid if with_grid==True

    '''
    def __init__(self, size, data=None, with_grid=False, grid_number=500, verbose=False, data_dir=None, readimg=False, savedir='new_data/real_2d', imagenorm=255):
        self.data = data
        self.data_dir = data_dir
        self.size = size
        self.with_grid = with_grid
        self.grid_number = grid_number
        self.verbose = verbose
        self.readimg = readimg
        self.imagenorm = imagenorm
        self.savedir = os.path.join(savedir, '{}'.format(size))
        if self.with_grid:
            self.savedir = self.savedir + '_with_grid'
        try: os.makedirs(self.savedir)
        except: pass


    def integration(self, x, y, z):    
        return integrate.simps([integrate.simps(zz_x,x) for zz_x in z], y)

    def get_pdf(self):

        count = 0
        if self.readimg:
            if not self.data_dir.endswith('*'):
                self.data_dir = os.path.join(self.data_dir, '*')
            file_list = glob.glob(self.data_dir)
            iteration = len(file_list)
        else:                
            if len(np.shape(self.data)) == 2:
                self.data = np.expand_dims(self.data, 0)
            iteration = len(self.data)

        for i in range(iteration):
            if self.verbose: process_bar(i, iteration, carriage_return=False) 
            
            if self.readimg:
                fname = file_list[i]
                try:
                    y = Image.open(fname).convert('L')
                    y = np.asarray(y)
                except: continue
            else:
                y = self.data[i]

            height, width = np.shape(y)
            row = np.arange(width) / (width-1)
            col = np.arange(height) / (height-1)

            y = y / self.imagenorm

            integral = self.integration(row, col, y)
            y = y / integral

            f = interpol((col, row), y)

            save_name = f'{count}.p'
            count += 1

            y_max = np.max(y)
            sample_size = self.size * 6
            points = []
            values = []
            while len(points) < self.size:
                test = np.random.rand(sample_size, 2)
                vals = f(test)
                ran = np.random.rand(sample_size) * y_max
                truth = ran < vals
                test = test[truth]
                vals = vals[truth]
                [points.append(t) for t in test]
                [values.append(t) for t in vals]
                if sample_size < 2500000:
                    sample_size *= 2
                
            points, values = np.asarray(points), np.asarray(values)
            points, values = points[:self.size], values[:self.size]
            points = np.transpose(points)
            sample = np.concatenate((points, np.expand_dims(values, 0)))
            

            if self.with_grid:
                x_grid = np.linspace(0, 1, self.grid_number)
                y_grid = np.linspace(0, 1, self.grid_number)
                grid = np.array(list(itertools.product(x_grid, y_grid)))
                z_grid = f(grid)
                grid = np.transpose(grid)
                sample_grid = np.concatenate((grid, np.expand_dims(z_grid, 0)))

            if self.with_grid:
                save_file = [sample, sample_grid]
            else:
                save_file = sample

            pickle.dump(save_file, open(os.path.join(self.savedir, save_name), 'wb'))

class Prob_dist_from_3D():
    '''
    Generates a probability density out of arbitrary 3-dimensional data and draws a sample distribution with size SIZE out of it.
    The probability values at arbitrary points are interpolated from the surrounding values by linear interpolation.
    Normalizes the domain to range [0,1]^3
    Dumps a list per sample in a directory, containing samples or [samples, grid_samples], where each is of size [n_dim, n_samples], n_dim = 3

    Args:
        size: INT, size of the drawn sample distribution
        data: array-like, list or array of data points (float or int), structures as [n_functions, n_layers, height, width]. Required if readimg==False
        with_grid: BOOL, if True a grid with the corresponding true probability values is also generated (for nicer plots)
        grid_number: INT, number of samples in the grid
        readimg: BOOL, if True the data is read from txt files in data_dir. Then data is expected to be imagefiles. where each volume is contained in a subdirectory with an image per layer.  
        data_dir: STRING, Directory with the data, only required if readtxt == True. data_dir is directly searched with glob.glob, thus '/*' is appended to the directory, if not already there. 
                          If subdirectories have to be searched, give path as 'path/to/data/*/*', where the last asterisk is the directory of readable datafiles.
        savedir: STRING, directory for the generated pdfs. will be appended by /"size" and _with_grid if with_grid==True

    '''
    def __init__(self, size, data=None, with_grid=False, grid_number=100, verbose=False, data_dir=None, readimg=False, savedir='new_data/real_3d'):
        self.data = data
        self.data_dir = data_dir
        self.size = size
        self.with_grid = with_grid
        self.grid_number = grid_number
        self.verbose = verbose
        self.readimg = readimg
        self.savedir = os.path.join(savedir, '{}'.format(size))
        if self.with_grid:
            self.savedir = self.savedir + '_with_grid'
        try: os.makedirs(self.savedir)
        except: pass

    def integration(self, z, x, y, p):
        integrand = []
        for p_xy in p:
            integrand.append(integrate.simps([integrate.simps(p_x, x) for p_x in p_xy], y))
        return integrate.simps(integrand, z)

    def get_pdf(self):

        count = 0
        if self.readimg:
            if not self.data_dir.endswith('*'):
                self.data_dir = os.path.join(self.data_dir, '*')
            dir_list = glob.glob(self.data_dir)
            iteration = len(dir_list)
        else:                
            if len(np.shape(self.data)) == 2:
                self.data = np.expand_dims(self.data, 0)
            iteration = len(self.data)

        for i in range(iteration):
            if self.verbose: process_bar(i, iteration, carriage_return=False) 
            
            if self.readimg:
                directory = dir_list[i]
                try:
                    y = []
                    for fname in glob.glob('{}/*.png'.format(directory)):
                        img = Image.open(fname)
                        img = np.asarray(img)
                        y.append(img)
                except: continue
            else:
                y = self.data[i]

            depth, height, width = np.shape(y)
            row = np.arange(width) / (width-1)
            col = np.arange(height) / (height-1)
            channel = np.arange(depth) / (depth-1)

            y_min, y_max = np.min(y), np.max(y)
            y = (y-y_min)/(y_max-y_min)

            integral = self.integration(channel, row, col, y)
            y = y / integral

            f = interpol((channel, col, row), y)

            save_name = f'{count}.p'
            count += 1

            y_max = np.max(y)
            sample_size = self.size * 6
            points = []
            values = []
            while len(points) < self.size:
                test = np.random.rand(sample_size, 3)
                vals = f(test)
                ran = np.random.rand(sample_size) * y_max
                truth = ran < vals
                test = test[truth]
                vals = vals[truth]
                [points.append(t) for t in test]
                [values.append(t) for t in vals]
                if sample_size < 2500000:
                    sample_size *= 2
                
            points, values = np.asarray(points), np.asarray(values)
            points, values = points[:self.size], values[:self.size]
            points = np.transpose(points)
            sample = np.concatenate((points, np.expand_dims(values, 0)))
            

            if self.with_grid:
                x_grid = np.linspace(0, 1, self.grid_number)
                y_grid = np.linspace(0, 1, self.grid_number)
                z_grid = np.linspace(0, 1, self.grid_number)
                grid = np.array(list(itertools.product(x_grid, y_grid, z_grid)))
                p_grid = f(grid)
                grid = np.transpose(grid)
                sample_grid = np.concatenate((grid, np.expand_dims(p_grid, 0)))

            if self.with_grid:
                save_file = [sample, sample_grid]
            else:
                save_file = sample

            pickle.dump(save_file, open(os.path.join(self.savedir, save_name), 'wb'))

class function_generation():
    '''
    This is the fast version of the function_generation_nd class, which contains only addition operator for dimension coupling.
    This function firtst builds DIM 1 dimensional functions, whose resulting functions are then coupled additvely to get the DIM dimensional function.
    The maximum of the DIM dimensional function is then calculated in "function_generator", which defines the functions.
    The generated functions are saved as pickled lists of size [num_funcs, dim+1, size]
    Args:   
            size:                       INT, number of samples drawn per function
            num_funcs:                  INT, number of functions to generate
            complexity:                 INT or tuple/list of two INTs, number of base_functions composed to the resulting 1D function compositions. If a tuple/list is provided, the complexity will vary for every function in the given range.
            scale:                      FLOAT or Tuple/List of two FLOATs, defines the range of the function support. Constant over all functions. Support will be [0, scale]. If tuple the scale will be randomly chosen in the provided range.
            dim:                        INT, dimensionality of the generated functions support
            max_initials:               INT, number of initial guesses for the maximum optimization to overcome discontinuities and local maxima.
            verbose:                    BOOL, adds some verbosity.
            naming:                     STRING, String appended to output filename to not overwrite other files with same selected functions, size, dim and num_funcs 
            pdf_save_dir:               String, directory of the saved PDFs. Contains lists to recreate the generated PDFs. 
            sampled_data_save_dir:      String, directory of the saved samples. Contains lists with size (num_funcs, dim+1, size)
            select_functions:           List of strings, group of functions with same characteristica to include in generation. Must be of the following: 'all', 'all_sub', 'gaussian', 'sinusoidal', 'linear', 'smooth', 'monotone', 'non_monotone'
            deselect_functions:         List of strings, group of functions with same characteristica to exclude in generation. Must be of the following: 'all', 'all_sub', 'gaussian', 'sinusoidal', 'linear', 'smooth', 'monotone', 'non_monotone'
    
    Usage: function_generation_nd(**kwargs).function_generation() # the functions are already defined during initialization of function_generation_nd
    
    '''

    def __init__(self, size=1000, num_funcs=1000, complexity=3, scale=1.0, dim=3, max_initials=None, verbose=True, naming='', pdf_save_dir=None, sampled_data_save_dir=None, select_functions=['all_sub'], deselect_functions=[]):
        self.dim = dim
        self.size = size
        self.naming = naming
        complexity = self.make_list(complexity)
        scale = self.make_list(scale)
        self.select_functions = self.make_list(select_functions)
        self.deselect_functions = self.make_list(deselect_functions)
        if len(complexity) == 1:
            self.complexity = [complexity[0] for i in range(num_funcs)] 
        elif len(complexity) == 2:
            self.complexity = [random.randint(min(complexity), max(complexity)) for i in range(num_funcs)]    
        else:
            raise ValueError('complexity must be either a single INT or a tuple/list of two INTs')
        if len(scale) == 1:
            self.scale = [np.float32(scale) for i in range(num_funcs)] 
        elif len(scale) == 2:
            self.scale = [np.float32(random.uniform(min(scale), max(scale))) for i in range(num_funcs)]
        else: 
            raise ValueError('scale must be either a FLOAT or a tuple/list of two FLOATs')

        self.verbose = verbose
        if max_initials is None:
            max_initials = 20 * max(complexity)/3
        self.max_initials = max_initials
        self.pdf_save_dir = pdf_save_dir
        self.sampled_data_save_dir = sampled_data_save_dir
        self.num_funcs = num_funcs
        self.rns = np.array([[np.float32(np.random.rand(self.complexity[i])) for j in range(dim)] for i in range(num_funcs)]) # results in shape (num_funcs, dim, complexity)
        self.names, self.tokens, self.maxima, self.integrals, self.integration_time = self.function_generator()

    def make_list(self, x):
        try:
            len(x)
            if isinstance(x, str):
                return [x]
            else:
                return x
        except:
            return [x]

    def get_max_1d(self, names, tokens, function_index, dim_index):
        '''
        This function calculates the maximum of the 1D functions. Since the DIM-D function is a additive composition of the 1D functions, the maximum of those can be upper bounded by the sum of all single maxima.
        Thus the drawback in time for importance sampling is at maximum the factor DIM.

        The "maximum" of the 1D function, i.e. the "c" in u <= f(x)/(c*g(x)) (rejection sampling) and a uniform distribution g.
        Formally c should be sup(f(x)/g(x)), i.e. sup(f(x)*M) for a uniform distribution g on support [0,M]. The factor of "M" is eliminated for uniform distribution g, as it appears again in the equation  for rejection sampling.
        Thus this implementation is for the check (C * u) <= f(x) where C ~ sup(f(x)). (C is an upper bound but in most cases not the sup) 
        '''
        def get_value_1d(x, names, tokens, function_index, dim_index):
            for i in range(self.complexity[function_index]):
                if i == 0:
                    val = utils(scale=self.scale[function_index], rn=self.rns[function_index, dim_index][i]).get_base_function(names[i])(x)
                else:
                    val = utils(scale=self.scale[function_index], rn=self.rns[function_index, dim_index][i]).get_operator(tokens[i-1])(
                                val, 
                                utils(scale=self.scale[function_index], rn=self.rns[function_index, dim_index][i]).get_base_function(names[i])(x))
            return -val

        maximum = 0
        max_initial_step = 1 / (self.max_initials * self.scale[function_index]/2.0)
        for x_ in np.arange(self.scale[function_index]*0.01, self.scale[function_index]+self.scale[function_index]*0.01, self.scale[function_index]*max_initial_step):
            max_candidat = -opt.minimize(get_value_1d, (x_+eps), args=(names, tokens, function_index, dim_index), method='L-BFGS-B', bounds=((0+eps,self.scale[function_index]),))['fun']
            if max_candidat > maximum:
                maximum = max_candidat
        if maximum < 0.1:
            for x in np.arange(0, self.scale[function_index], 0.0001):
                max_candidat = get_value_1d(x, names, tokens, function_index, dim_index)
                if max_candidat > maximum:
                    maximum = max_candidat
        return maximum

    def integration_1d(self, names, tokens, maximum, function_index, dim_index, n_samples=int(1e8)):
        '''
        Integration routine for 1-dimensional data. Used for integrating the 1 dimensional constituents of the n dimensional functions.
        Args:   names:  list of the function names passed to the get_value function.
                tokens: list of the operator names passed to the get_value function.
                maximum: max of the probability values of the 1d-function

        Kwargs: n_samples:  number of samples drawn in the range of the original lists to get a closer estimate of the integral.
        '''
        # The volume does not represent the volume of the 1D function, but the volume of (this decomposed part) of the DIM-D function
        volume = self.scale[function_index]**self.dim * maximum

        def get_value_1d(names, tokens, function_index, dim_index, x):
            for i in range(self.complexity[function_index]):
                if i == 0:
                    val = utils(scale=self.scale[function_index], rn=self.rns[function_index, dim_index][i]).get_base_functiontf(names[i])(x)
                else:
                    val = utils(scale=self.scale[function_index], rn=self.rns[function_index, dim_index][i]).get_operatortf(tokens[i-1])(
                                val, 
                                utils(scale=self.scale[function_index], rn=self.rns[function_index, dim_index][i]).get_base_functiontf(names[i])(x))
            return val

        support_samples = tf.random.uniform((1, n_samples), 0, self.scale[function_index])
        target_samples = tf.random.uniform((n_samples,), 0, maximum)
        value = get_value_1d(names, tokens, function_index, dim_index, support_samples)

        c_in_region = target_samples < value
        c_in_region = c_in_region.numpy()*1
        count = np.sum(c_in_region)
        result = float(count)/n_samples
        result *= volume
        result = np.squeeze(result)
        if result < 5e-1:
            # for a to small integral the function is most probably very flat with sharp peaks located over a small range on the support.
            # Thus the integral is made more accurate by repeating it a number of times which is equal to a higher sample count in the monte carlo integration.
            if self.verbose:
                print('integration correction is on')

            iterations = min(int(1e-2/result), 100)
            total_count = count
            for i in range(iterations):
                support_samples = tf.random.uniform((1, n_samples), 0, self.scale[function_index])
                target_samples = tf.random.uniform((n_samples,), 0, maximum)
                value = get_value_1d(names, tokens, function_index, dim_index, support_samples)
                c_in_region = target_samples < value
                c_in_region = c_in_region.numpy()*1
                count = np.sum(c_in_region)
                total_count += count
            result = float(total_count/(iterations+1))/n_samples
            result *=volume

        return np.float32(result)


    def function_generator(self):
        function_list = {
            'all'           :   ['x', 'x**2', 'sin(x)', 'cos(x)', 'sinabs(x)', 'sincabs(x)', 'cosabs(x)', 'sigmoid(x)', 'gauss(x)', 'gauss0(x)', 'gauss2(x)', 'gauss3(x)', 
                                    'gauss4(x)', 'gauss5(x)', 'gauss6(x)', 'gauss7(x)', 'inverse(x)', 'neg(x)', 'inverse_cut(x)', 'inverse_cut2(x)', 'inverse_cut3(x)', 
                                    'cut(x)', 'cut2(x)', 'twice(x)', 'thrice(x)' , 'half(x)', 'neg_square(x)', 'neg_square2(x)', 'sqrt(x)', 'potence(x)', 'potence2(x)', 
                                    'neg_potence(x)', 'neg_potence2(x)', 'step(x)', 'step2(x)'],
            'all_sub'       :   ['x', 'x**2', 'sin(x)', 'cos(x)', 'gauss5(x)', 'gauss6(x)', 'gauss7(x)', 'cut(x)', 'cut2(x)', 'twice(x)', 'half(x)', 'neg(x)', 
                                    'neg_square(x)', 'neg_square2(x)', 'sqrt(x)', 'potence(x)', 'potence2(x)', 'neg_potence(x)', 'neg_potence2(x)'], 
            'gaussian'      :   ['gauss(x)', 'gauss0(x)', 'gauss2(x)', 'gauss3(x)', 'gauss4(x)', 'gauss5(x)', 'gauss6(x)', 'gauss7(x)'], 
            'sinusoidal'    :   ['sin(x)', 'cos(x)', 'sinabs(x)', 'cosabs(x)'], 
            'linear'        :   ['x', 'neg(x)', 'cut(x)', 'cut2(x)', 'twice(x)', 'thrice(x)' , 'half(x)', 'step(x)', 'step2(x)'], 
            'smooth'        :   ['x', 'x**2', 'sin(x)', 'cos(x)', 'sigmoid(x)', 'gauss(x)', 'gauss0(x)', 'gauss2(x)', 'gauss3(x)', 'gauss4(x)', 'gauss5(x)', 
                                    'gauss6(x)', 'gauss7(x)', 'inverse(x)', 'neg(x)', 'twice(x)', 'thrice(x)' , 'half(x)', 'neg_square(x)', 'neg_square2(x)', 
                                    'sqrt(x)', 'potence(x)', 'potence2(x)', 'neg_potence(x)', 'neg_potence2(x)'], 
            'monotone'      :   ['x', 'x**2', 'sigmoid(x)', 'inverse(x)', 'neg(x)', 'inverse_cut(x)', 'inverse_cut2(x)', 'inverse_cut3(x)', 'cut(x)', 'cut2(x)', 
                                    'twice(x)', 'thrice(x)' , 'half(x)', 'neg_square(x)', 'neg_square2(x)', 'sqrt(x)', 'potence(x)', 'potence2(x)', 'neg_potence(x)', 'neg_potence2(x)'], 
            'non_monotone'  :   ['sin(x)', 'cos(x)', 'sinabs(x)', 'sincabs(x)', 'cosabs(x)', 'gauss(x)', 'gauss0(x)', 'gauss2(x)', 'gauss3(x)', 
                                    'gauss4(x)', 'gauss5(x)', 'gauss6(x)', 'gauss7(x)', 'step(x)', 'step2(x)']
                        }

        base_functions = []
        for function_set in self.select_functions:
            base_functions.append(function_list[function_set])
        
        base_functions = list(set(sum(base_functions, [])))
        if len(self.deselect_functions) >= 1:
            for function_set in self.deselect_functions:
                base_functions = list(set(base_functions) - set(function_list[function_set]))

        tokens = ['*', '+']
        if self.dim >= 50:
            tokens = ['+']

        all_names = [[[] for j in range(self.num_funcs)] for i in range(self.dim)]
        all_tokens = [[[] for j in range(self.num_funcs)] for i in range(self.dim)]
        maxima_1d = [[] for i in range(self.dim)]
        integral_1d = [[] for i in range(self.dim)]
        integration_time_1d = [[] for i in range(self.dim)]

        if self.verbose: 
            print('shape of the sampled indentifiers for names: ', np.shape(all_names))
            print('shape of the sampled indentifiers for tokens: ', np.shape(all_tokens))
            print('\n')
            print('calculating the maxima of the constituting 1D functions')

        for d in range(self.dim):
            if self.verbose: process_bar(d, self.dim, carriage_return=False) 
            for i in range(self.num_funcs):
                if self.verbose: process_bar(i, self.num_funcs, carriage_return=False) 
                checker = 0
                while checker == 0:
                    new_names = []
                    new_tokens = []
                    for j in range(self.complexity[i]):
                        new_names.append(random.choice(base_functions))
                    all_names[d][i] = new_names
                    for j in range(self.complexity[i]-1):
                        new_tokens.append(random.choice(tokens))
                    all_tokens[d][i] = new_tokens

                    current_maximum = self.get_max_1d(all_names[d][i], all_tokens[d][i], function_index=i, dim_index=d)

                    if current_maximum <= 1e-2:
                        checker = 0
                    else:
                        checker = 1

                maxima_1d[d].append(current_maximum)
                start = time.time()
                current_integral = self.integration_1d(all_names[d][i], all_tokens[d][i], current_maximum, function_index=i, dim_index=d)
                integral_1d[d].append(current_integral)
                integration_time_1d[d].append(time.time() - start)

        if self.verbose: 
            print('shape of the maxima_1d list:', np.shape(maxima_1d))
            print('the highest maximum of the generated 1D functions is:', np.max(maxima_1d))
            print('\n')

        maxima = np.sum(maxima_1d, axis=0)
        integrals = np.sum(integral_1d, axis=0)
        #if self.verbose:
            #print(f'list of {self.dim} dimensional integrals is:')
            #for i in range(self.dim):
                #print(integrals[i])
        integration_time = np.sum(integration_time_1d, axis=0)

        if self.verbose: 
            print('shape of the maxima list:', np.shape(maxima))
            print('the highest maximum of the generated DIM-D functions is:', np.max(maxima))
            print('\n')
        # before transpose shape is: (dim, num_funcs, complexity)
        # after: (num_func, dim, complexity)
        try:
            all_names = np.transpose(all_names, axes=[1,0])
            all_tokens = np.transpose(all_tokens, axes=[1,0])
        except:
            all_names = np.transpose(all_names, axes=[1,0,2])
            all_tokens = np.transpose(all_tokens, axes=[1,0,2])

        if self.verbose:
            print('shape of the transposed list for function identifiers', np.shape(all_names))
            print('shape of the transposed list for operator identifiers', np.shape(all_tokens))
            print('\n')

        return all_names, all_tokens, maxima, integrals, integration_time

    def get_valuetf(self, names, tokens, function_index, *args):
        val = 0
        for d in range(self.dim):
            for i in range(self.complexity[function_index]):
                if i == 0:
                    val_1d = utils(scale=self.scale[function_index], rn=self.rns[function_index, d][i]).get_base_functiontf(names[d][i])(args[d])
                else:
                    val_1d = utils(scale=self.scale[function_index], rn=self.rns[function_index, d][i]).get_operatortf(tokens[d][i-1])(
                                val_1d, 
                                utils(scale=self.scale[function_index], rn=self.rns[function_index, d][i]).get_base_functiontf(names[d][i])(args[d]))
            val += val_1d
        return val

    def parcalc(self, function_index, number):
        '''
        To work as a parallell implementation, this draws NUMBER random samples (new_support) and calculates the probability values for those with the functions provided in self. 
        Then a random uniform number, is drawn and the random samples are importance sampled according to u(x) < f(x)/(c*g(x)), where c is the maximum and g the uniform distribution.
        '''
        ran = tf.random.uniform((number,)) * self.maxima[function_index]                                     
        new_support = [tf.random.uniform((number,), maxval=self.scale[function_index]) for j in range(self.dim)]  
        newp = self.get_valuetf(self.names[function_index], self.tokens[function_index], function_index, *new_support)          
        accepted = ran < newp
        accepted, newp = accepted.numpy(), newp.numpy()
        new_support = [news.numpy() for news in new_support]
        new_support.append(newp)
        new_support = [news[accepted] for news in new_support]
        return np.asarray(new_support, dtype='float32')

    def function_generation(self):
        sample_set = np.zeros((self.num_funcs, self.dim+1, self.size), dtype='float32')
        if self.verbose: 
            print('shape of the proto sample set is:', np.shape(sample_set))
            print('\n')
            print ('sampling of the PDF data points')        

        for i in range(self.num_funcs):
            process_bar(i, self.num_funcs, carriage_return=False)

            val_=[[0],[0]]
            number = 2*self.size
            if self.dim <= 10:
                max_num = 81920000 
            elif self.dim > 10 and self.dim <50:
                max_num = 40960000
            else:
                max_num = 10240000            
            while np.shape(val_)[1] < self.size:
                if number < max_num:
                    number *= 2
                val_t = self.parcalc(function_index=i, number=number)
                val_t = val_t[:,:self.size]
                if number >= max_num:
                    val_ = np.concatenate((val_, val_t), axis=1)
                else:
                    val_ = copy.copy(val_t)
                val_ = val_[:,:self.size]
                #if self.verbose:
                    #print('shape of the val_ array (info necesssary to check for correct concatenation): ', np.shape(val_))

            sample_set[i] = val_
        if self.verbose: print('shape of the sample set is:', np.shape(sample_set))

        for j in range(self.num_funcs):
            if self.verbose: process_bar(j, self.num_funcs)
            sample_set[j, -1] = np.true_divide(sample_set[j, -1], self.integrals[j])

            sample_set[j, :-1] /= self.scale[j]
            sample_set[j, -1] *= self.scale[j]**self.dim

        pdf_save_file={'names': self.names,
                       'tokens': self.tokens,
                       'maxima': self.maxima,
                       'integrals': self.integrals,
                       'rns' : self.rns
                      }

        function_set_string = '{}'
        for i in range(len(self.select_functions)-1):
            function_set_string += '_{}'
        for j in range(len(self.deselect_functions)):
            function_set_string += '-{}'
        function_set_string = function_set_string.format(*self.select_functions, *self.deselect_functions)

        if self.pdf_save_dir is None:
            self.pdf_save_dir = f'new_data/pdf_data/generated_functions_{function_set_string}_dim_{self.dim}_size_{self.size}_numFuncs_{self.num_funcs}{self.naming}.p'
        if not self.pdf_save_dir.endswith('.p'):
            self.pdf_save_dir = self.pdf_save_dir + '.p'

        if self.sampled_data_save_dir is None:
            self.sampled_data_save_dir = f'new_data/sample_data/generated_functions_{function_set_string}_dim_{self.dim}_size_{self.size}_numFuncs_{self.num_funcs}{self.naming}.p'
        if not self.sampled_data_save_dir.endswith('.p'):
            self.sampled_data_save_dir = self.sampled_data_save_dir + '.p'

        try:
            os.makedirs(os.path.dirname(self.pdf_save_dir))
        except: pass
        try:
            os.makedirs(os.path.dirname(self.sampled_data_save_dir))
        except: pass

        pickle.dump(pdf_save_file, open(self.pdf_save_dir,'wb'))
        pickle.dump(sample_set, open(self.sampled_data_save_dir,'wb'))
        print('saved pdf data to ', self.pdf_save_dir)
        print('saved sample_data to ', self.sampled_data_save_dir)

class function_generation_locality_check():
    '''
    This class generates some 1d functions to check for the dependency on the local neighbourhood of a query point.
    The saved file is a list containing [pdf_samples, grid_samples, target_query_positions, target_query_probs]
    Args:   
            size:           INT, number of samples drawn per function
            verbose:        BOOL, adds some verbosity.
    '''

    def __init__(self, size=1000, verbose=True, sampled_data_save_dir=None):
        self.size = size
        self.scale = [2.0, 2.0, 1.0, 2.0, 3.0, 30.0, 2.0, 3.0, 6.0]
        self.num_funcs = len(self.scale)
        self.verbose = verbose
        self.sampled_data_save_dir = sampled_data_save_dir
        self.names, self.maxima, self.integrals = self.function_generator()
        
    def integration_1d(self, name, maximum, function_index, n_samples=int(1e8)):
        '''
        Integration routine for 1-dimensional data. Used for integrating the 1 dimensional constituents of the n dimensional functions.
        Args:   name:  function name passed to the get_value function.
                maximum: max of the probability values of the 1d-function

        Kwargs: n_samples:  number of samples drawn in the range of the original lists to get a closer estimate of the integral.
        '''
        # The volume does not represent the volume of the 1D function, but the volume of (this decomposed part) of the DIM-D function
        volume = self.scale[function_index] * maximum
        if self.verbose: print('current function name: ', name)
        def get_value_1d(name, x):
            val = utils().get_locality_base_functiontf(name)(x)
            return val

        support_samples = tf.random.uniform((1, n_samples), 0, self.scale[function_index])
        #support_samples = tf.transpose(support_samples)
        target_samples = tf.random.uniform((n_samples,), 0, maximum)
        value = self.get_valuetf(name, support_samples)

        c_in_region = target_samples < value
        c_in_region = c_in_region.numpy()*1
        count = np.sum(c_in_region)
        result = float(count)/n_samples
        result *= volume

        return np.float32(result)


    def function_generator(self):
        base_functions = ['step(x)', 'linear1(x)', 'linear2(x)', 'sin1(x)', 'sin2(x)', 'gauss(x)', 'potence1(x)', 'potence2(x)', 'sin3(x)']
        maxima = []
        integrals = []
        
        for n, bf in enumerate(base_functions):
            if self.verbose: process_bar(n, len(base_functions), carriage_return=True) 
            current_maximum = utils().get_locality_single_max(bf)
            maxima.append(current_maximum)
            integrals.append(self.integration_1d(bf, current_maximum, function_index=n))

        if self.verbose: 
            print('shape of the maxima list:', np.shape(maxima))
            print('the highest maximum of the generated DIM-D functions is:', np.max(maxima))
            print('the integrals should all be 1, check the following numbers to assert this')
            print(integrals)
            print('\n')

        return base_functions, maxima, integrals

    def get_valuetf(self, name, x):
        val = utils().get_locality_base_functiontf(name)(x)
        return val

    def parcalc(self, function_index, number):
        '''
        To work as a parallell implementation, this draws NUMBER random samples (new_support) and calculates the probability values for those with the functions provided in self. 
        Then a random uniform number, is drawn and the random samples are importance sampled according to u(x) < f(x)/(c*g(x)), where c is the maximum and g the uniform distribution.
        '''
        ran = tf.random.uniform((number,)) * self.maxima[function_index]                                     
        new_support = tf.random.uniform((number,), maxval=self.scale[function_index])  
        newp = self.get_valuetf(self.names[function_index], new_support)          
        accepted = ran < newp
        accepted, newp = accepted.numpy(), newp.numpy()
        new_support = [new_support.numpy()]
        new_support.append(newp)
        new_support = [news[accepted] for news in new_support]
        if self.verbose:
            print('shape of the new_support (check correct appending)', np.shape(new_support))
        return np.asarray(new_support, dtype='float32')

    def function_generation(self):
        sample_set = np.zeros((self.num_funcs, 2, self.size), dtype='float32')
        if self.verbose: 
            print('shape of the proto sample set is:', np.shape(sample_set))
            print('\n')
            print ('sampling of the PDF data points')        

        for i in range(self.num_funcs):
            process_bar(i, self.num_funcs, carriage_return=True)

            val_=[[0],[0]]
            number = 2*self.size
            max_num = 81920000 
            if self.verbose: print('current function names: ', self.names[i])
            while np.shape(val_)[1] < self.size:
                if number < max_num:
                    number *= 2
                val_t = self.parcalc(function_index=i, number=number)
                val_t = val_t[:,:self.size]
                if number >= max_num:
                    val_ = np.concatenate((val_, val_t), axis=1)
                else:
                    val_ = copy.copy(val_t)
                val_ = val_[:,:self.size]
                if self.verbose:
                    print('shape of the val_ array (info necesssary to check for correct concatenation): ', np.shape(val_))

            sample_set[i] = val_
        if self.verbose: print('shape of the sample set is:', np.shape(sample_set))

        if self.verbose: print('sampling the true value at target positions. Should be close to 1')
        target_positions = [1.0, 2.0, 0.5, pi/2-1e-7, pi/2, 15.0, 1.0, np.sqrt(3), 3*pi/2]
        target_values = []
        for i in range(self.num_funcs):
            target_values.append(self.get_valuetf(self.names[i], tf.cast(target_positions[i], tf.float32)).numpy())

        if self.verbose:
            print('assert that the following values are about 1')
            print(target_values)

        if self.verbose:
            print('sampling grid values for better visualization')

        grid_values = []
        for i in range(self.num_funcs):
            current_pos = tf.linspace(eps, self.scale[i], 10000)
            current_val = self.get_valuetf(self.names[i], current_pos)
            grid_values.append([current_pos.numpy(), current_val.numpy()])

        grid_values = np.asarray(grid_values)

        if self.verbose:
            print('shape of the samples grids is:', np.shape(grid_values))

        for j in range(self.num_funcs):
            if self.verbose: process_bar(j, self.num_funcs)
            sample_set[j, -1] = np.true_divide(sample_set[j, -1], self.integrals[j])

            sample_set[j, :-1] /= self.scale[j]
            sample_set[j, -1] *= self.scale[j]
            target_values[j] *= self.scale[j] 
            target_positions[j] /= self.scale[j]
            grid_values[j, :-1] /= self.scale[j]
            grid_values[j, -1] *= self.scale[j]



        if self.sampled_data_save_dir is None:
            self.sampled_data_save_dir = f'new_data/sample_data/generated_locality_test_functions_size_{self.size}.p'
        if not self.sampled_data_save_dir.endswith('.p'):
            self.sampled_data_save_dir = self.sampled_data_save_dir + '.p'


        try:
            os.makedirs(os.path.dirname(self.sampled_data_save_dir))
        except: pass

        pickle.dump([sample_set, grid_values, target_positions, target_values, self.scale], open(self.sampled_data_save_dir,'wb'))
        print('saved sample_data to ', self.sampled_data_save_dir)

class function_generation_more_time():
    '''
    This is the slow version of the function_generation_nd class, which couples 1d functions with '+' or '*'. As this means the dimensions can not be easily decoupled during integration and maxima calculation, it takes much longer for higher DIM.
    The function_generation_nd class grows linearly in time with DIM, while this class grows exponentially with DIM. 
    In contrast to before, this function firtst builds DIM 1 dimensional tree, whose resulting functions are then coupled additvely to get the DIM dimensional function.
    The maximum of the DIM dimensional function is then calculated in "function_generator", which defines the functions.
    Args:   
            size:                       INT, number of samples drawn per function
            num_funcs:                  INT, number of functions to generate
            complexity:                 INT or tuple/list of two INTs, number of base_functions composed to the resulting 1D function compositions. If a tuple/list is provided, the complexity will vary for every function in the given range.
            scale:                      FLOAT or Tuple/List of two FLOATs, defines the range of the function support. Constant over all functions. Support will be [0, scale]. If tuple the scale will be randomly chosen in the provided range.
            dim:                        INT, dimensionality of the generated functions support
            max_initials:               INT, number of initial guesses for the maximum optimization to overcome discontinuities and local maxima.
            verbose:                    BOOL, adds some verbosity.
            naming:                     STRING, String appended to output filename to not overwrite other files with same selected functions, size, dim and num_funcs 
            pdf_save_dir:               String, directory of the saved PDFs. Contains lists to recreate the generated PDFs. 
            sampled_data_save_dir:      String, directory of the saved samples. Contains lists with size (num_funcs, dim+1, size)
            select_functions:           List of strings, group of functions with same characteristica to include in generation. Must be of the following: 'all', 'all_sub', 'gaussian', 'sinusoidal', 'linear', 'smooth', 'monotone', 'non_monotone'
            deselect_functions:         List of strings, group of functions with same characteristica to exclude in generation. Must be of the following: 'all', 'all_sub', 'gaussian', 'sinusoidal', 'linear', 'smooth', 'monotone', 'non_monotone'

    Usage: function_generation_nd_more_time(**kwargs).function_generation() # the functions are already defined during initialization of function_generation_nd

    '''

    def __init__(self, size=1000, num_funcs=1000, complexity=3, scale=1.0, dim=3, max_initials=None, verbose=True, naming='', pdf_save_dir=None, sampled_data_save_dir=None, select_functions=['all_sub'], deselect_functions=[]):
        self.dim = dim
        self.naming = naming
        self.size = size
        complexity = self.make_list(complexity)
        scale = self.make_list(scale)
        self.select_functions = self.make_list(select_functions)
        self.deselect_functions = self.make_list(deselect_functions)
        if len(complexity) == 1:
            self.complexity = [complexity[0] for i in range(num_funcs)] 
        elif len(complexity) == 2:
            self.complexity = [random.randint(min(complexity), max(complexity)) for i in range(num_funcs)]    
        else:
            raise ValueError('complexity must be either a single INT or a tuple/list of two INTs')
        if len(scale) == 1:
            self.scale = [np.float32(scale) for i in range(num_funcs)] 
        elif len(scale) == 2:
            self.scale = [np.float32(random.uniform(min(scale), max(scale))) for i in range(num_funcs)]
        else: 
            raise ValueError('scale must be either a FLOAT or a tuple/list of two FLOATs')

        self.verbose = verbose
        if max_initials is None:
            max_initials = 20 * max(complexity)/3
        self.max_initials = max_initials
        self.pdf_save_dir = pdf_save_dir
        self.sampled_data_save_dir = sampled_data_save_dir
        self.num_funcs = num_funcs
        self.rns = np.array([[np.float32(np.random.rand(self.complexity[i])) for j in range(dim)] for i in range(num_funcs)]) # results in shape (num_funcs, dim, complexity)
        self.names, self.dimension_tokens, self.function_tokens, self.maxima, self.integrals, self.integration_time = self.function_generator()

    def make_list(self, x):
        try:
            len(x)
            if isinstance(x, str):
                return [x]
            else:
                return x
        except:
            return [x]

    def get_max(self, names, tokens, function_tokens, function_index):
        '''
        This function calculates the maximum of the DIM-D functions. The Maximum is simply overestimated 
        by connecting all maxima of the constituting 1D functions to the resulting DIM-D function with the respectively sampled operators.

        The "maximum" of the 1D function, i.e. the "c" in u <= f(x)/(c*g(x)) (rejection sampling) and a uniform distribution g.
        Formally c should be sup(f(x)/g(x)), i.e. sup(f(x)*M) for a uniform distribution g on support [0,M]. 
        The factor of "M" is eliminated for uniform distribution g, as it appears again in the equation  for rejection sampling.
        Thus this implementation is for the check (C * u) <= f(x) where C ~ sup(f(x)). (C is an upper bound but in most cases not the sup) 
        '''
        maximum = 0
        for c in range(self.complexity[function_index]):
            current_maximum = 0
            for d in range(self.dim):
                if d == 0:
                    current_maximum = utils(scale=self.scale[function_index], rn=self.rns[function_index, d][c]).get_single_max(names[c][d])
                else:
                    current_maximum = utils(scale=self.scale[function_index], rn=self.rns[function_index, d][c]).get_operator(tokens[c][d-1])(
                                            current_maximum,
                                            utils(scale=self.scale[function_index], rn=self.rns[function_index, d][c]).get_single_max(names[c][d]))
            if c == 0:
                maximum = current_maximum
            else:
                maximum = utils(scale=self.scale[function_index], rn=self.rns[function_index, d][c]).get_operator(function_tokens[c-1])(
                                maximum,
                                current_maximum)
        return maximum

    def get_value(self, names, tokens, function_tokens, function_index, *args):
        val = 0
        for c in range(self.complexity[function_index]):
            current_val = 0
            for d in range(self.dim):
                if d == 0:
                    current_val = utils(scale=self.scale[function_index], rn=self.rns[function_index, d][c]).get_base_functiontf(names[c][d])(args[d])
                else:
                    current_val = utils(scale=self.scale[function_index], rn=self.rns[function_index, d][c]).get_operatortf(tokens[c][d-1])(
                                            current_val,
                                            utils(scale=self.scale[function_index], rn=self.rns[function_index, d][c]).get_base_functiontf(names[c][d])(args[d]))
            if c == 0:
                val = current_val
            else:
                val = utils(scale=self.scale[function_index], rn=self.rns[function_index, d][c]).get_operatortf(function_tokens[c-1])(
                                val,
                                current_val)
        return val

    def integration(self, names, tokens, function_tokens, maximum, function_index):
        '''
        Integration routine for 1-dimensional data. Used for integrating the 1 dimensional constituents of the n dimensional functions.
        Args:   names:  list of the function names passed to the get_value function.
                tokens: list of the operator names passed to the get_value function.
                maximum: max of the probability values of the 1d-function

        Kwargs: n_samples:  number of samples drawn in the range of the original lists to get a closer estimate of the integral.
        '''

        volume = self.scale[function_index]**self.dim * maximum

        if self.dim <= 4:
            n_samples = int(1e8)
            n_iterations = self.dim
        elif self.dim <= 50:
            n_samples = int(1e7)
            n_iterations = self.dim * 2
        else:
            n_samples = int(1e6)
            n_iterations = self.dim * 10

        def compute_integral(n_samples=1e8, count=0):
            support_samples = tf.random.uniform((self.dim, n_samples), 0.0, self.scale[function_index])
            target_samples = tf.random.uniform((n_samples,), 0, maximum)
            value = self.get_value(names, tokens, function_tokens, function_index, *support_samples)

            c_in_region = target_samples < value
            c_in_region = c_in_region.numpy()*1
            count_ = np.sum(c_in_region)
            count_ = float(count_)/n_samples
            return count + count_

        count = 0
        for n in range(n_iterations):
            count = compute_integral(n_samples, count)

        count = count / n_iterations

        result = volume * count

        return np.float32(result)


    def function_generator(self):
        function_list = {
            'all'           :   ['x', 'x**2', 'sin(x)', 'cos(x)', 'sinabs(x)', 'sincabs(x)', 'cosabs(x)', 'sigmoid(x)', 'gauss(x)', 'gauss0(x)', 'gauss2(x)', 'gauss3(x)', 
                                    'gauss4(x)', 'gauss5(x)', 'gauss6(x)', 'gauss7(x)', 'inverse(x)', 'neg(x)', 'inverse_cut(x)', 'inverse_cut2(x)', 'inverse_cut3(x)', 
                                    'cut(x)', 'cut2(x)', 'twice(x)', 'thrice(x)' , 'half(x)', 'neg_square(x)', 'neg_square2(x)', 'sqrt(x)', 'potence(x)', 'potence2(x)', 
                                    'neg_potence(x)', 'neg_potence2(x)', 'step(x)', 'step2(x)'],
            'all_sub'       :   ['x', 'x**2', 'sin(x)', 'cos(x)', 'gauss5(x)', 'gauss6(x)', 'gauss7(x)', 'cut(x)', 'cut2(x)', 'twice(x)', 'half(x)', 'neg(x)', 
                                    'neg_square(x)', 'neg_square2(x)', 'sqrt(x)', 'potence(x)', 'potence2(x)', 'neg_potence(x)', 'neg_potence2(x)'], 
            'gaussian'      :   ['gauss(x)', 'gauss0(x)', 'gauss2(x)', 'gauss3(x)', 'gauss4(x)', 'gauss5(x)', 'gauss6(x)', 'gauss7(x)'], 
            'sinusoidal'    :   ['sin(x)', 'cos(x)', 'sinabs(x)', 'cosabs(x)'], 
            'linear'        :   ['x', 'neg(x)', 'cut(x)', 'cut2(x)', 'twice(x)', 'thrice(x)' , 'half(x)', 'step(x)', 'step2(x)'], 
            'smooth'        :   ['x', 'x**2', 'sin(x)', 'cos(x)', 'sigmoid(x)', 'gauss(x)', 'gauss0(x)', 'gauss2(x)', 'gauss3(x)', 'gauss4(x)', 'gauss5(x)', 
                                    'gauss6(x)', 'gauss7(x)', 'inverse(x)', 'neg(x)', 'twice(x)', 'thrice(x)' , 'half(x)', 'neg_square(x)', 'neg_square2(x)', 
                                    'sqrt(x)', 'potence(x)', 'potence2(x)', 'neg_potence(x)', 'neg_potence2(x)'], 
            'monotone'      :   ['x', 'x**2', 'sigmoid(x)', 'inverse(x)', 'neg(x)', 'inverse_cut(x)', 'inverse_cut2(x)', 'inverse_cut3(x)', 'cut(x)', 'cut2(x)', 
                                    'twice(x)', 'thrice(x)' , 'half(x)', 'neg_square(x)', 'neg_square2(x)', 'sqrt(x)', 'potence(x)', 'potence2(x)', 'neg_potence(x)', 'neg_potence2(x)'], 
            'non_monotone'  :   ['sin(x)', 'cos(x)', 'sinabs(x)', 'sincabs(x)', 'cosabs(x)', 'gauss(x)', 'gauss0(x)', 'gauss2(x)', 'gauss3(x)', 
                                    'gauss4(x)', 'gauss5(x)', 'gauss6(x)', 'gauss7(x)', 'step(x)', 'step2(x)']
                        }

        base_functions = []
        for function_set in self.select_functions:
            base_functions.append(function_list[function_set])
        
        base_functions = list(set(sum(base_functions, [])))
        if len(self.deselect_functions) >= 1:
            for function_set in self.deselect_functions:
                base_functions = list(set(base_functions) - set(function_list[function_set]))

        tokens = ['*', '+']
        token_weights_intra = [1, 2]
        token_weights_inter = [1, 2]
        if self.dim == 100:
            tokens = ['+']
            token_weights_intra = [1]
            token_weights_inter = [1]

        all_names = [[] for i in range(self.num_funcs)]
        dimension_tokens = [[] for i in range(self.num_funcs)]
        function_tokens = [[] for i in range(self.num_funcs)]
        maxima = []
        integral = []
        integration_time = []

        if self.verbose: print('sampling the function and operator identifiers')
        for i in range(self.num_funcs):
            for j in range(self.complexity[i]):
                new_names = []
                new_tokens = []
                new_names = random.choices(base_functions, k=self.dim)
                new_tokens = random.choices(tokens, weights=token_weights_intra, k=self.dim-1)
                all_names[i].append(new_names)
                dimension_tokens[i].append(new_tokens)

            function_tokens[i] = random.choices(tokens, weights=token_weights_inter, k=self.complexity[i]-1)


        if self.verbose: 
            print('names if a list of num_funcs list, each containing complexity lists of len dim. tokens accordingly.')
            print('shape of the sampled indentifiers for names: ', np.shape(all_names))
            print('shape of the sampled indentifiers for tokens: ', np.shape(dimension_tokens))
            print('\n')
            print('calculating the maxima of the constituting 1D functions')
  
        for i in range(self.num_funcs):
            if self.verbose: process_bar(i, self.num_funcs, carriage_return=False)
            maxima.append(self.get_max(all_names[i], dimension_tokens[i], function_tokens[i], function_index=i))
            start = time.time()
            integral.append(self.integration(all_names[i], dimension_tokens[i], function_tokens[i], maxima[i], function_index=i))
            integration_time.append(time.time() - start)

        if self.verbose: 
            print('shape of the maxima list:', np.shape(maxima))
            print('the highest maximum of the generated functions is:', np.max(maxima))
            print('the mean of the maxima list is:', np.mean(maxima))
            print('the std of the maxima list is:', np.std(maxima))
            print('\n')

        return all_names, dimension_tokens, function_tokens, maxima, integral, integration_time

    def parcalc(self, function_index, number):
        '''
        To work as a parallell implementation, this draws NUMBER random samples (new_support) and calculates the probability values for those with the functions provided in self. 
        Then a random uniform number, is drawn and the random samples are importance sampled according to u(x) < f(x)/(c*g(x)), where c is the maximum and g the uniform distribution.
        '''
        ran = tf.random.uniform((number,)) * self.maxima[function_index]                                     
        new_support = [tf.random.uniform((number,), maxval=self.scale[function_index]) for j in range(self.dim)]  
        newp = self.get_value(self.names[function_index], self.dimension_tokens[function_index], self.function_tokens[function_index], function_index, *new_support)          
        accepted = ran < newp
        accepted, newp = accepted.numpy(), newp.numpy()
        new_support = [news.numpy() for news in new_support]
        new_support.append(newp)
        new_support = [news[accepted] for news in new_support]
        return np.asarray(new_support, dtype='float32')

    def function_generation(self):
        sample_set = np.zeros((self.num_funcs, self.dim+1, self.size), dtype='float32')
        if self.verbose: 
            print('shape of the proto sample set is:', np.shape(sample_set))
            print('\n')
            print ('sampling of the PDF data points')        

        for i in range(self.num_funcs):
            process_bar(i, self.num_funcs, carriage_return=False)

            val_=[[0],[0]]
            number = 2*self.size
            if self.dim <= 10:
                max_num = 81920000 
            elif self.dim > 10 and self.dim <=50:
                max_num = 40960000
            else:
                max_num = 10240000  
            while np.shape(val_)[1] < self.size:
                if number < max_num:
                    number *= 2
                val_t = self.parcalc(function_index=i, number=number)
                val_t = val_t[:,:self.size]
                if number >= max_num:
                    val_ = np.concatenate((val_, val_t), axis=1)
                else:
                    val_ = copy.copy(val_t)
                val_ = val_[:,:self.size]
                if self.verbose:
                    print('shape of the val_ array (info necesssary to check for correct concatenation): ', np.shape(val_))

            #if self.verbose: print('shape of the values for one function is:', np.shape(val_))

            sample_set[i] = val_

        if self.verbose: print('shape of the sample set is:', np.shape(sample_set))

        for j in range(self.num_funcs):
            if self.verbose: process_bar(j, self.num_funcs)
            sample_set[j, -1] = np.true_divide(sample_set[j, -1], self.integrals[j])

            sample_set[j, :-1] /= self.scale[j]
            sample_set[j, -1] *= self.scale[j]**self.dim

        pdf_save_file={'names': self.names,
                       'dimension_tokens': self.dimension_tokens,
                       'function_tokens': self.function_tokens,
                       'maxima': self.maxima,
                       'integrals': self.integrals
                      }

        function_set_string = '{}'
        for i in range(len(self.select_functions)-1):
            function_set_string += '_{}'
        for j in range(len(self.deselect_functions)):
            function_set_string += '-{}'
        function_set_string = function_set_string.format(*self.select_functions, *self.deselect_functions)

        if self.pdf_save_dir is None:
            self.pdf_save_dir = f'new_data/pdf_data/more_random/generated_functions_{function_set_string}_dim_{self.dim}_size_{self.size}_numFuncs_{self.num_funcs}{self.naming}.p'
        if not self.pdf_save_dir.endswith('.p'):
            self.pdf_save_dir = self.pdf_save_dir + '.p'

        if self.sampled_data_save_dir is None:
            self.sampled_data_save_dir = f'new_data/sample_data/more_random/generated_functions_{function_set_string}_dim_{self.dim}_size_{self.size}_numFuncs_{self.num_funcs}{self.naming}.p'
        if not self.sampled_data_save_dir.endswith('.p'):
            self.sampled_data_save_dir = self.sampled_data_save_dir + '.p'

        try:
            os.makedirs(os.path.dirname(self.pdf_save_dir))
        except: pass
        try:
            os.makedirs(os.path.dirname(self.sampled_data_save_dir))
        except: pass

        pickle.dump(pdf_save_file, open(self.pdf_save_dir,'wb'))
        pickle.dump(sample_set, open(self.sampled_data_save_dir,'wb'))
        print('saved pdf data to ', self.pdf_save_dir)
        print('saved sample_data to ', self.sampled_data_save_dir)

             


