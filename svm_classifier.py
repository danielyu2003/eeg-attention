from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import csv
import os

os.chdir(os.path.dirname(__file__))

class EEGClassifier:
    '''
    Binary classification of EEG signal data using the scikit-learn support vector machine implementation
    This particular script distinguishes between rest and concentration using eye-tracking and visual-haptic feedback

    Attributes:
    ---
    kernel_choice : str
    data : numpy.ndarray.dtype(float64)
    target : numpy.ndarray.dtype(float64)
    test_size : float
    random_state : int
    shuffle_choice : bool

    Methods:
    ---
    fit()
        Trains the classifier on a portion of the dataset
    predict()
        Tests the classifier on a portion of the dataset
    predictExample(example)
        Passes a sample to the trained classifier for prediction
    getMetrics()
        Displays the metrics of the classifier after self.predict
    plotAcc()
        Plots each true label of each sample in the testing set and what the model predicted for each sample

    Functions:
    ---
    resample(signal, orig_freq, desired_freq)
        Resamples a given signal (or an array of signals) from its given frequency to a new one

    '''
    
    def __init__(self, data, target, test_ratio=0.25, random_seed=None, C_val=1.0, kernel_choice='rbf', gamma_choice = 'scale'):
        '''
        Parameters:
        ---
        kernel_choice : str
            default = 'rbf' (radial basis function)
                ideal for nonlinearly seperable data
            selects kernel choice for the support vector machine
            see the official scikit-learn website for more details 
            (https://scikit-learn.org/stable/modules/svm.html)
        data : numpy.ndarray.dtype(float64)
            an array/matrix of shape (num of samples/y, num of features/x)
        target : numpy.ndarray.dtype(int32)
            a 1d array of length (num of samples), where each sample is classified as a 1 or a 0
        test_size : float
            a decimal value between 0 and 1 representing what percent of the dataset to use for testing
        random_state : int
            an optional value which selects samples for training/testing from the given seed
        shuffle_choice : bool
            an optional value which determines whether or not the data is shuffled before splitting
        '''
        self.classifier = svm.SVC(C=C_val, kernel=kernel_choice, gamma=gamma_choice)
        self.data = data
        self.target = target
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data, self.target,test_size=test_ratio,random_state=random_seed)
        self.y_pred = np.zeros(self.y_test.size)
        pass

    def fit(self):
        '''
        Trains the classifier on a portion of the dataset
        '''
        self.classifier.fit(self.x_train, self.y_train)
        pass

    def predict(self):
        '''
        Tests the classifier on a portion of the dataset
        '''
        self.y_pred = self.classifier.predict(self.x_test)
        pass

    def predictExample(self, example):
        '''
        Passes a sample to the trained classifier for prediction

        Parameters:
        ---
        example : numpy.ndarray.dtype(float64)
            contains an array of samples (which are arrays of floats matching the feature dimensions)

        Returns:
        ---
            output : int
                either 0 or 1, corresponding to the label associated with each number
        '''
        output = self.classifier.predict(example)
        return output

    def getMetrics(self):
        '''
        Displays the metrics of the classifier after self.predict
        '''
        print("Accuracy:",metrics.accuracy_score(self.y_test, self.y_pred))
        print("Precision:",metrics.precision_score(self.y_test, self.y_pred))
        print("Recall:",metrics.recall_score(self.y_test, self.y_pred))
        pass
    
    def plotAcc(self):
        '''
        Plots each true label of each sample in the testing set and what the model predicted for each sample
        '''
        plt.figure()
        plt.plot(range(0, len(self.y_test)), self.y_test, 'b', label='truth')
        plt.plot(range(0,len(self.y_pred)), self.y_pred,'r',label='SVM')
        plt.legend()
        plt.show()
        pass


def resample(signal, orig_freq, desired_freq):
    '''
    Resamples a given signal (or an array of signals) from its given frequency to a new one

    Parameters:
    ---
    signal : numpy.ndarray.dtype(float64)
        Contains a 1d sequence of time frequency data, or a collection of such sequences
    orig_freq : int
        Defines the original sampling rate of the given signal
    desired_freq : int
        Defines the sampling rate the signal is resampled to

    Returns:
    ---
    resampled_signal : numpy.ndarray.dtype(float64)
        Contains a 1d sequence of the resampled time frequency data, 
        or a collection of each resampled sequence if there are multiple

    Notes:
    ---
    Uses linear interpolation to determine the resampled values
    For multidimensional arrays, only works if all columns share the same length
    '''
    scale = desired_freq/orig_freq
    new_len = round(len(signal) * scale)

    if (signal.ndim > 1):
        output = np.empty([new_len, signal.ndim+1])
        for ind in range(signal.ndim+1):
            output[:, ind] = resample(signal[:, ind], orig_freq, desired_freq)
        return output
    
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, new_len, endpoint=False),
        np.linspace(0.0, 1.0, len(signal), endpoint=False),
        signal)
    return resampled_signal

def meanAndStdev(samps, freq_wind, choice=True):
    '''
    Parameters:
    ---
    samps : np.ndarray.dtype(float)
        1d array of input data
    freq_wind : int
        Frequency window, approximately a tenth of the sampling frequency
    choice : bool

    Returns:
    ---
    targets : np.ndarray.dtype(float)
    reshaped : np.ndarray.dtype(float)

    Both return a 2d array containing the mean and stdev of each window specified by freq_wind
    '''
    remainder = len(samps)%freq_wind
    targets = [[np.mean(samps[ind*freq_wind:(ind*freq_wind)+freq_wind]), np.std(samps[ind*freq_wind:ind*freq_wind + freq_wind])] for ind in range(len(samps)//freq_wind)]
    if choice:
        reshaped = []
        for ind in range(len(targets)):
            reshaped += [targets[ind]]*freq_wind
    if remainder:
        targets += [[np.mean(samps[remainder:]), np.std(samps[remainder:])]]
        if choice:
            reshaped += remainder*[targets[-1]]
    if choice:
        return np.array(reshaped)
    return np.array(targets)

def averageThree(arr1, arr2, arr3, samp_freq):
    '''
    assuming all three inputs are of the same length

    returns average of each mean and stdev of each sample
    '''
    averaged = []
    data1 = meanAndStdev(arr1, samp_freq//10)
    data2 = meanAndStdev(arr2, samp_freq//10)
    data3 = meanAndStdev(arr3, samp_freq//10)

    for ind in range(len(data1)):
        averaged+=[[np.mean((data1[ind, 0], data2[ind, 0], data3[ind, 0])), np.mean((data1[ind, 1], data2[ind, 1], data3[ind, 1]))]]

    return np.array(averaged)

def targets(arr):
    print(arr)
    result = np.empty(len(arr))
    for ind in range(len(arr)):
        
        if arr[ind, 0] > arr[ind, 1]:
            result[ind] = 0 
        else:
            result[ind] = 1

    return result

def getTargets(eye_path, samp_freq, desired_freq):
    '''
    Takes in eye tracking data (as csv) and returns a np array of 1s and 0s

    Parameters:
    ---
    eye_path : str
    samp_freq : int

    Returns:
    ---
    targets : np.ndarray.dtype(int)
    '''

    df = pd.read_csv(eye_path, skiprows=1)[1:]

    col1 = df.loc[:, "EyeX"]
    col2 = df.loc[:, "EyeY"]
    col3 = df.loc[:, "EyeZ"]

    eye_samps = pd.concat([col1, col2, col3], axis=1)

    eye_samps = eye_samps.to_numpy()

    upsampled = resample(eye_samps, samp_freq, desired_freq)

    transposed = upsampled.T

    arr1 = transposed[0]
    arr2 = transposed[1]
    arr3 = transposed[2]

    samp_targets = targets(averageThree(arr1, arr2, arr3, samp_freq))

    return samp_targets

if __name__ == '__main__':

    # getTargets(r"C:\Users\yudan\OneDrive\Desktop\eeg_attention\data\eye\BLOCK_1\TRAINING\Trial_2.csv", 45, 256)

    # leave out eyeZ data?

    # assign labels based on coefficient of variation? stdev/mean

    df = pd.read_csv(r"C:\Users\yudan\OneDrive\Desktop\eeg_attention\data\eye\BLOCK_1\TRAINING\Trial_2.csv", skiprows=1)[1:]

    col1 = df.loc[:, "EyeX"]
    col2 = df.loc[:, "EyeY"]
    col3 = df.loc[:, "EyeZ"]

    eye_samps = pd.concat([col1, col2, col3], axis=1)

    eye_samps = eye_samps.to_numpy()

    upsampled = resample(eye_samps, 45, 256)

    print(upsampled[350:355])

    # print(f"[{np.std(upsampled[start:end, 0])/np.mean(upsampled[start:end, 0])}]\n[{np.std(upsampled[start:end, 1])/np.mean(upsampled[start:end, 1])}]\n[{np.std(upsampled[start:end, 2])/np.mean(upsampled[start:end, 2])}]")

    pass
