'''
Author: Daniel Yu
'''

from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from itertools import repeat
from sklearn import metrics
from sklearn import utils
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

    def predictExample(self, example, labels=[], metric=False, plot=False):
        '''
        Passes a sample to the trained classifier for prediction

        Parameters:
        ---
        example : numpy.ndarray.dtype(float64)
            contains an array of samples (which are arrays of floats matching the feature dimensions)
        labels : numpy.ndarray.dtype(int)
            contains an array of labels (either 1 or 0) corresponding to each sample in example
        metrics : bool
            if enabled, prints accuracy, precision, and recall of the predicted examples
        plot : bool
            if enabled, plots the accuracy of the predicted labels against truth

        Returns:
        ---
            output : numpy.ndarray.dtype(int)
                an array containing 0 and 1s, corresponding to the label predicted for each sample
        '''
        output = self.classifier.predict(example)

        if metric==True:
            print("Accuracy:",metrics.accuracy_score(labels, output))
            print("Precision:",metrics.precision_score(labels, output))
            print("Recall:",metrics.recall_score(labels, output))

        if plot==True:
            plt.figure()
            plt.plot(range(0, len(labels)), labels, 'b', label='truth')
            plt.plot(range(0,len(output)), output,'r',label='SVM')
            plt.legend()
            plt.show()

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

def label(stdev):
    if stdev < 0.002:
        return 1
    else:
        return 0

def targets(arr, desired_freq, verbose=False):
    '''
    Returns a 1d array of targets determined by the averaged coefficient of variation of each window in the given array
    '''

    averages = [np.mean(samp) for samp in arr]

    interval = desired_freq//10

    remainder = len(averages)%interval

    stdev = [np.std(averages[ind:ind+interval]) for ind in range(len(averages)//interval)]

    result = [x for item in stdev for x in repeat(item, interval)]

    if remainder:
        result+=[np.std(averages[-remainder:])]*remainder

    labels = [label(num) for num in result]

    if verbose == True:
        print(f"percent of attentive samples: {round(sum(labels)/len(labels) * 100, 2)}%")

    return labels


def getTargets(eye_path, samp_freq, desired_freq, startind, endind, verbose=False):
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

    eye_samps = eye_samps.to_numpy()[startind:endind]

    upsampled = resample(eye_samps, samp_freq, desired_freq)

    samp_targets = targets(upsampled, desired_freq, verbose)

    return samp_targets

def interpol(signal, desired_len):
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, desired_len, endpoint=False),
        np.linspace(0.0, 1.0, len(signal), endpoint=False),
        signal)
    return resampled_signal

if __name__ == '__main__':

    # No feedback

    block1 = pd.read_csv(r"data\eeg\Training_1.csv", header=None)
    bl_1_samples1 = block1.loc[36526:45328]
    bl_1_samples2 = block1.loc[83421:92819]
    bl_1_samples3 = block1.loc[113067:119148]
    bl_1_samples4 = block1.loc[127966:138086]
    bl_1_samples5 = block1.loc[147872:158022]
    bl_1_targets1 = getTargets(r"data\eye\BLOCK_1\TRAINING\Trial_1.csv", 51, 256, 844,2610)
    bl_1_targets2 = getTargets(r"data\eye\BLOCK_1\TRAINING\Trial_2.csv", 51, 256, 673,2292)
    bl_1_targets3 = getTargets(r"data\eye\BLOCK_1\TRAINING\Trial_3.csv", 51, 256, 943,2574)
    bl_1_targets4 = getTargets(r"data\eye\BLOCK_1\TRAINING\Trial_4.csv", 51, 256, 733,2598)
    bl_1_targets5 = getTargets(r"data\eye\BLOCK_1\TRAINING\Trial_5.csv", 51, 256, 637,2174)
    bl_1_samps = pd.concat([bl_1_samples1, bl_1_samples2, bl_1_samples3, bl_1_samples4, bl_1_samples5]).to_numpy()
    bl_1_targs = np.rint(interpol(np.concatenate((bl_1_targets1, bl_1_targets2, bl_1_targets2, bl_1_targets4, bl_1_targets5)), len(bl_1_samps)))

    block3 = pd.read_csv(r"data\eeg\Training_3.csv", header=None)
    bl_3_samples1 = block3.loc[16557:26620]
    bl_3_samples2 = block3.loc[37056:46653]
    bl_3_samples3 = block3.loc[68025:78037]
    bl_3_samples4 = block3.loc[87211:97344]
    bl_3_samples5 = block3.loc[117232:126776]
    bl_3_targets1 = getTargets(r"data\eye\BLOCK_3\TRAINING\Trial_1.csv", 51, 256, 1088, 2873)
    bl_3_targets2 = getTargets(r"data\eye\BLOCK_3\TRAINING\Trial_2.csv", 51, 256, 789, 2436)
    bl_3_targets3 = getTargets(r"data\eye\BLOCK_3\TRAINING\Trial_3.csv", 51, 256, 1793, 3573)
    bl_3_targets4 = getTargets(r"data\eye\BLOCK_3\TRAINING\Trial_4.csv", 51, 256, 1005, 2804)
    bl_3_targets5 = getTargets(r"data\eye\BLOCK_3\TRAINING\Trial_5.csv", 51, 256, 1836, 3564)
    bl_3_samps = pd.concat([bl_3_samples1, bl_3_samples2, bl_3_samples3, bl_3_samples4, bl_3_samples5]).to_numpy()
    bl_3_targs = np.rint(interpol(np.concatenate((bl_3_targets1, bl_3_targets2, bl_3_targets3, bl_3_targets4, bl_3_targets5)), len(bl_3_samps)))

    multimodal = EEGClassifier(bl_3_samps, bl_3_targs)
    multimodal.fit()
    
    no_feed = EEGClassifier(bl_1_samps, bl_1_targs)
    no_feed.fit()
    no_feed.predict()

    print("multimodal performance: \n")
    multimodal.predictExample(bl_1_samps, bl_1_targs, metric=True)
    print("\n")
    print("no feedback perfomance: \n")
    no_feed.getMetrics()


    pass
