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
        kernel_choice : str, default='rbf' (radial basis function)
            selects kernel choice for the support vector machine
            Notes:
            ---
            rbf is ideal for nonlinearly seperable data with more samples than features
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

def data(abs_path, type=True):
    df = pd.read_csv(f'{abs_path}', header=None)
    if (type==True):
        return df.to_numpy()[1:,1:]
    else:
        return df.to_numpy()

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


if __name__ == '__main__':
    # data_path = 'data\\Sub_1_Block_1.csv'

    # test = data(data_path)

    df = pd.read_csv(r"C:\Users\yudan\Downloads\Training_11222021_134103.csv", skiprows=1)[1:]

    col1 = df.loc[:, "EyeX"]
    col2 = df.loc[:, "EyeY"]
    col3 = df.loc[:, "EyeZ"]

    eye_samps = pd.concat([col1, col2, col3], axis=1)

    eye_samps = eye_samps.to_numpy()

    upsampled = resample(eye_samps, 120, 256)

    # fig, (ax1, ax2) = plt.subplots(2)

    # ax1.plot(range(len(upsampled[:, 0])), upsampled[:, 0])
    # ax2.plot(range(len(col1)), col1)

    # plt.show()

    transposed = upsampled.T

    sample = transposed[0]

    # which data should be mapped? The resampled or the original?
    # print(np.std(sample[:25]))
    # print(np.std(eye_samps[:12, 0]))

    print(np.mean([0,2,4,6,8,10,12,14]))
    pass
