from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os

os.chdir(os.path.dirname(__file__))

class EEGClassifier:
    
    def __init__(self, data, target, test_ratio=0.25, random_seed=None, C_val=1.0, kernel_choice='rbf', gamma_choice = 'scale'):
        '''
        @param kernel_choice : str
            default = 'rbf' (radial basis function)
                ideal for nonlinearly seperable data
            selects kernel choice for the support vector machine
            see the official scikit-learn website for more details 
            (https://scikit-learn.org/stable/modules/svm.html)
        @param data : numpy int[]
            an array/matrix of shape (num of samples/y, num of features/x)
        @param target : numpy int[]
            a 1d array of length (num of samples), where each sample is classified as a 1 or a 0
        @param test_size : float
            a decimal value between 0 and 1 representing what percent of the dataset to use for testing
        @param random_state : int
            an optional value which selects samples for training/testing from the given seed
        @param shuffle_choice : bool
            an optional value which determines whether or not the data is shuffled before splitting
        '''
        self.classifier = svm.SVC(C=C_val, kernel=kernel_choice, gamma=gamma_choice)
        self.data = data
        self.target = target
        # splitting the testing and training dataset could be done manually, 
        # just keep in mind the correct format for numpy array inputs, etc.
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data, self.target,test_size=test_ratio,random_state=random_seed)
        
        self.y_pred = np.zeros(self.y_test.size)
        pass

    def fit(self):
        self.classifier.fit(self.x_train, self.y_train)
        pass

    def predict(self):
        self.y_pred = self.classifier.predict(self.x_test)
        pass

    def predictExample(self, example):
        '''
        @param example : float[][]
            contains an array of samples (which are arrays of floats matching the feature dimensions)
        '''
        return self.classifier.predict(example)

    def getMetrics(self):
        print("Accuracy:",metrics.accuracy_score(self.y_test, self.y_pred))
        print("Precision:",metrics.precision_score(self.y_test, self.y_pred))
        print("Recall:",metrics.recall_score(self.y_test, self.y_pred))
        pass
    
    def plotAcc(self):
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

def plotData(abs_path, startInd, endInd, choi_type=True):
    test = data(abs_path, choi_type)
    
    AF3 = test[startInd:endInd, 3]
    AF4 = test[startInd:endInd, 5]
    F3 = test[startInd:endInd, 9]
    F4 = test[startInd:endInd, 13]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

    ax1.plot(range(endInd-startInd), AF3)
    ax2.plot(range(endInd-startInd), AF4)
    ax3.plot(range(endInd-startInd), F3)
    ax4.plot(range(endInd-startInd), F4)
    plt.show()

def test():
    from sklearn import datasets
    cancer = datasets.load_breast_cancer()
    test = EEGClassifier(cancer.data, cancer.target, 0.3, 109)
    test.fit()
    test.predict()
    test.getMetrics()
    test.plotAcc()
    pass

if __name__ == '__main__':
    # data_path = 'data\\Sub_1_Block_1.csv'

    # test = data(data_path)



    pass
