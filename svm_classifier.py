from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os

os.chdir(os.path.dirname(__file__))

class EEGClassifier:
    
    def __init__(self, data, target, test_ratio=0.25, random_seed=None, shuffle_choice=True, C_val=1.0, kernel_choice='linear', gamma_choice = 'scale'):
        '''
        @param kernel_choice : str
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
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data, self.target,test_size=test_ratio,random_state=random_seed,shuffle=shuffle_choice)
        
        self.y_pred = np.zeros(self.y_test.size)
        pass

    def fit(self):
        self.classifier.fit(self.x_train, self.y_train)
        pass

    def predict(self):
        self.y_pred = self.classifier.predict(self.x_test)
        pass

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

def data(abs_path):
    df = pd.read_csv(f'{abs_path}', header=None)
    return df.to_numpy()[1:,1:]

def test():
    from sklearn import datasets
    cancer = datasets.load_breast_cancer()
    test = EEGClassifier(cancer.data, cancer.target, 0.3, 109)
    test.fit()
    test.predict()
    test.getMetrics()
    test.plotAcc()
    pass

def main():
    '''
    target1, target2, the resting dataset, and the attention dataset should all have the same length/number of samples
    0 represents resting
    1 represents attentive
    '''
    
    target1 = np.zeros(1000)
    target2 = np.ones(1000)
    
    rest_test = 10 * (np.random.rand(1000, 4) - 0.5)
    # generates a fake sample of small amplitudes between -5 and 5
    atten_test = 100 * (np.random.rand(1000, 4) - 0.5)
    # generated a fake sample of large amplitudes between -50 and 50
    
    concatTarget1 = target1.reshape(-1, 1)
    concatTarget2 = target2.reshape(-1, 1)
    
    test1 = np.concatenate((concatTarget1, rest_test), axis=1)
    test2 = np.concatenate((concatTarget2, atten_test), axis=1)
    
    all_samples = np.concatenate((test1, test2))
    
    np.random.shuffle(all_samples)
    
    seperated = np.hsplit(all_samples, [1])
    
    targets = seperated[0].flatten()
    samples = seperated[1]
    
    test = EEGClassifier(samples, targets, 0.3, 109)
    test.fit()
    test.predict()
    test.getMetrics()
    test.plotAcc()
    pass

if __name__ == '__main__':
    # main()
    test()
    pass
