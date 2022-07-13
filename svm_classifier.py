from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class EEGClassifier:
    
    def __init__(self, data, target, test_ratio=0.25, random_seed=None, shuffle_choice=True, C_val=1.0, kernel_choice='linear', gamma_choice = 'scale'):
        '''
        @param kernel_choice : str
            selects kernel choice for the support vector machine
            see the official scikit-learn website for more details 
            (https://scikit-learn.org/stable/modules/svm.html)
        @param data : numpy int[]
            a 2d array/matrix of shape (num of samples, num of features)
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
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data, self.target,test_size=test_ratio,random_state=random_seed,shuffle=shuffle_choice)
        
        self.y_pred = np.zeros(self.x_test.size)
        pass

    def fit(self):
        self.classifier.fit(self.x_train, self.y_train)
        pass

    def predict(self):
        self.y_pred = self.classifier.predict(self.x_test)
        pass

    # def getMetrics(self):
    #     print("Accuracy:",metrics.accuracy_score(self.y_test, self.y_pred))
    #     print("Precision:",metrics.precision_score(self.y_test, self.y_pred))
    #     print("Recall:",metrics.recall_score(self.y_test, self.y_pred))
    #     pass
    
def main():
    print("starting")
    X = np.arange(10).reshape((5, 2))
    y = range(5)
    test = EEGClassifier(X, y,0.33,42)
    print("finished")
    pass

if __name__ == '__main__':
    main()
