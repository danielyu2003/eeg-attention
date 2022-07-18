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

    def plotSupportVectors(self):
        '''
        Note that the initial data param must be unshuffled, in order, and balanced for this to work
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        support_vectors = self.classifier.support_vectors_

        print(f"Num of support vectors: {len(support_vectors)}")

        class1 = self.data[:len(self.data)//2]
        class2 = self.data[len(self.data)//2:]

        supp_x = support_vectors[:, 0]
        supp_y = support_vectors[:, 1]
        supp_z = support_vectors[:, 2]
        supp_c = support_vectors[:, 3]

        class1_x = class1[:, 0]
        class1_y = class1[:, 1]
        class1_z = class1[:, 2]
        class1_c = class1[:, 3]

        class2_x = class2[:, 0]
        class2_y = class2[:, 1]
        class2_z = class2[:, 2]
        class2_c = class2[:, 3]

        img1 = ax.scatter(supp_x, supp_y, supp_z, c=supp_c, label='support vectors', cmap=plt.summer())
        img2 = ax.scatter(class1_x, class1_y, class1_z, c=class1_c, label='resting', cmap=plt.cool())
        img3 = ax.scatter(class2_x, class2_y, class2_z, c=class2_c, label='concentrating', cmap=plt.hot())

        ax.legend()
        fig.colorbar(img1)
        fig.colorbar(img2)
        fig.colorbar(img3)
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

    quad_1 = rest_test[:, 0]
    quad_one = atten_test[:, 0]

    quad_2 = rest_test[:, 1]
    quad_two = atten_test[:, 1]

    quad_3 = rest_test[:, 2]
    quad_three = atten_test[:, 2]

    quad_4 = rest_test[:, 3]
    quad_four = atten_test[:, 3]


    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

    ax1.plot(range(1000), quad_one, label='concentrating')
    ax1.plot(range(1000), quad_1, label='resting')
    ax2.plot(range(1000), quad_two)
    ax2.plot(range(1000), quad_2)
    ax3.plot(range(1000), quad_three)
    ax3.plot(range(1000), quad_3)
    ax4.plot(range(1000), quad_four)
    ax4.plot(range(1000), quad_4)
    fig.legend()
    plt.show()


    
    samples = np.concatenate((rest_test, atten_test))
    targets = np.concatenate((target1, target2)) 
    
    test = EEGClassifier(samples, targets, test_ratio=0.3, random_seed=42)

    test.fit()
    test.predict()
    test.getMetrics()
    test.plotSupportVectors()
    test.plotAcc()
    # print(test.predictExample([[15,15,15,15]]))
    pass

if __name__ == '__main__':
    # data_path = 'data\\Sub_1_Block_1.csv'

    # test = data(data_path)

    # AF3 = test[0:1000, 3][np.newaxis]
    # AF4 = test[0:1000, 5][np.newaxis]
    # F3 = test[0:1000, 9][np.newaxis]
    # F4 = test[0:1000, 13][np.newaxis]

    # samples = np.hstack((AF3.T, AF4.T, F3.T, F4.T))

    # plotData(data_path, 2000, 3000)

    main()

    pass
