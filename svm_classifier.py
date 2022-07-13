from sklearn import svm
from sklearn import metrics


class EEGClassifier:

    def __init__(self, kernel_choice='linear'):
        '''
        @param kernel_choice : str
            selects kernel choice for the support vector machine
            see the official scikit-learn website for more details 
            (https://scikit-learn.org/stable/modules/svm.html)
        '''

        self.classifier = svm.SVC(kernel=kernel_choice)

        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def getMetrics(self):
        pass

if __name__ == '__main__':
    print("starting")
    test = EEGClassifier()