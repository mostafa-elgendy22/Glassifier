# Imports
from hinge_feature import get_hinge_features
from chaincode_feature import get_chaincode_features
import cv2
import numpy as np
import glob
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES_SAVED = True

class Classifier:
    
    def __init__(self, SPLIT_TRAINING_DATA: bool, test_dataset_path = None):
        self.__SPLIT_TRAINING_DATA = SPLIT_TRAINING_DATA
        self.__test_dataset_path = test_dataset_path

        X, Y = self.__read_training_dataset()
        
        if FEATURES_SAVED == False:
            hinge, chaincode = self.get_feature_vector(X)
            np.save('../Features/hinge.npy', hinge)
            np.save('../Features/chaincode.npy', chaincode)    
        else:
            hinge = np.load('../Features/hinge.npy')
            chaincode = np.load('../Features/chaincode.npy')

        training_feature_vector = np.concatenate((hinge, chaincode), axis = 1)
        self.__X_train, self.__X_test, self.__Y_train, self.__Y_test = self.__get_datasets(X = training_feature_vector, Y = Y)
        print(f"The dimensions of the feature vector of the training data = {self.__X_train.shape}")
        self.__train_model()


    def __read_training_dataset(self):
        x_train = []
        y_train = []
        # Male training data acquisition
        for file_name in sorted(glob.glob('../Training Dataset/CMP_23/Males/*.jpg')):
            # img = cv2.imread(file_name)      # cv2.imread reads images in RGB format
            x_train.append(file_name)
            y_train.append(1)

        # Female training data acquisition
        for file_name in sorted(glob.glob('../Training Dataset/CMP_23/Females/*.jpg')):
            # img = cv2.imread(file_name)      # cv2.imread reads images in RGB format
            x_train.append(file_name)
            y_train.append(0)

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        return x_train, y_train


    # for project submission
    def read_test_dataset(self):
        x_test = []
        for file_name in sorted(glob.glob(self.__test_dataset_path + "*.jpg")):
            img = cv2.imread(file_name)      # cv2.imread reads images in RGB format
            x_test.append(img)
        x_test = np.asarray(x_test)
        return x_test


    def __get_datasets(self, X, Y):
        
        if self.__SPLIT_TRAINING_DATA:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
            print(f"Size of the training dataset is: {X_train.shape[0]} ({np.sum(Y_train)} males, {X_train.shape[0] - np.sum(Y_train)} females).")
            print(f"Size of the test dataset is: {X_test.shape[0]} ({np.sum(Y_test)} males, {X_test.shape[0] - np.sum(Y_test)} females).")
            return X_train, X_test, Y_train, Y_test
        else:
            X_train, Y_train = X, Y
            print(f"Size of the training dataset is: {X_train.shape[0]} ({np.sum(Y_train)} males, {X_train.shape[0] - np.sum(Y_train)} females).")
            return X_train, None, Y_train, None


    # get training dataset feature vector
    def get_feature_vector(self, X, single_data_item = False):
        if single_data_item:
            feature_vector = np.concatenate(get_hinge_features(image = X), get_chaincode_features(image = X))
            return feature_vector
        else:
            hinge = []
            chaincode = []
            for i in range(len(X)):
                hinge.append(get_hinge_features(image_path = X[i]))
                chaincode.append(get_chaincode_features(image_path = X[i]))
                
            hinge = np.asarray(hinge)
            chaincode = np.asarray(chaincode)
            return hinge, chaincode


    def __train_model(self):
        # svm_classifier = svm.SVC(kernel='linear')
        # svm_classifier.fit(self.__training_feature_vector, self.__Y_train)
        # self.__classifier = svm_classifier
        sc = StandardScaler()
        sc.fit(self.__X_train)
        self.__X_train_std = sc.transform(self.__X_train)
        self.__sc = sc
        self.__classifier = svm.SVC(C = 1.0, random_state = 1, kernel = 'linear')
        self.__classifier.fit(self.__X_train_std, self.__Y_train)
        

    def classify(self, feature_vector = None):
        if self.__SPLIT_TRAINING_DATA:
            X_test_std = self.__sc.transform(self.__X_test)
            Y_predicted = self.__classifier.predict(X_test_std)
            # Y_predicted = cross_val_predict(self.__classifier, self.__X_test, self.__Y_test, cv = 10)  
            return (Y_predicted, metrics.accuracy_score(self.__Y_test, Y_predicted))
        else:
            return self.__classifier.predict(feature_vector)