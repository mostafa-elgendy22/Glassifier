# Imports
from hinge_feature import get_hinge_features
from chaincode_feature import get_chaincode_features
import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics


class Classifier:
    
    def __init__(self, SPLIT_TRAINING_DATA: bool, test_dataset_path = None):
        self.__SPLIT_TRAINING_DATA = SPLIT_TRAINING_DATA
        self.__test_dataset_path = test_dataset_path

        # datasets
        self.__X_train, self.__X_test, self.__Y_train, self.__Y_test = self.__get_datasets()
        self.__training_feature_vector = self.get_feature_vector(self.__X_train)
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


    def __get_datasets(self):
        X, Y = self.__read_training_dataset()
        if self.__SPLIT_TRAINING_DATA:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
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
            feature_vector = []
            for i in range(len(X)):
                hinge = get_hinge_features(image_path = X[i])
                chaincode = get_chaincode_features(image_path = X[i])
                feature_vector.append(np.concatenate((hinge, chaincode)))
                
            feature_vector = np.asarray(feature_vector)
            return feature_vector


    def __train_model(self):
        svm_classifier = svm.SVC(kernel='linear')
        svm_classifier.fit(self.__training_feature_vector, self.__Y_train)
        self.__classifier = svm_classifier


    def classify(self, feature_vector = None):
        if self.__SPLIT_TRAINING_DATA:
            Y_predicted = self.__classifier.predict(self.get_feature_vector(self.__X_test))
            return (Y_predicted, metrics.accuracy_score(self.__Y_test, Y_predicted))
        else:
            return self.__classifier.predict(feature_vector)