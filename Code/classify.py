# Imports
import numpy as np
from feature_extractor import FeatureExtractor
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Classifier:

    def __init__(self, SPLIT_TRAINING_DATA: bool, test_dataset_path: str, test_dataset_size: float):
        self.__SPLIT_TRAINING_DATA = SPLIT_TRAINING_DATA
        self.__test_dataset_path = test_dataset_path
        self.__test_dataset_size = test_dataset_size

        # Extract the feature vector of the training dataset and save them if they are not already saved
        self.__feature_extractor = FeatureExtractor()
        self.__feature_extractor.save_data()

        # Load the saved features and labels of the training dataset
        training_feature_vector = np.load('../Saved Data/training_feature_vector.npy')
        Y = np.load('../Saved Data/training_data_labels_and_image_names.npy')


        self.__X_train, self.__X_test, self.__Y_train, self.__Y_test = self.__get_datasets(X = training_feature_vector, Y = Y)
        self.__train_model()


    def __get_datasets(self, X, Y):
        if self.__SPLIT_TRAINING_DATA:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = self.__test_dataset_size)
        
            # Training dataset information
            print(f"The size of the training dataset is {X_train.shape[0]} ({Y_train[:, 0].astype(np.int8).sum()} males, {Y_train.shape[0] - Y_train[:, 0].astype(np.int8).sum()} females).\n")
            
            # Test dataset information
            print(f"The size of the test dataset is {X_test.shape[0]} ({Y_test[:, 0].astype(np.int8).sum()} males, {Y_test.shape[0] - Y_test[:, 0].astype(np.int8).sum()} females).\n")

            return X_train, X_test, Y_train, Y_test
        else:
            X_train, Y_train = X, Y
            X_test, Y_test = self.__feature_extractor.read_test_dataset(self.__test_dataset_path)
            # Training dataset information
            print(f"The size of the training dataset is {X_train.shape[0]} ({Y_train[:, 0].astype(np.int8).sum()} males, {Y_train.shape[0] - Y_train[:, 0].astype(np.int8).sum()} females).\n")
            
            # Test dataset information
            print(f"The size of the test dataset is {X_test.shape[0]} ({Y_test[:, 0].astype(np.int8).sum()} males, {Y_test.shape[0] - Y_test[:, 0].astype(np.int8).sum()} females).\n")

            return X_train, X_test, Y_train, Y_test


    def __train_model(self):
        self.__sc = StandardScaler()
        self.__sc.fit(self.__X_train)
        self.__X_train_std = self.__sc.transform(self.__X_train)
        self.__classifier = svm.SVC(C = 10)
        self.__classifier.fit(self.__X_train_std, self.__Y_train[:, 0])


    def classify(self):
        X_test_std = self.__sc.transform(self.__X_test)
        Y_predicted = self.__classifier.predict(X_test_std)
        return Y_predicted, self.__Y_test