# Imports
import glob
import numpy as np
from hinge_feature import get_hinge_features
from chaincode_feature import get_chaincode_features

class FeatureExtractor:

    def __init__(self):
        pass


    def save_data(self):
    
        # Check if the feature vector is already saved before, if not, save it
        try:
            np.load('../Saved Data/training_feature_vector.npy')
            print("The feature vector of the training dataset is saved in the 'Saved Data' directory\n")
        except: 
            print("The feature vector of the training dataset is not saved in the 'Saved Data' directory\n")
            print("The feature vector of the training dataset is being generated...\n")
            X, _ = self.__read_training_dataset(False)
            np.save('../Saved Data/training_feature_vector.npy', X)

        # Check if the label vector is already saved before, if not, save it
        try:
            np.load('../Saved Data/training_data_labels_and_image_names.npy')
        except:
            _, Y = self.__read_training_dataset(True)
            np.save('../Saved Data/training_data_labels_and_image_names.npy', Y)


    # Read CMP_23 dataset
    # This function reads the path of all images not the images themselves and then extract the features of the images one by one to save the program memory
    def __read_training_dataset(self, features_saved: bool):
        X_train = []
        Y_train = []

        # Male training data acquisition
        for file_name in sorted(glob.glob('../Training Dataset/CMP_23/Males/*.jpg')):
            X_train.append(file_name)
            Y_train.append(1)

        # Female training data acquisition
        for file_name in sorted(glob.glob('../Training Dataset/CMP_23/Females/*.jpg')):
            X_train.append(file_name)
            Y_train.append(0)

        Y_train = np.concatenate((np.reshape(np.asarray(Y_train), (len(Y_train), 1)), np.reshape(np.asarray(X_train), (len(X_train), 1))), axis = 1)
        if features_saved == False:
            X_train = self.get_feature_vector(X_train)
        return X_train, Y_train


    # Read the test dataset from the given path and convert it to a feature vector
    def read_test_dataset(self, test_dataset_path):
        X_test = []
        Y_test = []
        for file_name in sorted(glob.glob(test_dataset_path + "*.jpg")):
            X_test.append(file_name)
        
        with open(test_dataset_path + "labels.txt") as f:
            Y_test = f.readlines()
        for i in range(len(Y_test)):
            Y_test[i] = int(Y_test[i])
        
        Y_test = np.concatenate((np.reshape(np.asarray(Y_test), (len(Y_test), 1)), np.reshape(np.asarray(X_test), (len(X_test), 1))), axis = 1)
        X_test = self.get_feature_vector(X_test)
        return X_test, Y_test


    # This function returns the feature vector of a given list of image paths
    def get_feature_vector(self, X):
        hinge = []
        chaincode = []
        for i in range(len(X)):
            hinge.append(get_hinge_features(image_path = X[i]))
            chaincode.append(get_chaincode_features(image_path = X[i]))

        hinge = np.asarray(hinge)
        chaincode = np.asarray(chaincode)
        feature_vector = np.concatenate((hinge, chaincode), axis = 1)
        return feature_vector