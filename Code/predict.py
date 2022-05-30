import sys
import time
from handwritting_based_gender_classifier import *

SPLIT_TRAINING_DATA = True      # Split the CMP_23 dataset into training and test data

def main():

    if SPLIT_TRAINING_DATA == True:        # for development
        classifier = Classifier(SPLIT_TRAINING_DATA)
        (predictions, accuracy) = classifier.classify()
        # print(f"Model Accuracy = {round(accuracy * 100, 2)}%.")
        print(round(accuracy * 100, 2))
        
    else:                                   # for project submission
        test_dataset_path = sys.argv[1]
        output_dir_path = sys.argv[2]
        classifier = Classifier(SPLIT_TRAINING_DATA)
        X_test = classifier.read_test_dataset(test_dataset_path)
        execution_times = []
        classification_results = []

        for i in range(len(X_test)):
            start = time.time()
            feature_vector = classifier.get_feature_vector(X_test[i], single_data_item = True)
            classification_results.append(classifier.classify(feature_vector)[0])
            end = time.time()
            execution_time = round(end - start, 2)
            if execution_time == 0.0:
                execution_time == 10 ** -3
            execution_times.append(execution_time)

        results_file = open(output_dir_path + 'results.txt', 'w')
        times_file = open(output_dir_path + 'times.txt', 'w')

        for i in range(len(execution_times)):
            results_file.write(str(classification_results[i]))
            times_file.write(str(execution_times[i]))
            if i != len(execution_times) - 1:
                results_file.write('\n')
                times_file.write('\n')



main()