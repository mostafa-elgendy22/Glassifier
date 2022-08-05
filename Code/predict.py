import sys
from sklearn import metrics
from classify import *

def main():

    ########################################################################################################
    # Input validation
    if len(sys.argv) != 4:
        print(sys.argv)
        print("Invalid arguments.\n")
        sys.exit(0)

    test_dataset_size = None
    output_dir_path = sys.argv[1]
    if output_dir_path[-1] != '/' and output_dir_path[-1] != '\\':
        output_dir_path += '/'

    if sys.argv[2] != '-s' and sys.argv[2] != '-k':
        print("Invalid arguments.\n")
        sys.exit(0)
    
    if sys.argv[2] == '-s':
        SPLIT_TRAINING_DATA = True
        test_dataset_path = None
        try:
            test_dataset_size = float(sys.argv[3])
            if test_dataset_size <= 0.001 or test_dataset_size >= 0.999:
                raise ValueError
        except:
            print("Invalid arguments.\n")
            sys.exit(0)
    else:
        SPLIT_TRAINING_DATA = False
        test_dataset_path = sys.argv[3]
    ########################################################################################################
    
    # Create the gender classifier with the input parameters
    classifier = Classifier(SPLIT_TRAINING_DATA, test_dataset_path, test_dataset_size)
    Y_predicted, Y_test = classifier.classify()

    classification_accuracy = round(metrics.accuracy_score(Y_test[:, 0], Y_predicted) * 100, 3)

    # Replace the the '0' and '1' with 'Male' and 'Female' for printing the results
    Y_predicted[Y_predicted == '0'] = 'Female'
    Y_predicted[Y_predicted == '1'] = 'Male'
    Y_test[:, 0][Y_test[:, 0] == '0'] = 'Female'
    Y_test[:, 0][Y_test[:, 0] == '1'] = 'Male'

    # Print the results in the output file
    results_file = open(output_dir_path + 'classification_results.txt', 'w')
    dash = '-' * 60
    results_file.write('{:<12s}\t{:^12s}\t\t{:^12s}\n'.format("Image Name", "Predicted Label", "Actual Label"))
    results_file.write(dash + '\n')

    for i in range(len(Y_predicted)):
        # Find the index of the last occurence of the '\' character in the image path to find the name of the image
        idx = Y_test[i][1].rfind('\\') + 1
        results_file.write('{:<12s}\t{:^12s}\t\t{:^12s}\n'.format(Y_test[i][1][idx:], Y_predicted[i], Y_test[i][0]))
    results_file.close()

    print(f"The classification accuracy = {classification_accuracy}%.\n")
    print("The results are successfully written to the file 'classification_results.txt' in the given output directory.\n")


print(" ")
main()


# python predict.py 'path/to/output' -s 0.35
# python predict.py 'path/to/output' -k 'path/to/test/dataset' 