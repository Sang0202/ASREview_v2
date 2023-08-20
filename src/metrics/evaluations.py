from sklearn import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def evaluate_matrix(actual_label, predicted_label):
    """
    Evaluate the model performance using different metrics.
    """
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(actual_label, predicted_label)
    # Calculate the accuracy score
    accuracy = accuracy_score(actual_label, predicted_label)
    # Calculate the precision score per class
    precision_per_class = precision_score(actual_label, predicted_label, average=None)
    # Calculate the recall score per class
    recall_per_class = recall_score(actual_label, predicted_label, average=None)
    # Calculate the F1 score
    f1_per_class = f1_score(actual_label, predicted_label, average=None)
    
    # Print the metrics
    # print("Confusion Matrix: \n", conf_matrix)
    # print("Accuracy: ", accuracy)
    # print("Precision: ", precision_per_class)
    # print("Recall: ", recall_per_class)
    # print("F1 Score: ", f1_per_class)

    return conf_matrix, accuracy, precision_per_class, recall_per_class, f1_per_class