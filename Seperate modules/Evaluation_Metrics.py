def evaluation_metric(prediction, actual):  # calculate accuracy,precision,recall,f1_score

    import numpy as np
    import pandas as pd

    prediction_onehot = np.zeros_like(prediction)  # array of zeros with same shape of prediction
    prediction_onehot[
        prediction.argmax(0), np.arange(prediction.shape[1])] = 1  # all equal zero except maximum =1 [0 0 0 1 0 0 0]

    predictedClass = prediction_onehot.T  # transpose one hot code
    one_dim_predicted_class = []
    for i in range(len(predictedClass)):
        one_dim_predicted_class.append(
            np.argmax(predictedClass[i]))  # convert one hot code to 1D array (returns the index of the max value)

    actual_list = actual.tolist()  # convert the array to list
    sum = 0
    for i in range(len(actual_list)):
        if actual_list[i] == one_dim_predicted_class[i]:
            sum += 1  # for every true prediction
    print("accuracy = ",
          (sum / len(actual_list)) * 100)  # calculate the accuracy (true predictions/total no. of examples)
    currentDataClass = actual_list
    classes = set(currentDataClass)  # get the available labels without repetition

    y_actu = pd.Series(currentDataClass)  # construct a series of true labels
    y_pred = pd.Series(one_dim_predicted_class)  # construct a series of predictions
    conf_matrix = pd.crosstab(y_actu, y_pred)  # construct confusion matrix
    cm = conf_matrix.values

    true_pos = np.diag(cm)  # get true positives
    false_pos = np.sum(cm, axis=0) - true_pos  # get false positives
    false_neg = np.sum(cm, axis=1) - true_pos  # get false negatives

    precision = true_pos.sum() / (true_pos.sum() + false_pos.sum() + 0.00001)  # P=TP/TP+FP
    recall = true_pos.sum() / (true_pos.sum() + false_neg.sum())  # R=TP/TP+FN
    f1_score = 2 * ((precision * recall) / (precision + recall))  # F1Score=P*R/P+R
    print("precision = ", precision * 100)
    print("f1_score = ", f1_score * 100)
    print("recall = ", recall * 100)