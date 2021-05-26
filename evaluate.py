from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import predict
import csv


def format_row(a,b,c):
    return ('{:.4f}'.format(a), b, c)


def evaluate():
    x_normal = predict.load_data("Data/normal_parsed.txt")
    x_anomalous = predict.load_data("Data/anomalous_parsed.txt")
    x_normal_len = len(x_normal)
    x_anomalous_len = len(x_anomalous)
    x = x_normal  + x_anomalous

    no_yy = [0 for i in range(x_normal_len)]
    an_yy = [1 for i in range(x_anomalous_len)]
    y = no_yy + an_yy


    data_train, data_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)
    print('len x_train: {}, len y_train: {}'.format(len(data_test), len(y_test)))

    test_data = data_test
    test_y = y_test

    predictions = predict.predict(test_data)

    normalized_predictions = []
    default_predictions = []
    
    for prediction in predictions:
        
        default_predictions.append(round(prediction[0], 4))

        if prediction[0] > 0.80:
            normalized_predictions.append(1)
            continue
        
        normalized_predictions.append(0)

    report = classification_report(test_y, normalized_predictions)
    print(report)
    tn, fp, fn, tp = confusion_matrix(test_y, normalized_predictions).ravel()
    print("TN: {}, FP: {}, FN: {}, TP: {}".format(tn, fp, fn, tp))

    file = open('Data/predictions2.csv', 'w')
    with file:    
        write = csv.writer(file, delimiter=',')
        write.writerows([
            format_row(a,b,c) for (a,b,c) in 
            zip(default_predictions, normalized_predictions, test_y)
        ])



if __name__ == '__main__':
    evaluate()
    