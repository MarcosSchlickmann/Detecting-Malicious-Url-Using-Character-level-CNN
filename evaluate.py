from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import predict
import csv


def format_row(a,b,c,d):
    return ('{:.4f}'.format(a), '{:.4f}'.format(b), c, d)


def evaluate():
    x_normal = predict.load_data("Data/normal_parsed.txt")
    x_anomalous = predict.load_data("Data/anomalous_parsed.txt")
    x_normal_len = len(x_normal)
    x_anomalous_len = len(x_anomalous)
    x = x_normal  + x_anomalous

    no_yy = [0 for i in range(x_normal_len)]
    an_yy = [1 for i in range(x_anomalous_len)]
    y = no_yy + an_yy
    y=to_categorical(y)

    data_train, data_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)
    print('len x_train: {}, len y_train: {}'.format(len(data_test), len(y_test)))
    
    test_data = data_test
    test_y = [ [0] if i[0] == 1 else [1] for i in y_test ]
    test_y = test_y

    predictions = predict.predict(test_data)

    normalized_predictions = []
    default_predictions_normal = []
    default_predictions_anom = []
    for prediction in predictions:
        
        default_predictions_normal.append(round(prediction[0], 4))
        default_predictions_anom.append(round(prediction[1], 4))

        if prediction[0] > 0.80:
            normalized_predictions.append(0)
            continue
        if prediction[1] > 0.80:
            normalized_predictions.append(1)
            continue
        
        normalized_predictions.append(0)

    report = classification_report(test_y, normalized_predictions)
    print(report)

    file = open('Data/predictions.csv', 'w')
    with file:    
        write = csv.writer(file, delimiter=',')
        write.writerows([
            format_row(a,b,c,d) for (a,b,c,d) in 
            zip(default_predictions_normal, default_predictions_anom, normalized_predictions, test_y)
        ])



if __name__ == '__main__':
    evaluate()
    