import numpy as np
import pandas as pd
import io

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#DATA PREPROCESSING SECTION BEGINS
#loading the parsed files
def load_data(file):
    with io.open(file, 'r', encoding="utf8") as f:
        data = f.readlines()
    result = []
    for d in data:
        d = d.strip()
        if (len(d) > 0):
            result.append(d)
    return result


x_normal = load_data("Data/normal_parsed.txt")
x_anomalous = load_data("Data/anomalous_parsed.txt")
x_normal_len = len(x_normal)
x_anomalous_len = len(x_anomalous)
x = x_normal  + x_anomalous
# print('x')
# print(x[:10])

no_yy = [0 for i in range(x_normal_len)]
an_yy = [1 for i in range(x_anomalous_len)]
y = no_yy + an_yy
# print('y')
# print(y[:10])
y=to_categorical(y)
# print('yc')
# print(y[:10])
### x_train, x_test, y_train, y_test
data_train, data_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)
print('len x_train: {}, len y_train: {}'.format(len(data_test), len(y_test)))

#Loading dataset
## load_data = pd.read_csv('Data/shuf_csic.csv', header=None,sep='\t')

##Loading saved model
#saved_model=load_model('model_final_version_2.h5')

#Initializing data_values
## data=load_data[0].values
## data=[s.lower() for s in data]

#Initializing corresponding label for data_values
## outcome=load_data[1].values

#Hot Embedding labels
## outcome=to_categorical(outcome)

#Splitting the dataset into training and testing data
##data_train, data_test, y_train, y_test = train_test_split(
##        data, outcome, test_size=0.20, random_state=1000)

#Initializing tokenizer for character-level splitting
tk = Tokenizer(char_level=True)

##Creating a vocabulary set based on training data
#tk.fit_on_texts(data_train)

#Creating a vocabulary set of 69 characters manually
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i + 1
tk.word_index = char_dict.copy()
tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

#Converting characters of each training and 
#testing data observations to their corresponding values in vocabulary set
train_sequences = tk.texts_to_sequences(data_train)
test_sequences = tk.texts_to_sequences(data_test)

#Add padding to make each observations of training 
#and testing data to a fixed length input of 1014
train_data = pad_sequences(train_sequences, maxlen=1000, padding='post')
test_data = pad_sequences(test_sequences, maxlen=1000, padding='post')

#Converting each observation of training and testing data to arrays
train_data = np.array(train_data)
test_data = np.array(test_data)

#DATA PREPROCESSING SECTION ENDS

