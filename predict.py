import keras
from keras import models
import io
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random


model = models.load_model('model/clcnn-model.h5')
model.load_weights('model/clcnn-weights.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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
# to_predict = [x_normal[random.randint(0, len(x_normal))], x_anomalous[random.randint(0, len(x_anomalous))]]
to_predict = ["gethttp://localhost:8080/tienda1/publico/anadir.jsp?id=2&nombre=jam�n+ib�rico&precio=85", "gethttp://localhost:8080/teste.jsp?id=20&nombre=';+drop+table"]

print(to_predict)

tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i + 1
tk.word_index = char_dict.copy()
tk.word_index[tk.oov_token] = max(char_dict.values()) + 1


seq = tk.texts_to_sequences(to_predict)

xx = pad_sequences(seq, maxlen=1014, padding='post')
xxf = np.array(xx, dtype='float32')

prediction = model(xxf)

print(len(xxf))
print(xxf)
print(prediction)
print("req0:: norm: {:.4f}, anom: {:.4f}".format(prediction[0][0], prediction[0][1]))
print("req1:: norm: {:.4f}, anom: {:.4f}".format(prediction[1][0], prediction[1][1]))
