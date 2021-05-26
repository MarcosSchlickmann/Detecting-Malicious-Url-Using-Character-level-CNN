import keras
from keras import models
import io
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import json


def load_data(file):
    with io.open(file, 'r', encoding="utf8") as f:
        data = f.readlines()
    result = []
    for d in data:
        d = d.strip()
        if (len(d) > 0):
            result.append(d)
    return result


def prepare_tokenizer():
	tk = Tokenizer(char_level=True)

	alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
	char_dict = {}
	for i, char in enumerate(alphabet):
	    char_dict[char] = i + 1
	tk.word_index = char_dict.copy()
	tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

	return tk


def load_tokenizer():
	with open('Data/tokenized-chars.json') as json_file:
		tokenizer_conf = json.load(json_file)

	tokenizer = tokenizer_from_json(tokenizer_conf)
	return tokenizer


def predict(requests):
	model = models.load_model('model/clcnn-model.h5')
	model.load_weights('model/clcnn-weights.h5')
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	# tk = prepare_tokenizer()
	tk = load_tokenizer()
	seq = tk.texts_to_sequences(requests)

	xx = pad_sequences(seq, maxlen=1000, padding='post')
	xxf = np.array(xx)

	prediction = model.predict(xxf, verbose=1)
	return prediction
	# print(len(xxf))
	# print(xxf)
	# print("req0:: norm: {:.4f}, anom: {:.4f}".format(prediction[0][0], prediction[0][1]))
	# print("req1:: norm: {:.4f}, anom: {:.4f}".format(prediction[1][0], prediction[1][1]))
