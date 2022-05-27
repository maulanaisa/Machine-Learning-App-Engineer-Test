import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import re
import os

dir_path = r"src\Q3"
regex = r"([A-Z]+)\s*(.+)\n*"
label=[]
data=[]

trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

training_split = 0.7
vocab_size = 4000
max_length = 100
embedding_dim = 16

# Data Preprocessing
def extractData(dir_path):
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)) and path.endswith(".txt"):
            with open(os.path.join(dir_path, path),"r") as openfileobject:
                for line in openfileobject:
                    match = re.match(regex,line)
                    if match :
                        label.append(match.group(1))
                        data.append(match.group(2))
            openfileobject.close()
    return

# Data Preprocessing
def prepareData(label,data) :
  training_size = int(training_split*len(data))

  label = np.array(integerMapping(label))
  training_labels = label[0:training_size]
  testing_labels = label[training_size:]

  training_sentences = data[0:training_size]
  testing_sentences = data[training_size:]

  tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)
  tokenizer.fit_on_texts(training_sentences)
  word_index = tokenizer.word_index
  print(f'number of words in word_index: {len(word_index)}')

  training_sequences = tokenizer.texts_to_sequences(training_sentences)
  training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

  testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
  testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

  return training_padded, training_labels, testing_padded, testing_labels

# Integer Encoding
def integerMapping(label) :
  temp = {}
  output = []
  i=0
  for x in label :
    if x not in temp :
      temp[x] = i
      i+=1
    output.append(temp[x])
  return output

# Train Model
def trainModel(X_train, y_train, X_test, y_test) :
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
  ])
  model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
    
if __name__ == '__main__':
    extractData(dir_path)
    X_train, y_train, X_test, y_test = prepareData(label,data)
    trainModel(X_train, y_train, X_test, y_test)