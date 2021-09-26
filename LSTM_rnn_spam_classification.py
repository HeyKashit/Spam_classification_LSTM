#!/usr/bin/env python
# coding: utf-8

# # Spam Classification using LSTM(RNN)

# In[1]:


import numpy as np 
import pandas as pd 


# In[2]:


from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', 1000)

messages = pd.read_csv('spam.csv', encoding='latin-1')
messages = messages.drop(labels = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
messages.columns = ["label", "text"]
labels = np.where(messages['label']=='spam', 1, 0)

X_train, X_test, y_train, y_test = train_test_split(messages['text'],
                                                    labels, test_size=0.2)

messages.head()


# In[3]:


def remove_punctuation(text):
    import string
    text = "".join([char for char in text if char not in string.punctuation])
    return text


# In[4]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[5]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)


# In[6]:


print(X_train)


# In[7]:


X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)


# In[8]:


X_train_seq_padded = pad_sequences(X_train_seq, 50)
X_test_seq_padded = pad_sequences(X_test_seq, 50)


# In[9]:


X_train_seq[0]


# In[10]:


from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential


# In[11]:


model = Sequential()

model.add(Embedding(len(tokenizer.index_word)+1, 32))
model.add(LSTM(32))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# In[12]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[13]:


history = model.fit(X_train_seq_padded, y_train, 
                    batch_size=32, epochs=10,
                    validation_data=(X_test_seq_padded, y_test))


# In[14]:


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training', 'Validation'])
plt.title('Training and validation accuracy')
plt.xlabel('epoch')


# In[ ]:




