#!/usr/bin/env python
# coding: utf-8

# In[3]:

# function for computing accuracy
def compute_accuracy(Y_true, Y_pred):  
    correctly_predicted = 0  
    # iterating over every label and checking it with the true sample  
    for true_label, predicted in zip(Y_true, Y_pred):  
        if true_label == predicted:  
            correctly_predicted += 1  
    # computing the accuracy score  
    accuracy_score = correctly_predicted / len(Y_true)  
    return accuracy_score  


# In[4]:


import pickle
import numpy as np
from RandomForestManual import RandomForest

# Loading the data and converting the ‘data’ and ‘label’ list into numpy arrays:
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Shuffle data and labels
indices = np.random.permutation(len(data))  # Generate shuffled indices
data = data[indices]
labels = labels[indices]

# Manual split
test_size = 0.2  # Proportion of test data
split_index = int(len(data) * (1 - test_size))  # Calculate split index

x_train, x_test = data[:split_index], data[split_index:]  # Split data
y_train, y_test = labels[:split_index], labels[split_index:]  # Split labels

# Creating the model using random forest classifier and training the model with the training dataset
model = RandomForest()

model.fit(x_train, y_train)
# Making predictions on new data points
y_predict = model.predict(x_test)
# Computing the accuracy of the model
score = compute_accuracy(y_test,y_predict)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()


# In[ ]:




