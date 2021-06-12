#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


glass= pd.read_csv("C:/Users/vinay/Downloads/glass.csv")


# In[3]:


glass


# In[4]:


glass["Type"].value_counts()


# In[5]:


##Checking for the data distribution of the data
data = glass.describe() ;data


# ## As, there is difference in the scale of the values, we normalise the data.

# In[6]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[7]:


norm = norm_func(glass.iloc[:,0:9])
glass1 = glass.iloc[:,9]


# In[8]:


glass


# ## Splitting the data into train and test data using stratified sampling

# In[10]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(norm,glass1,test_size = 0.4,stratify = glass1)


# In[12]:


x_train


# ## Checking the distribution of the labels which are taken

# In[13]:


glass["Type"].value_counts()
y_train.value_counts()
y_test.value_counts()


# ## Building the model

# In[15]:


from sklearn.neighbors import KNeighborsClassifier as KN

model = KN(n_neighbors = 5)
model.fit(x_train,y_train)


# In[17]:


##Finding the accuracy of the model on training data
train_accuracy = np.mean(model.predict(x_train)==y_train)
train_accuracy #76.5%


# In[18]:


##Accuracy on test data
test_accuracy = np.mean(model.predict(x_test)==y_test) 
test_accuracy  ##58.13%


# ## Changing the K value

# In[25]:


model2 = KN(n_neighbors = 9)
model2.fit(x_train,y_train)


# In[27]:


##Accuracy on training data
train_two = np.mean(model2.predict(x_train)==y_train)
train_two   ##72.65%


# In[28]:


##Accuracy on test data
test_two = np.mean(model2.predict(x_test)==y_test) 
test_two ## 60.46%


# In[29]:


# creating empty list variable 
acc = []


# ## Running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# ## Storing the accuracy values 

# In[30]:


for i in range (4,30,1):
    model = KN(n_neighbors = i)
    model.fit(x_train,y_train)
    train_acc = np.mean(model.predict(x_train)==y_train)
    test_acc = np.mean(model.predict(x_test)==y_test)
    acc.append([train_acc, test_acc])


# In[31]:


train_acc


# ## Training accuracy plot

# In[34]:


import matplotlib.pyplot as plt
plt.plot(np.arange(4,30,1),[i[0] for i in acc],'bo-')


# ## Test accuracy plot

# In[36]:


plt.plot(np.arange(4,30,1),[i[1] for i in acc],'ro-')
plt.legend(["train","test"])


# ## Changing the K value

# In[37]:


model3 = KN(n_neighbors = 6)
model3.fit(x_train,y_train)


# In[47]:


pred_train = model3.predict(x_train)
cross_tab = pd.crosstab(y_train,pred_train)
pred_train


# In[43]:


train_accuracy = np.mean(pred_train == y_train)
train_accuracy ##76.56%


# In[46]:


pred_test = model3.predict(x_test)
cross_tab_test = pd.crosstab(y_test,pred_test)
pred_test


# In[48]:


test_accuracy=np.mean(pred_test ==y_test)
test_accuracy## 58.2%

