#!/usr/bin/env python
# coding: utf-8

# In[116]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[117]:


#IMPORT DATASET
df=pd.read_csv('car.data',header=None,delimiter=',',names=['buying','maint','doors','persons','lug_boot','safety','CAR'])
df.head()


# In[118]:


#CONVERTING TO NUMERICAL VALUES 
from sklearn.preprocessing import LabelEncoder
LEC=LabelEncoder()
df["buying"]=LEC.fit_transform(df["buying"])
df["maint"]=LEC.fit_transform(df["maint"])
df["lug_boot"]=LEC.fit_transform(df["lug_boot"])
df["safety"]=LEC.fit_transform(df["safety"])
df["doors"]=LEC.fit_transform(df["doors"])
df["persons"]=LEC.fit_transform(df["persons"])

df.head()


# In[119]:


#CHECKING IF ALL ARE CONVERTED INTO NUMERICAL OR NOT EXCEPT DEPENDENT VARIABLE
df.applymap(np.isreal).head()


# In[120]:


#DROPPING THE DEPENDENT VARIABLE
x=df.drop(["CAR"],axis=1)
y=df["CAR"]


# In[121]:


#IMPORTING TRAIN_TEST_SPLIT
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[122]:


#find optimum value of K
from sklearn.neighbors import KNeighborsClassifier
from math import sqrt
from sklearn.metrics import mean_squared_error
rmse=[]
y1=LEC.fit_transform(df["CAR"])
for k in range(20):
    k=k+1;
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x,y1)
    rmse.append(sqrt(mean_squared_error(y1,knn.predict(x))))
    print("K_value",k,"rmse_value",sqrt(mean_squared_error(y1,knn.predict(x))))


# In[123]:


#FROM ABOVE K=9 IS THE THRESHOLD VALUE SO WE SELECT 9
knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)


# In[124]:


#FINDING ACCURACY OF MODEL
from sklearn import metrics
print("accuracy",metrics.accuracy_score(y,knn.predict(x)))


# In[ ]:




