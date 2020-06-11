
# coding: utf-8

# # import libraries

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp


# # import dataset

# In[8]:


data_set = pd.read_csv("Data.csv")


# In[9]:


data_set


# # divideded into depedent and independent

# In[16]:


x = data_set.iloc[:,:-1].values


# In[17]:


x


# In[18]:


y = data_set.iloc[:,3].values


# In[19]:


y


# # handeling missing data

# In[20]:


from sklearn.preprocessing import Imputer


# In[22]:


imputer = Imputer(missing_values="NaN", strategy='mean', axis=0) 


# In[26]:


imputerimputer = imputer.fit(x[:,1:3])


# In[27]:


x[:,1:3] = imputer.transform(x[:,1:3])


# In[28]:


x


# # label encoding

# In[29]:


from sklearn.preprocessing import LabelEncoder


# In[30]:


label_encoder_x = LabelEncoder()


# In[31]:


label_encoder_x


# In[32]:


x[:,0] = label_encoder_x.fit_transform(x[:,0])


# In[33]:


x


# # encoding for dummy values

# In[34]:


from sklearn.preprocessing import OneHotEncoder


# In[35]:


onehot_encoder = OneHotEncoder(categorical_features=[0])


# In[36]:


x = onehot_encoder.fit_transform(x).toarray()


# In[37]:


x


# # label encoder for Y

# In[39]:


labelencoder_y = LabelEncoder()


# In[41]:


y = labelencoder_y.fit_transform(y)


# In[42]:


y


# # spliting in train & test

# In[46]:


from sklearn.model_selection import train_test_split


# In[63]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[64]:


x_train


# In[65]:


x_test


# In[66]:


y_train


# In[67]:


y_test


# # feature scaling

# In[68]:


from sklearn.preprocessing import StandardScaler


# In[69]:


st_x = StandardScaler()


# In[71]:


x_train = st_x.fit_transform(x_train)


# In[72]:


x_train


# # we will directly apply transform() function instead of fit_transform() because it is already done in training set

# In[75]:


x_test = st_x.transform(x_test) 


# In[76]:


x_test

