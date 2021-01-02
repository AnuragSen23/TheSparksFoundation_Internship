#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation

# ## Data Science & Business Analytics Internship

# ## Task 1: Prediction using Supervised ML

# ### This task focuses on predicting the score obtained by a student based on the number of study hour. For the prediction, Linear Regression model under supervised ML is employed.

# #### Based on Data Source: http://bit.ly/w-data

# ### demonstrated and performed by : Anurag Sen | sen1anurag@gmail.com

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[12]:


data = pd.read_csv("C:\\Users\Anurag Sen\Desktop\StudentHourStudydata.csv")
data


# In[13]:


plt.title("Raw data - Hours studied vs Marks scored")
plt.xlabel('Number of study hours')
plt.ylabel('Score achieved')
plt.scatter(data.Hours,data.Scores,color='green',label = 'Data Distribution')
plt.legend(['Data Distribution'])


# ### Training the dataset

# In[14]:


LR = linear_model.LinearRegression()
LR.fit(data[['Hours']],data.Scores)


# In[15]:


m = LR.coef_
m


# In[16]:


b = LR.intercept_
b


# In[17]:


#Now using the equation of Linear Regression Line, which is Y = M*X + B
Predicted_Score = data[['Hours']]*m + b


# In[18]:


plt.title("Prediction of Hours studied vs Marks scored using Linear Regression")
plt.xlabel('Number of study hours')
plt.ylabel('Score achieved')
plt.scatter(data.Hours,data.Scores,color='green', label = 'Data Distribution')
plt.plot(data.Hours,Predicted_Score,color='red', label = 'Linear Regression Line')
plt.legend(['Linear Regression Line','Data Distribution'])


# ### Testing the dataset on the given problem statement

# In[19]:


def LR_Prediction(hour):
    score = m*hour + b
    return score


# ###### Problem statement: What will be the predicted score, if the student studies for 9.25 hours/day?

# In[20]:


Prediction = LR_Prediction(9.25)
print("Given, that the student studies 9.25 hour/day, the prediction of Score achieved is %.4f"%Prediction)


# In[21]:


from sklearn.metrics import mean_absolute_error
print("Mean of Absolute Error:", mean_absolute_error(data.Scores, Predicted_Score))

