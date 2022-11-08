#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[36]:


loan = pd.read_csv("loan_data_2007_2014.csv")


# In[37]:


loan.head(10)


# In[38]:


loan.shape


# In[39]:


loan.tail()


# In[40]:


loan.info()


# # Data Cleaning

# In[41]:


loan_clean = loan


# In[42]:


loan_clean.shape


# In[43]:


loan_clean.info()


# In[44]:


loan_clean.head()


# In[45]:


loan_clean = loan_clean.drop(["id", "member_id"], axis = 1)


# In[46]:


loan_clean.shape


# In[47]:


loan_clean.describe()


# In[48]:


loan_clean.loan_status.value_counts(normalize=True)*100


# In[49]:


loan_clean["loan_status"].unique()


# In[50]:


not_paid = ["Charged Off", "Late (31-120 days)", "Late (16-30 days)", "Default", "Does not meet the credit policy. Status:Charged Off"]


# In[51]:


loan_clean["fail_status"] = np.where(loan_clean["loan_status"].isin(not_paid), 1,0)
loan_clean["fail_status"] = loan_clean["fail_status"].astype('float')


# In[52]:


loan_clean["emp_length"].unique()


# In[53]:


loan_clean["new_emp_length"] = loan_clean["emp_length"].str.replace("\+ years", " ")
loan_clean["new_emp_length"] = loan_clean["new_emp_length"].str.replace("< 1 year", str(0))
loan_clean["new_emp_length"] = loan_clean["new_emp_length"].str.replace(" years", " ")
loan_clean["new_emp_length"] = loan_clean["new_emp_length"].str.replace(" year", " ")


# In[54]:


loan_clean["new_emp_length"].unique()


# In[55]:


loan_clean["new_emp_length"] = loan_clean["new_emp_length"].astype(float)


# In[56]:


loan_clean.drop('emp_length', axis = 1, inplace = True )


# In[57]:


loan_clean["new_term"] = loan_clean['term'].str.replace(' months', ' ')


# In[58]:


loan_clean['new_term'].unique()


# In[59]:


loan_clean["new_term"] = loan_clean["new_term"].astype(float)


# In[60]:


loan_clean.drop('term', axis = 1, inplace = True)


# In[61]:


loan_clean.head()


# In[62]:


loan_clean.info()


# # Exploratory Data Analysis

# In[85]:


loan_corr = loan_clean_new.corr()
plt.figure(figsize = (20,12))
sns.heatmap(loan_corr, cmap = 'Pastel1')


# In[64]:


loan_clean.info()


# In[67]:


loan_clean_new = loan_clean.drop(['grade', 'sub_grade', 'emp_title', 'home_ownership', 'verification_status', 'issue_d', 'loan_status', 'pymnt_plan', 'url', 'desc', 'purpose', 'title', 'zip_code', 'addr_state', 'earliest_cr_line', 'initial_list_status','last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 'application_type', "annual_inc_joint","dti_joint","verification_status_joint", "dti_joint", "verification_status_joint", "open_acc_6m", "open_il_6m", "open_il_12m","open_il_24m", "mths_since_rcnt_il","total_bal_il", "il_util", "open_rv_12m","open_rv_24m", "max_bal_bc", "all_util","inq_fi", "total_cu_tl", "inq_last_12m"],axis=1)


# In[68]:


loan_clean_new.info()


# In[69]:


loan_clean_new.dropna(inplace = True)


# In[71]:


loan_clean_new.shape


# # Modeling

# In[72]:


from sklearn.model_selection import train_test_split
X = loan_clean_new.drop('fail_status', axis = 1)
y = loan_clean_new['fail_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 53)


# In[73]:


X_train.shape


# In[74]:


X_test.shape


# In[75]:


from sklearn.linear_model import LogisticRegression


# In[79]:


model_lr = LogisticRegression()


# In[80]:


model_lr.fit(X_train, y_train)


# In[81]:


y_pred = model_lr.predict(X_test)


# In[83]:


from sklearn.metrics import classification_report


# In[84]:


print(classification_report(y_test,y_pred))

