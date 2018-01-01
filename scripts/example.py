
# coding: utf-8

# # Example

# In[3]:

import turicreate as tc


# ## Get the data

# In[22]:

data = 'path-to-data-here'
sf = tc.SFrame(data).dropna(columns=['Age'])
train, test = sf.random_split(fraction=0.8)
test, validations = test.random_split(fraction=0.5)


# ## Modeling 

# In[27]:

from turicreate import logistic_classifier
model = logistic_classifier.create(train, target='Survived',validation_set=validations)


# ## Evaluate

# Use turi 

# In[ ]:



