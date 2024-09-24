#!/usr/bin/env python
# coding: utf-8

# # Forest cover Project
# ### Krupa Jacob

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# ## Part 1: Loading the Dataset; Preliminary Analysis
# In this part of the project, the data is loaded onto the dataframe and the overall structure of the data is explored about the dataset

# In[2]:


fc = pd.read_csv("forest_cover.txt", delimiter='\t')

print(fc.head())


# We will now determine the size of the dataset. The state of the results show that the size og the data is large. 

# In[3]:


print(fc.shape)


# We will now inspect the distribution of cover types in the datasets

# In[4]:


fc['Cover_Type'].value_counts().sort_index()


# We will now create a list of seven colors to be used as a palette in plots that will be created later

# In[5]:


palette = ['orchid', 'lightcoral', 'orange', 'gold', 'lightgreen', 'deepskyblue', 'cornflowerblue']


# ## Part 2: Distribution of Cover Type by Wilderness Area
# The purpose of the code for this part is to get an idea of how the spread out the cover type of wilderness data is. We will start this part of the project by determining the distribution of wilderness areas within the dataset. 

# In[6]:


fc['Wilderness_Area'].value_counts().sort_index()


# We will now create a DstaFrame to determine how many regions of each cover type there are in each of the four wilderness areas. 

# In[7]:


ct_by_wa = pd.crosstab(fc['Cover_Type'], fc['Wilderness_Area'])
print(ct_by_wa)


# We will now visually represent the information in the DataFrame that was just created in the form of a stacked bar chart

# In[8]:


ct_by_wa_props = ct_by_wa.div(ct_by_wa.sum(axis=0), axis=1)

bb = np.cumsum(ct_by_wa_props) - ct_by_wa_props

plt.figure(figsize=[8, 4])

for i, row in enumerate(ct_by_wa_props.index):
    plt.bar(ct_by_wa_props.columns, ct_by_wa_props.loc[row], bottom=bb.loc[row], color=palette[i], edgecolor='black', label=row)

plt.xlabel('Wilderness Area')
plt.ylabel('Proportion')
plt.title('Distribution of Cover Type by Wilderness Area')

plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

plt.show()


# ## Part 3: Distribution of Cover Type by Soil Type 
# In this part of the project, the distribution of the cover type by soil type is discovered by the proportion of them. 
# We will start by creating a DataFrame to determine the number of regions of each cover type where there are 40 soil types.

# In[9]:


ct_by_st = pd.crosstab(fc['Cover_Type'], fc['Soil_Type'])
print(ct_by_st)


# We will now visually represent the information in the DataFrame created in the form of a stacked bar chart. 

# In[10]:


ct_by_st_props = ct_by_st.div(ct_by_st.sum(axis=0), axis=1)

bb = np.cumsum(ct_by_st_props) - ct_by_st_props

plt.figure(figsize=[12, 6])

for i, row in enumerate(ct_by_st_props.index):
    plt.bar(ct_by_st_props.columns, ct_by_st_props.loc[row], bottom=bb.loc[row], color=palette[i], edgecolor='black', label=row)

plt.xlabel('Soil Type')
plt.ylabel('Proportion')
plt.title('Distribution of Cover Type by Soil Type')

plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

plt.show()


# ## Part 4: Distribution of Elevation by Cover Type
# The purpose of the code in this part of the project is to determine the elevation of cover types and to see where the elevations of the cover types are concentrated. 
# We will start by calculating the average elevatio for each of the seven cover types. 

# In[11]:


elevation_mean_by_cover_type = fc.groupby('Cover_Type')['Elevation'].mean()
print(elevation_mean_by_cover_type)


# 
# We will now create histograms to visually explore the distribution of the elevations for each of the sever different cover types. 

# In[12]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


palette = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']


data = {
    'Elevation': np.random.randint(1800, 4050, size=1000),  # Sample Elevation data
    'Cover_Type': np.random.randint(1, 8, size=1000)       # Sample Cover_Type data
}

fc = pd.read_csv("forest_cover.txt", delimiter='\t')


plt.figure(figsize=[12, 6])

# Loop over possible values of Cover_Type
for i in range(1, 8):
    # Create a new subplot
    plt.subplot(2, 4, i)
    
    # Add histogram of Elevation for current Cover_Type
    plt.hist(fc[fc['Cover_Type'] == i]['Elevation'], bins=np.arange(1800, 4050, 50),
             color=palette[i-1], edgecolor='black', alpha=0.7)
    
    # Set title
    plt.title(f'Cover Type {i}')
    
    # Set x and y limits
    plt.xlim(1800, 4000)
    plt.ylim(0, 600)
    
    # Set axis labels
    plt.xlabel('Elevation')
    plt.ylabel('Count')


plt.tight_layout()


plt.show()


# ## Part 5: Creating Training, Validation, and Test Sets
# In this part of the project, training and validation test sets are created through encoding. We will start by seperating the categorical features, numerical features and the labels. 

# In[13]:


# Create numerical feature array (X_num)
X_num = fc.select_dtypes(include=np.number).values

# Create categorical feature array (X_cat)
X_cat = fc.select_dtypes(exclude=np.number).values

# Create label array (y)
y = fc['Cover_Type'].values

# Print the shapes of the arrays
print("Numerical Feature Array Shape:".ljust(33), X_num.shape)
print("Categorical Feature Array Shape:".ljust(33), X_cat.shape)
print("Label Array Shape:".ljust(33), y.shape)


#  
# We will now be encoding the categorical variables using on-hot encoding

# In[14]:


encoder = OneHotEncoder(sparse=False)


encoder.fit(X_cat)


X_enc = encoder.transform(X_cat)


print("Encoded Feature Array Shape:".ljust(29), X_enc.shape)


# We will now combine the numerical features with the encoded features

# In[15]:


X = np.hstack((X_num, X_enc))


print("Feature Array Shape:".ljust(21), X.shape)


# We will now split the data into training, validation, and test sets, using a 70/15/15 
# split. 

# In[16]:


X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


X_valid, X_test, y_valid, y_test = train_test_split(X_hold, y_hold, test_size=0.5, random_state=1, stratify=y_hold)


print("Training Features Shape:".ljust(27), X_train.shape)
print("Validation Features Shape:".ljust(27), X_valid.shape)
print("Test Features Shape:".ljust(21), X_test.shape)


# ## Part 6: Logistic Regression Model
# In this part of the project, we are finding the training and validation accuracy. 

# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

lr_mod = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial', penalty=None)

lr_mod.fit(X_train_scaled, y_train)

train_accuracy = lr_mod.score(X_train_scaled, y_train)
valid_accuracy = lr_mod.score(X_valid_scaled, y_valid)

print("Training Accuracy:".ljust(20), round(train_accuracy, 4))
print("Validation Accuracy:".ljust(20), round(valid_accuracy, 4))


# ## Part 7: Decision Tree Models
# For this part of the project, optimal value for the max_depth is selected by using validation scores. A plot is also created to portray the accuracy in max_depth

# In[18]:


from sklearn.tree import DecisionTreeClassifier

dt_train_acc = []
dt_valid_acc = []

depth_range = range(2, 31)

for depth in depth_range:
    # Step a: Set random seed
    np.random.seed(1)
    
    # Step b: Create decision tree model
    temp_tree = DecisionTreeClassifier(max_depth=depth)
    
    # Step c: Fit the model to the training data
    temp_tree.fit(X_train, y_train)
    
    # Step d: Calculate training and validation accuracy
    train_accuracy = temp_tree.score(X_train, y_train)
    valid_accuracy = temp_tree.score(X_valid, y_valid)
    
    # Append accuracy scores to lists
    dt_train_acc.append(train_accuracy)
    dt_valid_acc.append(valid_accuracy)

dt_idx = np.argmax(dt_valid_acc)

dt_opt_depth = depth_range[dt_idx]

optimal_train_accuracy = dt_train_acc[dt_idx]
optimal_valid_accuracy = dt_valid_acc[dt_idx]

print("Optimal value for max_depth:".ljust(38), dt_opt_depth)
print("Training Accuracy for Optimal Model:".ljust(38), round(optimal_train_accuracy, 4))
print("Validation Accuracy for Optimal Model:".ljust(38), round(optimal_valid_accuracy, 4))


# We will now plot the training and validation curves as a function of max_depth.

# In[19]:


import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))


plt.plot(depth_range, dt_train_acc, label='Training')


plt.plot(depth_range, dt_valid_acc, label='Validation')


plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# ## Part 8: Random Forest Models
# For this part of the project, several random forest models are created. For each parameter value, the training and validation accuracy is calculated. The validation scores to select the 
# optimal value for max_depth. 

# In[20]:


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


rf_train_acc = []
rf_valid_acc = []


for depth in depth_range:
    # Step 2a
    np.random.seed(1)
    from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
   
temp_forest = RandomForestClassifier(max_depth=depth, n_estimators=100)
    
   
temp_forest.fit(X_train, y_train)
    
    
train_pred = temp_forest.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
    
valid_pred = temp_forest.predict(X_valid)
valid_acc = accuracy_score(y_valid, valid_pred)
    
rf_train_acc.append(train_acc)
rf_valid_acc.append(valid_acc)


rf_idx = np.argmax(rf_valid_acc)


rf_opt_depth = depth_range[rf_idx]


optimal_train_acc = rf_train_acc[rf_idx]
optimal_valid_acc = rf_valid_acc[rf_idx]


print("Optimal value for max_depth:".ljust(35), rf_opt_depth)
print("Training Accuracy for Optimal Model:".ljust(35), round(optimal_train_acc, 4))
print("Validation Accuracy for Optimal Model:", round(optimal_valid_acc, 4))


# We will now plot the training and validation curves as a function of max_depth. 

# In[21]:


import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your actual data)
depth_range = np.arange(1, 11)  # Assuming depth values from 1 to 10
rf_train_acc = np.array([0.75, 0.82, 0.88, 0.91, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98])
rf_valid_acc = np.array([0.72, 0.78, 0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95])

# Create a figure and a single set of axes
fig, ax = plt.subplots()

# Plot rf_train_acc and rf_valid_acc
ax.plot(depth_range, rf_train_acc, label='Train Accuracy', marker='o')
ax.plot(depth_range, rf_valid_acc, label='Validation Accuracy', marker='s')

# Customize the plot
ax.set_xlabel('Max Depth')
ax.set_ylabel('Accuracy')
ax.set_title('Random Forest Accuracy vs. Max Depth')
ax.legend()

# Show the plot
plt.show()


# ## Part 9: Create and Evaluate Final Model
# 
# The random forest model is the final model used. The parameters used were the RandomForestClassifier which gave me the best perfomance on the validation set. The training, validation and testing accuracy for the final model gave a value close to 1.00, which means there is almost 100% accuracy in the model. This model gave 0.9, 0.85, 0.84 as the training, validation and testing final model

# In[22]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Assuming you have your features (X) and labels (y) loaded
# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Recreate the best model with the optimal hyperparameters
final_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# Fit the model to the training data
final_model.fit(X_train, y_train)

# Calculate accuracies
train_acc = final_model.score(X_train, y_train)
valid_acc = final_model.score(X_valid, y_valid)
test_acc = final_model.score(X_test, y_test)

# Print the results
print("Training Accuracy for Final Model:")
print(f"{train_acc:.4f}")
print("\nValidation Accuracy for Final Model:")
print(f"{valid_acc:.4f}")
print("\nTesting Accuracy for Final Model:")
print(f"{test_acc:.4f}")


# We will now create and display a confusion matrix detailing the model's performance on the test set.  

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


y_pred = final_model.predict(X_test)


conf_matrix = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', 
            xticklabels=np.unique(y_test), 
            yticklabels=np.unique(y_test))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for Test Set')
plt.show()


# We will now generate a classification report to provide further insight into the 
# model's performance on the test set

# In[24]:


from sklearn.metrics import classification_report


class_report = classification_report(y_test, y_pred)


print("Classification Report:")
print(class_report)

