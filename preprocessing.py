'''execute dataset preprocessing code'''
import scipy
import time
import os
import numpy as np
import numpy.lib.scimath as sc

####################
def fillDict(dicFall, f, fall3, FT):
  """
  fills a dictionary (dicFall) with keeps the different types of falls in separated entries.
  Initially all the entries of dicFall are empty lists
  f is an open text file with the key of each fall
  So the line n in file pointed by f is a key word corresponding to the type of fall.
  It should correspond to vector components in rows 3*n, 3*n+1 and 3*n+2 in fall3
  Fall3 is a vector 3N x 301 (301 = 6 s at 50 Hz), N number of falls, in which
  the rows are successively x,y,z,x,y,z etc components
  FT is a list of keywords
  f is closed inside the function
  Fall3 is in vectorial form, but we extract the total acceleration
  """
  #FT=['forw_free','forw_prot','latl', 'latr', 'back', 'obst','sync','empt']
  # Get total acceleration. Checked that this is OK even if we make no mention to the second dimension
  fallat=sc.sqrt(fall3[0::3]**2+fall3[1::3]**2+fall3[2::3]**2)
  n=0
  s=' '
  while(s!=''):
    s=f.readline()
    for key in FT:
      if(key in s):
        dicFall[key].append(fallat[n,:])
        break
    n +=1
  f.close()
  return
#################
def getFallDataAsDictWKeys(position, FT=['forw_free','forw_prot','latl', 'latr', 'back', 'obst','sync','empt']):
  """
  Gets the fall data in a dictionary for all people. Each entry in the dictionary is a type of
  fall, and is given as an array, where we store the total acceleration (each row is a different fall)
  Position can only be 'pocket' for the moment
  FT is the list of keys. Default value separates all kind of falls we used
  It is supposed that the raw acceleration data files have names that include one of the keywords
  in FT

  Output: dicFall, the dictionary
  """
  dicFall={}
  if(position=='pocket'):
    for key in FT:
      dicFall[key]=[]
    fall0=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person0/fallProcessedVector/0fallPV.dat')
    f=open('temp_extract/publicFallDetector201307/data201307/person0/fallProcessedVector/0fallPVKeys.dat','r')
    fillDict(dicFall, f, fall0,FT)
    ##
    fall1=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person1/fallProcessedVector/1fallPV.dat')
    f=open('temp_extract/publicFallDetector201307/data201307/person1/fallProcessedVector/1fallPVKeys.dat','r')
    fillDict(dicFall, f, fall1,FT)
    ##
    fall2=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person2/fallProcessedVector/2fallPV.dat')
    f=open('temp_extract/publicFallDetector201307/data201307/person2/fallProcessedVector/2fallPVKeys.dat','r')
    fillDict(dicFall, f, fall2, FT)
    ##
    fall3=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person3/fallProcessedVector/3fallPV.dat')
    f=open('temp_extract/publicFallDetector201307/data201307/person3/fallProcessedVector/3fallPVKeys.dat','r')
    fillDict(dicFall, f, fall3, FT)
    ##
    fall4=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person4/fallProcessedVector/4fallPV.dat')
    f=open('temp_extract/publicFallDetector201307/data201307/person4/fallProcessedVector/4fallPVKeys.dat','r')
    fillDict(dicFall, f, fall4, FT)
    ###
    fall5=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person5/fallProcessedVector/5fallPV.dat')
    f=open('temp_extract/publicFallDetector201307/data201307/person5/fallProcessedVector/5fallPVKeys.dat','r')
    fillDict(dicFall, f, fall5, FT)
    ###
    fall6=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person6/fallProcessedVector/6fallPV.dat')
    f=open('temp_extract/publicFallDetector201307/data201307/person6/fallProcessedVector/6fallPVKeys.dat','r')
    fillDict(dicFall, f, fall6, FT)
    ###
    fall7=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person7/fallProcessedVector/7fallPV.dat')
    f=open('temp_extract/publicFallDetector201307/data201307/person7/fallProcessedVector/7fallPVKeys.dat','r')
    fillDict(dicFall, f, fall7, FT)
    ###
    fall8=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person8/fallProcessedVector/8fallPV.dat')
    f=open('temp_extract/publicFallDetector201307/data201307/person8/fallProcessedVector/8fallPVKeys.dat','r')
    fillDict(dicFall, f, fall8, FT)
    ###
    fall9=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person9/fallProcessedVector/9fallPV.dat')
    f=open('temp_extract/publicFallDetector201307/data201307/person9/fallProcessedVector/9fallPVKeys.dat','r')
    fillDict(dicFall, f, fall9, FT)
    ### Transform to arrays
    for key in FT:
      dicFall[key]=np.array(dicFall[key])
    return dicFall
########################
def getAllDataAsListNew(kind, position):
  """
  Obtains data of all people together as a list (each member for a given person)
  Each entry is an array. We use the data in vectorial form to get only the total acceleration
  kind='fall' or 'adl'
  position='pocket' or 'hbag'
  Some combinations are not implemented yet
  Returns the list of data. Each element of the list is an array, in which each row is a temporal sequence
  of acceleration values
  """
  if(kind=='fall' and position=='pocket'):
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person0/fallProcessedVector/0fallPV.dat')
    fall0=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person1/fallProcessedVector/1fallPV.dat')
    fall1=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person2/fallProcessedVector/2fallPV.dat')
    fall2=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person3/fallProcessedVector/3fallPV.dat')
    fall3=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person4/fallProcessedVector/4fallPV.dat')
    fall4=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person5/fallProcessedVector/5fallPV.dat')
    fall5=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person6/fallProcessedVector/6fallPV.dat')
    fall6=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person7/fallProcessedVector/7fallPV.dat')
    fall7=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person8/fallProcessedVector/8fallPV.dat')
    fall8=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person9/fallProcessedVector/9fallPV.dat')
    fall9=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    return (fall0, fall1, fall2, fall3, fall4, fall5, fall6, fall7, fall8, fall9)
    ####################
  elif(kind=='fall' and position=='hbag'):
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person0/fallProcessedVector/0fallPVHbag.dat')
    fall0=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person1/fallProcessedVector/1fallPVHbag.dat')
    fall1=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person2/fallProcessedVector/2fallPVHbag.dat')
    fall2=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person3/fallProcessedVector/3fallPVHbag.dat')
    fall3=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person4/fallProcessedVector/4fallPVHbag.dat')
    fall4=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person5/fallProcessedVector/5fallPVHbag.dat')
    fall5=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person6/fallProcessedVector/6fallPVHbag.dat')
    fall6=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person7/fallProcessedVector/7fallPVHbag.dat')
    fall7=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person8/fallProcessedVector/8fallPVHbag.dat')
    fall8=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    falldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person9/fallProcessedVector/9fallPVHbag.dat')
    fall9=sc.sqrt(falldum[0::3]**2+falldum[1::3]**2+falldum[2::3]**2)
    ###
    return (fall0, fall1, fall2, fall3, fall4, fall5, fall6, fall7, fall8, fall9)
  elif(kind=='adl' and position=='pocket'):
    adldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person0/adlProcessedVector/0adlPV.dat')
    adl0=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ###
    adldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person1/adlProcessedVector/1adlPV.dat')
    adl1=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ###
    adldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person2/adlProcessedVector/2adlPV.dat')
    adl2=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ###
    adldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person3/adlProcessedVector/3adlPV.dat')
    adl3=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ###
    adldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person4/adlProcessedVector/4adlPV.dat')
    adl4=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ####
    adldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person5/adlProcessedVector/5adlPV.dat')
    adl5=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ###
    adldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person6/adlProcessedVector/6adlPV.dat')
    adl6=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ###
    adldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person7/adlProcessedVector/7adlPV.dat')
    adl7=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ###
    adldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person8/adlProcessedVector/8adlPV.dat')
    adl8=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ###
    adldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person9/adlProcessedVector/9adlPV.dat')
    adl9=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ###
    return (adl0, adl1, adl2, adl3, adl4, adl5, adl6, adl7, adl8, adl9)
  elif(kind=='adl' and position=='hbag'):
    adldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person0/adlProcessedVector/0adlPVHbag.dat')
    adl0=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ###
    adldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person1/adlProcessedVector/1adlPVHbag.dat')
    adl1=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ###
    adldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person2/adlProcessedVector/2adlPVHbag.dat')
    adl2=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ###
    adldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person3/adlProcessedVector/3adlPVHbag.dat')
    adl3=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ###
    #adldum=sc.loadtxt('data201307/person4/adlProcessedVector/4adlPVHbag.dat')
    #adl4=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ####
    #adldum=sc.loadtxt('data201307/person5/adlProcessedVector/5adlPVHbag.dat')
    #adl5=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ###
    #adldum=sc.loadtxt('data201307/person6/adlProcessedVector/6adlPVHbag.dat')
    #adl6=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ###
    #adldum=sc.loadtxt('data201307/person7/adlProcessedVector/7adlPVHbag.dat')
    #adl7=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ###
    #adldum=sc.loadtxt('data201307/person8/adlProcessedVector/8adlPVHbag.dat')
    #adl8=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ###
    adldum=np.loadtxt('temp_extract/publicFallDetector201307/data201307/person9/adlProcessedVector/9adlPVHbag.dat')
    adl9=sc.sqrt(adldum[0::3]**2+adldum[1::3]**2+adldum[2::3]**2)
    ###
    #return (adl0, adl1, adl2, adl3, adl4, adl5, adl6, adl7, adl8, adl9)
    return (adl0,adl1,adl2,adl3,adl9)
  else:
    return ()
########################
'''execute combining data and adding labels'''
import numpy as np

# Step 1: Get fall data as dictionary
fall_data_dict = getFallDataAsDictWKeys(position='pocket')

# Step 2: Get ADL data as list
adl_data_list = getAllDataAsListNew(kind='adl', position='pocket')

# Step 3: Combine fall and ADL data
combined_data = []
labels = []

# Append fall data to combined_data
for fall_type, data_array in fall_data_dict.items():
    combined_data.extend(data_array)
    labels.extend([1] * len(data_array))  # Label 1 for falls

# Append ADL data to combined_data
for adl_data in adl_data_list:
    combined_data.extend(adl_data)
    labels.extend([0] * len(adl_data))  # Label 0 for ADLs

# Convert combined_data and labels to NumPy arrays
combined_data = np.array(combined_data)
labels = np.array(labels)

# Shuffle the data and labels in unison
random_indices = np.random.permutation(len(combined_data))
combined_data = combined_data[random_indices]
labels = labels[random_indices]

# Check the shape of the combined data and labels
print("Shape of combined data:", combined_data.shape)
print("Shape of labels:", labels.shape)

'''execute xgboost algorithm'''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Assuming you have already imported the necessary libraries and defined your combined data and labels (combined_data, labels)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(combined_data, labels, test_size=0.3, random_state=42)

'''execute deepsense'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM

# Assuming combined_data and labels are your feature matrix and target labels, respectively

# Define the input shape based on the number of features (assuming 153 features)
input_shape = (combined_data.shape[1], 1)  # (Time steps, Features)

# Reshape the combined_data to fit the input shape
X = combined_data.reshape((combined_data.shape[0], combined_data.shape[1], 1))

# Define the DeepSense model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(64, return_sequences=True),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
#model.fit(X, labels, epochs=3, batch_size=32, validation_split=0.1)

#loss, accuracy = model.evaluate(X, labels)
#print(f'Test Loss: {loss}')
#print(f'Test Accuracy: {accuracy}')

'''execute deepsense+xgboost'''
# Train the DeepSense model
history = model.fit(X, labels, epochs=8, batch_size=32, validation_split=0.1)

# Generate predictions using the DeepSense model
deep_sense_predictions_train = model.predict(X_train)
deep_sense_predictions_test = model.predict(X_test)

# Train the XGBoost algorithm on the output of the DeepSense model
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

xgb_model = XGBClassifier()
xgb_model.fit(deep_sense_predictions_train, y_train)

# Evaluate the XGBoost model
xgb_predictions = xgb_model.predict(deep_sense_predictions_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
print("XGBoost Accuracy:", xgb_accuracy)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Calculate confusion matrix for XGBoost predictions
conf_matrix = confusion_matrix(y_test, xgb_predictions)

# Extract true negatives (TN), false positives (FP), false negatives (FN), and true positives (TP)
TN, FP, FN, TP = conf_matrix.ravel()

# Calculate sensitivity (recall), specificity, precision, and F1-score
sensitivity = recall_score(y_test, xgb_predictions)
specificity = TN / (TN + FP)
precision = precision_score(y_test, xgb_predictions)
f1 = f1_score(y_test, xgb_predictions)

print("Sensitivity (Recall):", sensitivity)
print("Specificity:", specificity)
print("Precision:", precision)
print("F1-Score:", f1)
