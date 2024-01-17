# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# Import the modules
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
import seaborn as sns
import sklearn
import imblearn

# Let's ignore some warnings
import warnings
warnings.filterwarnings ('ignore')

# Tune the settings
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=sys.maxsize)  # Display all array elements
np.set_printoptions(precision=3)  # Set precision to 3 decimal places
sns.set(style="darkgrid")
plt.rcParams['axes.labelsize'] = 14  # Correct parameter for axis label size
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

#STEP1: Import load the training and testing dataset
train_data = pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/Python_IDS_2/Train_data.csv")
test_data = pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/Python_IDS_2/Test_data.csv")

from google.colab import files

# Upload the train dataset file
# train_data_upload1 = files.upload()

# Let's upload the test dataset
# test_data_upload2 = files.upload()

import pandas as pd

# file_name1 =  'UNSW_NB15_training-set (1).csv' # Let this name be the same with the file uploaded in 'data_uploaded1' above
# file_name2 =  'UNSW_NB15_testing-set.csv' # Let this name be the same with the file uploaded in 'data_uploaded2' above

# train_data = pd.read_csv(file_name1)
# test_data = pd.read_csv(file_name2)

#STEP2: Let's perform Exploratory Data Analysis (EDA) on the train dataset

# Let's first get the descriptive statistics
train_data.describe()

# Let check the first few headers

train_data.head()

# Use decribe() to check the properties of the test dataset
test_data.describe()

print(train_data.columns)

# Let's check for numm values in train_data
train_data.isnull().sum()

test_data.head()

test_data.isnull().sum()

print(test_data.columns)

# STEP 3: Let's Scale the Numerical Attributes
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit the scaler only on the training data.
scaler.fit(train_data.select_dtypes(include=['float64', 'int64']))

# Transform both training and test data, using the the fitted scaler from earlier.
sc_train_data = scaler.transform(train_data.select_dtypes(include=['float64', 'int64']))
sc_test_data = scaler.transform(test_data.select_dtypes(include=['float64', 'int64']))

# But we must extract the column names from the original numerical features before
# we can create DataFrames from the scaled data.
cols = train_data.select_dtypes(include=['float64', 'int64']).columns

# Now, we can create DataFrames from the scaled data
sc_train_data_df = pd.DataFrame(sc_train_data, columns=cols)
sc_test_data_df = pd.DataFrame(sc_test_data, columns=cols)

# STEP 4: Let's Encode the Categorical Attributes
from sklearn.preprocessing import OneHotEncoder

# Extract categorical columns (excluding 'class')
cattrain = train_data.select_dtypes(include=['category', 'object'])  # Assuming categorical columns
cat_cols = [col for col in cattrain.columns if col != 'class']
cattrain = cattrain[cat_cols]

# Extract categorical features from test data
cattest = test_data.select_dtypes(include=['category', 'object'])  # Assuming categorical columns
cattest = cattest[cat_cols]  # Keep consistent columns

# Create and fit the encoder
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(pd.concat([cattrain, cattest], axis=0))

# Transform both datasets
traincat = encoder.transform(cattrain)
testcat = encoder.transform(cattest)

# Create DataFrames from the encoded data
enctrain = pd.DataFrame(traincat.toarray())
testcat = pd.DataFrame(testcat.toarray())

# Let's combine the numerical and categorical features into a single DataFrame.
train_data_x = pd.concat([sc_train_data_df, enctrain], axis=1)

# Separate the target variable ('class') for training the model.
train_data_y = train_data['class']

# Print the output to confirm the dimensions of the prepared training data.
train_data_x.shape

# Let's prepare the test data by combining numerical and categorical features
sc_test_data_df = pd.concat([sc_test_data_df, testcat], axis=1)

# Print the output to show the dimensions of the prepared test data, confirming its readiness for model evaluation
sc_test_data_df.shape

# Import the RandomForestClassifier for feature selection
from sklearn.ensemble import RandomForestClassifier

# Create a RandomForestClassifier instance
rfc = RandomForestClassifier()

# Ensure column names are strings for compatibility (if needed)
train_data_x.columns = train_data_x.columns.astype(str)

# Fit the model to the training data to learn feature importances
rfc.fit(train_data_x, train_data_y)

# Extract feature importance scores from the trained model
score = np.round(rfc.feature_importances_, 3)  # Round scores to 3 decimal places

# Create a DataFrame to organize feature importances
importances = pd.DataFrame({
    'feature': train_data_x.columns,  # Feature names
    'importance': score  # Corresponding importance scores
})

# Sort features by importance in descending order and set feature names as index
importances = importances.sort_values('importance', ascending=False).set_index('feature')

# Visualize the top 30 most important features using a bar plot
importances.head(30).plot.bar();

# Import the RFE (Recursive Feature Elimination) class for feature selection
from sklearn.feature_selection import RFE

# Import itertools for efficient feature mapping
import itertools

# Create a RandomForestClassifier instance for RFE
rfc = RandomForestClassifier()

# Initialize an RFE model with RandomForestClassifier as estimator
# Specify the number of features to select
rfe = RFE(rfc, n_features_to_select=15)

# Fit the RFE model to the training data to identify important features
rfe = rfe.fit(train_data_x, train_data_y)

# Create a feature map to link feature indices with their names
feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.support_, train_data_x.columns)]

# Extract the names of selected features based on their support (True/False)
# selected_features = [v for i, v in feature_map if i == True]
selected_features = train_data_x.columns[rfe.support_]

# Print the selected features
print(selected_features)

print(rfe.support_)
print(feature_map)

# Let's check the shape of train_data_x and train_data_y before splitting
print(train_data_x.shape, train_data_y.shape)

# Ensure consistent preprocessing for both training and test data


# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Ensure X_train, X_test, Y_train, Y_test are all consistent
X_train, X_test, Y_train, Y_test = train_test_split(train_data_x, train_data_y, train_size=0.70, random_state=2)

# Check shapes after splitting
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# Ensure that X_train, X_test have consistent shapes in terms of the number of columns (features)
# Ensure Y_train, Y_test have consistent shapes in terms of the number of labels

# STEP 7: Let's Start Fitting the Models
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Let's train the KNeighborsClassifier Model
KNN_Classifier = KNeighborsClassifier(n_jobs=-1)
KNN_Classifier.fit(X_train, Y_train);

# Let's train the LogisticRegression Model
LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0, max_iter=1000)  # Adjust max_iter as needed
# LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0)
LGR_Classifier.fit(X_train, Y_train);

# Now, let's train the Gaussian Naive Baye Model
BNB_Classifier = BernoulliNB()
BNB_Classifier.fit(X_train, Y_train)

# LEt's train the Decision Tree Model
DTC_Classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
DTC_Classifier.fit(X_train, Y_train)
print(DTC_Classifier)  # Print the full model object

# Train the Logistic Regression model on the scaled training data
LGR_Classifier.fit(X_train, Y_train)

# STEP 8: Now, let's Evaluate the Models
from sklearn import metrics

models = []
models.append(('Naive Baye Classifier', BNB_Classifier))
models.append(('Decision Tree Classifier', DTC_Classifier))
models.append(('KNeighborsClassifier', KNN_Classifier))
models.append(('LogisticRegression', LGR_Classifier))

for i, v in models:
  scores = cross_val_score(v, X_train, Y_train, cv=10)
  accuracy = metrics.accuracy_score(Y_train, v.predict(X_train))
  confusion_matrix = metrics.confusion_matrix(Y_train, v.predict(X_train))
  classification = metrics.classification_report(Y_train, v.predict(X_train))
  print()
  print('====================={} Model Evaluation ===================.format(i)')
  print()
  print("Cross Validation Mean Score:" "\n", scores.mean())
  print()
  print("Confusion matrix:" "\n", confusion_matrix)
  print()
  print("Classification report:" "\n", classification)
  print()

# STEP 9: Let's Validate the Models :)
for i, v in models:
  accuracy = metrics.accuracy_score(Y_test, v.predict(X_test))
  confusion_matrix = metrics.confusion_matrix(Y_test, v.predict(X_test))
  print()
  print('=============== {} Model Test Results =================')
  print()
  print("Model Accuracy:" "\n", accuracy)
  print()
  print("Confusin matrix:" "\n", confusion_matrix)
  print()
  print("Classifcation report:" "\n", classification)
  print()

# Ensure string column names
sc_test_data_df.columns = sc_test_data_df.columns.astype(str)

# Now, let's test the models with the Test Dataset
pred_knn = KNN_Classifier.predict(sc_test_data_df)
pred_NB = BNB_Classifier.predict(sc_test_data_df)
pred_log = LGR_Classifier.predict(sc_test_data_df)
pred_dt = DTC_Classifier.predict(sc_test_data_df)

# #Let's print the result of the models
print(pred_knn)
print(pred_NB)
print(pred_log)
print(pred_dt)

import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, classes, title='Confusion Matrix for BNB_Classifier', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Replace model with your chosen model
cm = metrics.confusion_matrix(Y_test, BNB_Classifier.predict(X_test))
plot_confusion_matrix(cm, classes=['normal', 'anomaly'])

import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, classes, title='Confusion Matrix for DTC_Classifier', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Example usage:
cm = metrics.confusion_matrix(Y_test, DTC_Classifier.predict(X_test))  # Replace model with your chosen model
plot_confusion_matrix(cm, classes=['normal', 'anomaly'])

import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, classes, title='Confusion Matrix for KNN_Classifier', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Example usage:
cm = metrics.confusion_matrix(Y_test, KNN_Classifier.predict(X_test))  # Replace model with your chosen model
plot_confusion_matrix(cm, classes=['normal', 'anomaly'])

import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, classes, title='Confusion Matrix for LGR_Classifier', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Replace model with your chosen model
cm = metrics.confusion_matrix(Y_test, LGR_Classifier.predict(X_test))
plot_confusion_matrix(cm, classes=['normal', 'anomaly'])

# Let's save the trained models
import pickle

# LEt's specify the desired path
model_path = "/content/drive/MyDrive/Colab_Notebooks/Python_IDS_2"

# Let's first check to see if the path exits or we create one if necessary
import os
os.makedirs(model_path, exist_ok=True)

# Now, let's save the models using the full path
with open(os.path.join(model_path, 'BNB_Classifier.pkl'), 'wb') as f:
    pickle.dump(BNB_Classifier, f)

with open(os.path.join(model_path, 'KNN_Classifier.pkl'), 'wb') as f:
    pickle.dump(KNN_Classifier, f)

with open(os.path.join(model_path, 'LGR_Classifier.pkl'), 'wb') as f:
    pickle.dump(LGR_Classifier, f)

with open(os.path.join(model_path, 'DTC_Classifier.pkl'), 'wb') as f:
    pickle.dump(DTC_Classifier, f)

# Let's check the saved model files
for filename in ['BNB_Classifier.pkl', 'KNN_Classifier.pkl', 'LGR_Classifier.pkl', 'DTC_Classifier.pkl']:
  full_path = os.path.join(model_path, filename)
  if os.path.exists(full_path):
    print(f"{filename} exists in {model_path}")
  else:
    print(f"{filename} not found in {model_path}")