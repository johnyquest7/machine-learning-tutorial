# Machine learning tutorial

To run this ipython notebook in an executable / interactive environment without installing any software clik the link below. 

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/johnyquest7/machine-learning-tutorial)

This works best on chrome browser. If asked for a kernel, pick Python 3

The service is provided by an open source project mybinder.org. Since it is self funded service outages are possible. 

If you would like to learn how to impliment machine learning algorithms using Python, head over to the [wiki page](https://github.com/johnyquest7/machine-learning-tutorial/wiki)
   
# Code walk-through

The code is located in the file "Breast_cancer_predict_logist_python3.ipynb"

If you have installed python using Anaconda, you will have all the required packages to run this python file.

import pandas as pd

Panda is a data analysis library in Python. We need this to read our database file.

data = pd.read_csv('wisconsin_breast_cancer.csv')

This line reads our database file "wisconsin_breast_cancer.csv" and stores the contents into a Pandas data frame variable "data".

data.head()

Displays the first 5 rows of the csv file. It is always a good practice to inspect the contents of the file after importing it.

data.shape

Displays the total number of rows and columns.

data.isnull().sum()

We want to make sure that there are no missing values. 'isnull()' will return the missing values. Running this code will show that the column 'nuclei' has 16 missing values.

data=data.dropna(how='any')

Drops any rows that has missing values.

Next we have to divide the data set into x and y. x will contain features that we will use to predict cancer. x will contain the all columns of the dataframe variable 'data' except 'id' and 'class'.
y will contain the data from the column 'class'. 'class' has two values 1= cancer and 0=no cancer. This whole operation is done by the following two lines of code.

x=data[['thickness','size','shape','adhesion','single','nuclei','chromatin','nucleoli','mitosis']]   
y=data['class']

Our goal is to create a logistic regression model to predict breast cancer. Once we create the model, we need to test it to assess the accuracy of our model. To achieve this, we split our data into training and testing data set.A logistic model will be created based on the data in the training set. Then we use this model to predict cancer in the testing set. Since we already know the occurrence of cancer in the test data, the predicted values can be compared with the known values to assess the performance of the model.
We need Sklearn Python library to split the data. This library also has the logistic regression function in it.

from sklearn.cross_validation import train_test_split   
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

First we import the function train_test_split from the cross_validation section of Sklearn library. The second line splits our x (features) and y(cancer or not) data into x_train, x_test, y_train and y_test.

from sklearn.linear_model import LogisticRegression logreg = LogisticRegression() logreg.fit(x_train,y_train)

Next we import logistic regression model from Sklearn. Then we assign LogisticRegression() function to the model variable logreg. After this we train the model logreg using the fit function. Our model is trained and we can use it to predict cancer.

y_pred_class=logreg.predict(x_test)   

We use our model 'logreg''s 'predict' function to make predictions based on x_test and store it to the y_pred_class.

from sklearn import metrics print (metrics.accuracy_score(y_test, y_pred_class))
metrics.accuracy_score function is used to predict the accuracy of or model. This function takes two parameters y_test (real cancer occurrence) and y_pred_class (predict cancer occurrence) to calculate the accuracy.
