
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from csvApp.models import dataFields

from plotly.offline import plot
import plotly.graph_objs as go

def naive_bayes():

    # Importing the dataset

    temp_csv_file=os.getcwd() + "\\temp_files\\tempcsv.csv"
    dataset = pd.read_csv(temp_csv_file)

    var = dataFields.y_field
    print(var)

    # define X Y 
    # Set the dependent variable array
    y=dataset[var].values

    xlist=[]

    for col in dataFields.x_fields:
        if not isinstance(dataset[col].values.tolist()[0],str) and col != var:
            xlist.append(col)

    X = dataset[xlist].values    

    # add missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

    imputer = imputer.fit(X)
    X = imputer.transform(X)

    # Split the train and test data

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # If Required apply feature scaling
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # Define the function
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train,y_train)

    # Apply regression model 
    y_pred = classifier.predict(X_test)

    # Calculate the Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    return_dic = {'cm': cm}    

    return return_dic


## Function to print the Graph ##
def Naive_Bayes_graph(request):

    pred_values = naive_bayes()

    print(" Inside naive_bayes_graph")

    cm = pred_values['cm']

    cm1 = cm[0,0]
    cm2 = cm[0,1]
    cm3 = cm[1,0]
    cm4 = cm[1,1]
    

    context = {
        'modelType' : dataFields.modelType,
        'modelName' : dataFields.modelName,
        'cm1'       : cm1,
        'cm2'       : cm2,
        'cm3'       : cm3,
        'cm4'       : cm4,

    }
    return context
