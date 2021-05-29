# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from csvApp.models import dataFields

from plotly.offline import plot
import plotly.graph_objs as go

def simpleLinear():

    # Importing the dataset

    temp_csv_file=os.getcwd() + "\\temp_files\\tempcsv.csv"
    dataset = pd.read_csv(temp_csv_file)

    var = dataFields.y_field
    print(var)

    # Set the dependent variable array
    y=dataset[var].values

    # Set the independent variables array
    xlist=[]

    for col in dataset.columns:
        if not isinstance(dataset[col].values.tolist()[0],str) and col != var:
            xlist.append(col)

    X = dataset[xlist].values    

    print("############## #############")
    print("############## #############")

    # Splitting the dataset into the Training set and Test set
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

    # Feature Scaling
    # from sklearn.preprocessing import StandardScaler
    # sc_X = StandardScaler()
    # X_train = sc_X.fit_transform(X_train)
    # X_test = sc_X.transform(X_test)
    # sc_y = StandardScaler()
    # y_train = sc_y.fit_transform(y_train)

    # Fitting Simple Linear Regression to the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    # regressor.fit(X_train, y_train)
    regressor.fit(X, y)

    # y_pred_train = regressor.predict(X_train)
    # Predicting the Test set results
    # y_pred = regressor.predict(X_test)

    y_pred = regressor.predict(X)

    score = regressor.score(X,y)

    # return_dic = {'X_train': X_train, 'y_train' :y_train, 
    #                    'X_test' :X_test, 'y_test':y_test, 
    #                    'y_pred_train' : y_pred_train, 'y_pred' : y_pred}

    return_dic = {'X': X, 'y': y, 'y_pred': y_pred, 'score': score}
    
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(y_test, y_pred)

    # print(cm)

    return return_dic

## Function to print the Graph ##
def SimpleLinear_graph(request):

    pred_values = simpleLinear()

    # X_train = pred_values['X_train'][:,0].tolist()
    # y_train = pred_values['y_train']
    # y_pred_train = pred_values['y_pred_train']

    # X_test = pred_values['X_test'][:,0].tolist()
    # y_test = pred_values['y_test']
    X = pred_values['X'][:,0].tolist()
    y = pred_values['y']
    y_pred = pred_values['y_pred']
    score = pred_values['score']

    # fig1 = go.Figure()
    # fig1.add_trace(go.Scatter(x=X_train, y=y_train,
    #     mode='markers', name='markers',
    #     marker_color='green'))
        
    # fig1.add_trace(go.Scatter(x=X_train, y=y_pred_train,
    #     mode='lines', name='lines',
    #     marker_color='red'))
        
    # fig1.update_layout(
    #         autosize=False,
    #         width=500,
    #         height=400,
    #         yaxis = dict(
    #             title_text = "Actual Output"
    #             ),
    #         xaxis = dict(
    #             title_text = "Independent Variables"
    #         ))


    # plt_div1 = plot(fig1, output_type='div')

    ## Calculate the accuracy of model


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y,
            mode='markers', name='actual values',
            marker_color='green'))
        
    fig.add_trace(go.Scatter(x=X, y=y_pred,
            mode='lines', name='predicted',
            marker_color='red'))

    fig.update_layout(
            autosize=False,
            width=500,
            height=400,
            yaxis = dict(
                title_text = "Predicted Output"
                        ),
            xaxis = dict(
                title_text = "Independent Variables"
                        )
            )

    plt_div = plot(fig, output_type='div')

    context = {
        'plt_div'   : plt_div,
        'score'     : score,
        'modelType' : dataFields.modelType,
        'modelName' : dataFields.modelName,
    }
        # cm = simpleLinear.simpleLinear()
        # print(cm)
    return context