
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from csvApp.models import dataFields

from plotly.offline import plot
import plotly.graph_objs as go

def knn():

    # Importing the dataset

    print('Inside Knn function....................')

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
    from sklearn.neighbors import KNeighborsClassifier
    # classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p =2)
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean', p =2)
    classifier.fit(X_train, y_train)

    # Apply regression model 
    y_pred = classifier.predict(X_test)

    # Calculate the Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    # Visualising the Test set results
    # from matplotlib.colors import ListedColormap
    # X_set, y_set = X_test, y_test

    # X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
    #                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

    # print("##############  ###########")
    # print(X1.ravel())
    # print(X2.ravel())
    # print(np.array([X1.ravel(), X2.ravel()]))

    # print((np.array([X1.ravel(), X2.ravel()]).T).shape)

    # print("############## 22222  ###########")
    # y_pred_new = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)

    # print("y_pred_new")
    # print(y_pred_new)

    # return_dic = {'cm': cm, 'X1': X1, 'X2' : X2, 'y_pred_new' : y_pred_new, 'y_set' : y_set, 'X_set' : X_set}
    return_dic = {'cm': cm}    

    return return_dic


## Function to print the Graph ##
def Knn_graph(request):

    print('Inside Knn_graph function')
    pred_values = knn()

    # X = pred_values['X'][:,0].tolist()
    
    # y = pred_values['y']
    # X_grid = pred_values['X_grid'][:,0].tolist()
    # y_pred = pred_values['y_pred']
    print(" Inside Knn_graph")
    # X1 = pred_values['X1']
    # X2 = pred_values['X2']

    # y_pred_new = pred_values['y_pred_new']
    # y_set = pred_values['y_set']
    # X_set = pred_values['X_set']
    cm = pred_values['cm']

    print('CM = ', cm)

    cm1 = cm[0,0]
    cm2 = cm[0,1]
    cm3 = cm[1,0]
    cm4 = cm[1,1]
    
    # from matplotlib.colors import ListedColormap

    # fig = go.Figure()

    # fig.add_trace(go.Contour(x=X1, y=X2, z=y_pred_new))

    # # fig.add_trace(go.xlim(X1.min(), X1.max()))

    # # fig.add_trace(go.ylim(X2.min(), X2.max()))

    # for i, j in enumerate(np.unique(y_set)):
    #     fig.add_trace(go.Scatter(x=X_set[y_set == j, 0], y=X_set[y_set == j, 1]))

    # fig.update_layout(
    #         autosize=False,
    #         width=500,
    #         height=400,
    #         yaxis = dict(
    #         title_text = "Predicted Output"
    #                     ),
    #         xaxis = dict(
    #         title_text = "Independent Variables"
    #                     )
    #         )

    # plt_div = plot(fig, output_type='div')

    context = {
        # 'plt_div'   : plt_div,
        'modelType' : dataFields.modelType,
        'modelName' : dataFields.modelName,
        'cm1'       : cm1,
        'cm2'       : cm2,
        'cm3'       : cm3,
        'cm4'       : cm4,

    }
        # cm = simpleLinear.simpleLinear()
        # print(cm)
    return context
