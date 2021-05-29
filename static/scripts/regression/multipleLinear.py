# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from csvApp.models import dataFields

from plotly.offline import plot
import plotly.graph_objs as go

def multipleLinear():

    # Importing the dataset

    temp_csv_file=os.getcwd() + "\\temp_files\\tempcsv.csv"
    dataset = pd.read_csv(temp_csv_file)

    var = dataFields.y_field
    y = dataset[var].values

    ## Loop over all the columns od X and check if its a text field
    ## Preapre a list of non text columns
    ## ******* For time being excluding the text columns ********** 
    
    xlist=[]

    for col in dataset.columns:
        if not isinstance(dataset[col].values.tolist()[0],str) and col != var:
            xlist.append(col)

    X = dataset[xlist].values

    # For time being encoding of Catagorical variables is not required. 

    # from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    # labelencoder = LabelEncoder()
    # X[:, 3] = labelencoder.fit_transform(X[:, 3])
    # from sklearn.compose import ColumnTransformer
    # cl = ColumnTransformer([("countries", OneHotEncoder(), [3])], remainder="passthrough")
    # X = cl.fit_transform(X)


    # Avoiding Dummy Variable trap. 
    # X=X[:, 1:]

    # Fitting multiple limnear regression to our model. 
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X, y)

    y_pred = regressor.predict(X)

    score = regressor.score(X,y)

    # X_axis = np.array([i for i in range(1,len(X)+1)])

        # Applying backward elimination
    # Adding b0 constant to the data set
    #import statsmodels.formula.api as sm
    import statsmodels.api as sm
    X=np.append(arr = np.ones((len(X),1)).astype(int), values = X, axis = 1)

    sigLevel = 0.05
    X_opt = X
    X_opt = np.array(X_opt, dtype=float)
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    pVals = regressor_OLS.pvalues

    while pVals[np.argmax(pVals)] > sigLevel:
        X_opt = np.delete(X_opt, np.argmax(pVals), axis = 1)
        print("pval of dim removed: " + str(np.argmax(pVals)))
        print(str(X_opt.shape[1]) + " dimensions remaining...")
        X_opt = np.array(X_opt, dtype=float)
        regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
        pVals = regressor_OLS.pvalues

    X_opt = X_opt[:, 1:]

    # X_opt = np.array(X_opt, dtype=int)

    return_dic = {'X': X_opt, 'y' :y, 
                  'y_pred' : y_pred,
                  'score': score}

    # Visualising the Training set results
    # plt.scatter(X_axis, y, color = 'red')
    # plt.plot(X_axis, y_pred, color = 'blue')
    # plt.title('Salary vs Experience (Training set)')
    # plt.xlabel('Years of Experience')
    # plt.ylabel('Salary')
    # plt.show()

    # Applying backward elimination
    # Adding b0 constant to the data set
    # #import statsmodels.formula.api as sm
    # import statsmodels.api as sm
    # X=np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

    # sigLevel = 0.05
    # X_opt = X[:,[0,1,2,3,4]]
    # X_opt = np.array(X_opt, dtype=float)
    # regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    # pVals = regressor_OLS.pvalues

    # while pVals[np.argmax(pVals)] > sigLevel:
    #     X_opt = np.delete(X_opt, np.argmax(pVals), axis = 1)
    #     print("pval of dim removed: " + str(np.argmax(pVals)))
    #     print(str(X_opt.shape[1]) + " dimensions remaining...")
    #     X_opt = np.array(X_opt, dtype=float)
    #     regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    #     pVals = regressor_OLS.pvalues
    
    # regressor_OLS.summary()


    # X_opt = np.array(X_opt, dtype=int)

    return return_dic

def multipleLinear_graph(request):

    pred_values = multipleLinear()

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