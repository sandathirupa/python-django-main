from django.shortcuts import render
from io import StringIO
import pandas as pd
import os
# from static.scripts.regression import simpleLinear
from static.scripts.regression import simpleLinear, multipleLinear, polynomial
from static.scripts.classification import knn, kernal_svm, logistic, naive_bayes, svm

# from static.scripts import readcsv
from csvApp.models import dataFields
from sklearn.metrics import confusion_matrix


# Create your views here.
def home_view(request):
    context={}
    return render(request, 'home.html', context)


# Read and display the csv file
def displayFile(request):
    context={}

    if request.method == 'POST':
        result = True
        try:
            request.FILES['filePath'].name
        except:
            result = False

        if result == False:
            print(" Please select approriate file")
            Error_op="ERROR !!! Please select the file first!!! "
            context = {
                'Error_op' : Error_op,
            }
            return render(request, 'home.html', context)
        else:
            # Read the file uploaded from input submit
            test_file = request.FILES['filePath'].read()
            # Convert the file from binary to readable format
            data=test_file.decode("utf-8")
            # Create dataframe using the decodated data StringIO will convert it to string 
            df = pd.read_csv(StringIO(data))
            # Create the temp file path
            temp_csv_file=os.getcwd() + "\\temp_files\\tempcsv.csv"

            df.to_csv(temp_csv_file, index=False)

            df_head = df.head()

            context = {
                'dataframe' : df_head,
                'headers'   : df.columns,   
            }

            return render(request, 'fileview.html', context)
    
# select the Dependent Variable
# Stored in model - - dataFields.y_field
def fieldSelection(request):
    context={}
    temp_csv_file=os.getcwd() + "\\temp_files\\tempcsv.csv"
    df = pd.read_csv(temp_csv_file)
    print(df.columns)

    if request.method == 'POST':
        
        y_field = request.POST.get("radio")
        print(y_field)

        x_fields = request.POST.getlist('chkbox')
        print(x_fields)

        context = {
            'Y_field' : y_field,
            'x_fields' : x_fields
        }
        dataFields.y_field = y_field
        dataFields.x_fields = x_fields

        print(dataFields.x_fields)
    
        # for col in df.columns:
        #     colname = col + '_chk'
        #     print(request.POST.get(colname))

        
    return render(request, 'modelSelect.html', context)

# Select the Model Type Regression / Clasification
# Stored in model - - dataFields.modeltype
def modelTypeSelection(request):
    context={}

    if request.method == 'POST':
        modeltype = request.POST.get("radio")
        print(modeltype)
        dataFields.modelType = modeltype
        context = {
            'modelType' : modeltype,
        }

    return render(request, 'modelSelect.html', context)

# Select Specific Model 
# Stored in model - - dataFields.modelName
def modelNameSelection(request):
    
    if request.method == 'POST':
        modelName = request.POST.get("radio")
        dataFields.modelName = modelName

        print(modelName)
        # print(dataFields.y_field)
        # print(dataFields.modelType)
        # print(dataFields.modelName)

        if modelName == "SimpleLinear":
            context = simpleLinear.SimpleLinear_graph(request)

            return render(request, 'graphs.html', context)
        
        if modelName == "MultipleLinear":
            context = multipleLinear.multipleLinear_graph(request)

            return render(request, 'graphs.html', context)

        if modelName == "Polynomial":
            context = polynomial.polynomial_graph(request)

            return render(request, 'graphs.html', context)

        if modelName == "K-NN":
            print('Inside KNN......')
            context = knn.Knn_graph(request)

            return render(request, 'graphs.html', context)

        if modelName == "KernelSVM":
            context = kernal_svm.Kernal_SVM_graph(request)

            return render(request, 'graphs.html', context)
        
        if modelName == "NaiveBayes":
            context = naive_bayes.Naive_Bayes_graph(request)

            return render(request, 'graphs.html', context)
        
        if modelName == "SVM":
            context = svm.SVM_graph(request)

            return render(request, 'graphs.html', context)



        # if modelName == "SimpleLinear":
        #     SimpleLinear_graph()

        # if modelName == "SimpleLinear":
        #     SimpleLinear_graph()

        # if modelName = "SimpleLinear":
        #     SimpleLinear_graph()

        

