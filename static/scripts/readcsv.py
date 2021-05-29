import pandas as pd
import cgi, os
import cgitb; cgitb.enable()
form = cgi.FieldStorage()

# fileitem = form['filePath']

# if fileitem.filePath:
#     fn = os.path.basename(fileitem.filePath)
#     print(" I am here")
#     print(fn)

def getCsvFile():
    dataset = pd.read_csv("csvApp/test.csv")
    print(dataset)

def printTest(field, modelName):
    print(field + '  '+ modelName)