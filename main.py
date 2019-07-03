"""
@author: JAYRAJ
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection,metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


#url = "F:\ML\BigMarket\Train.csv"
train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train,test],ignore_index=True,sort=True)
print(train.shape)
print(test.shape)
print(data.shape)

##Checking null values
print(data.apply(lambda x: sum(x.isnull())))

print(data.describe())

#check unique values 
print(data.apply(lambda x: len(x.unique())))

#filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]

#print frequency of categories
for i in categorical_columns:
    print('\nFrquency of Categories for variable ',i)
    print(data[i].value_counts())


#Impute data and check #missing values before and after imputation to confirm for weight
print('Original missing values in weight',sum(data['Item_Weight'].isnull()))

data.fillna({"Item_Weight":data['Item_Weight'].median()},inplace=True)
print('Original missing values in weight after imputing',sum(data['Item_Weight'].isnull()))

#Impute data and check #missing values before and after imputation to confirm for Outlet_Size
print('Original missing values in outlet size',sum(data['Outlet_Size'].isnull()))
data.fillna({"Outlet_Size":data['Outlet_Size'].mode()[0]},inplace=True)
print('Original missing values in outlet size after imputing',sum(data['Outlet_Size'].isnull()))

#Impute data and check #missing values before and after imputation to confirm for Item_Visibility
print('Original missing values in Item_Visibility',sum(data['Item_Visibility']==0))
data['Item_Visibility'] = data['Item_Visibility'].replace(0,data['Item_Visibility'].mean())
print('Original missing values in Item_Visibility after imputing',sum(data['Item_Visibility']==0))


#Check for Item_Fat_Content
print(data['Item_Fat_Content'].value_counts())
data["Item_Fat_Content"] = data["Item_Fat_Content"].replace({"LF":"Low Fat","reg":"Regular","low fat":"Low Fat"})
print(data['Item_Fat_Content'].value_counts())


#One-Hot-Coding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])

data = pd.get_dummies(data,columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type','Outlet'])

#Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

#Drop unnecessary columns
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions
train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)

#Creating Baseline Model
#Mean based:
mean_sales = train['Item_Outlet_Sales'].mean()
base1 = test[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales
#Export submission file
base1.to_csv("alg0.csv",index=False)

#Function for predictive model
def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    #cv_score = cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    #cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    #print("\nModel Report")
    #print("RMSE : %.4g" % np.sqrt(mean_squared_error(dtrain[target].values, dtrain_predictions)))
    #print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)


#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']

#Linear Regression Model
predictors = [x for x in train.columns if x not in [target]+IDcol]
print(predictors)
alg1 = LinearRegression(normalize=True)
modelfit(alg1,train,test,predictors,target,IDcol,'alg1.csv')
coef1 = pd.Series(alg1.coef_,predictors).sort_values()
coef1.plot(kind='bar',title = 'LinearRegression Model Coefficients')


