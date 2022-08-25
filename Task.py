import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import model_selection
#Reading All CSV files
scan3= pd.read_csv("Scanner_3.csv")
scan4= pd.read_csv("scanner_4.csv")
scan11= pd.read_csv("scanner_11.csv")
scan22= pd.read_csv("Scanner_22.csv")
scan41= pd.read_csv("Scanner_41.csv")
scan74= pd.read_csv("scanner_74.csv")
scan52= pd.read_csv("scannerr_52.csv")
Ref= pd.read_csv("Reference values_3fills.csv")

#Combining Them in one DataFranme
df1= pd.merge(pd.merge(scan3,scan4,on='WVN CM-1'),scan11,on='WVN CM-1')
df2=pd.merge(pd.merge(scan22,scan41,on='WVN CM-1'),scan74,on='WVN CM-1')
df3=pd.merge(pd.merge(scan52,df1,on='WVN CM-1'),df2,on='WVN CM-1')
df3["Reading Number"]=df3["WVN CM-1"]%10
df3["Sample ID"]=(df3["WVN CM-1"]/10).astype(int)
df4=pd.merge(df3,Ref,on='Sample ID')
df4.drop("WVN CM-1", axis=1, inplace=True)

#Delete Duplicates Rows
df5 = df4.drop_duplicates(keep='first')

#Check Null values
print(df5.isnull().values.any()) #there is no null values
print(df5.isnull().sum())

#Checking Outliers
column_headers = list(df5.columns.values)
plt.boxplot(df5["3948.74189980328_x"])
plt.show()
#Checking Distribution
plt.hist(df5["3948.74189980328_x"])
plt.show()
plt.plot(df5["3948.74189980328_x"])
plt.show()

#Scatter plot for two variables
plt.scatter(df5["3948.74189980328_x"],df5["3975.97501054406_x"])
plt.show()
plt.scatter(df5["3948.74189980328_x"],df5["Starch/SUM"])
plt.show()
plt.scatter(df5["3948.74189980328_x"],df5["TLC/SUM"])
plt.show()

#Correlation Matrix
Correlation=df5.corr()

#Heat Map
#sns.heatmap(Correlation,cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=df5.columns, xticklabels=df5.columns, cmap="Spectral_r")
#plt.show()

#Realtion Between Variables
#sns.pairplot(df5)
#plt.show()

#delete outliers
print(df5)
for col in df5:
  if pd.api.types.is_numeric_dtype(df5[col]) and (len(df5[col].value_counts()) > 0) and not all(df5[col].value_counts().index.isin([0, 1])):

    q1 = df5[col].quantile(.25)
    q3 = df5[col].quantile(.75)
    min = q1 - (1.5 * (q3 - q1))
    max = q3 + (1.5 * (q3 - q1))
    for i in range(len(df5[col])):
        if df5.iloc[i][col]>max or df5.iloc[i][col]<min:
            df5.iloc[i][col]=min
print(df5)
'''
Q1 = df5[column_headers].quantile(0.25)
Q3 = df5[column_headers].quantile(0.75)
IQR = Q3 - Q1
LB= Q1 - (1.5* IQR)
UB= Q3 + (1.5*IQR)
#print(df5[column_headers][~(Lout|Uout)])
#df = df5[~((df5[column_headers] < (Q1 - 1.5 * IQR)) |(df5[column_headers] > (Q3 + 1.5 * IQR))).any(axis=1)]
#print(df)
'''

#Prediction Model

# X and Y data
X=df5.iloc[:,0:1803]
Y=df5["TLC/SUM"]
#Encoding data for string values as i got error in it
X = X.apply(pd.to_numeric, errors='coerce')
Y = Y.apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)
Y.fillna(0, inplace=True)

#model=SelectKBest(chi2,k=500)
#new=model.fit(X,Y)
#print(new)
#X_new=new.transform(X)
#print(X_new)
#features=new.get_support(indices=True)
#X_new1=X.iloc[:,features]
#print(X_new1)

#Split data to train and test
X_train= X[:(int((len(X)*0.8)))]
X_test=  X[(int((len(X)*0.8))):]
Y_train= Y[:(int((len(Y)*0.8)))]
Y_test = Y[(int((len(Y)*0.8))):]
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
Y_pred=regr.predict(X_test)
#Accuracy
Accuracy=r2_score(Y_test,Y_pred)
print(" Accuracy of the model is: ",Accuracy)

#Mean Square Error Matrix
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearRegression()
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='neg_mean_squared_error')
print("MSE: %.3f (%.3f)" % (results.mean(), results.std()))

#R^2 Matrix
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring="r2")
print("R^2: %.3f (%.3f)" % (results.mean(), results.std()))

#Mean Absolute Error
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='neg_mean_absolute_error')
print("MAE: %.3f (%.3f)" % (results.mean(), results.std()))
#plot the actual vs predict to show thw relation between them
plt.scatter(Y_test,Y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
sns.regplot(x=Y_test,y=Y_pred,ci=None,color ='red')
plt.show()
