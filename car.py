# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:18:31 2020

@author: zeynep_endes
"""
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.model_selection import cv_scores

dataset = pd.read_csv("car_data.csv")
print(dataset.head(1000))
print("\n car names:\n",dataset["Name"].value_counts())
caaaarr = pd.read_csv("car_data.csv")
print(caaaarr.head(1000))
print("\nFuel_Type missing value \n",sum(caaaarr["Fuel_Type"].isnull()))
make_name = dataset["Name"].str.split(" ", expand = True)
dataset["Manufacturer"] = make_name[0]
dataset.drop(['Name'],axis=1,inplace=True)

plt.figure(figsize = (20, 8))
plot = sns.countplot(x = 'Manufacturer', data = dataset)
plt.xticks(rotation = 90)
for p in plot.patches:
    plot.annotate(p.get_height(), 
                        (p.get_x() + p.get_width() / 2.0, 
                         p.get_height()), 
                        ha = 'center', 
                        va = 'center', 
                        xytext = (0, 5),
                        textcoords = 'offset points')
plt.title("Count of cars based on manufacturers")
plt.xlabel("Manufacturer")
plt.ylabel("Count of cars")

dataset.drop(['Location'],axis=1,inplace=True)
curr_time = datetime.datetime.now()
dataset['Year']=dataset['Year'].apply(lambda x : curr_time.year - x)
print("\n\nyear\n\n",dataset.head)
#control of missing values
print("\nFuel_Type missing value counts\n",sum(dataset["Fuel_Type"].isnull()))
print("\nTransmission missing value counts\n",sum(dataset["Transmission"].isnull()))
print("\nOwner_Type missing value counts\n",sum(dataset["Owner_Type"].isnull()))

#'Fuel_Type','Transmission','Owner_Type these columns are categorical
#it should be converted into dummy variable
dummy_Fuel_Type=pd.get_dummies(dataset["Fuel_Type"],drop_first=True)
print("\n\nFuel_Type dummy variable\n",dummy_Fuel_Type)
dummy_Transmission=pd.get_dummies(dataset["Transmission"],drop_first=True)
print("\n\nTransmission dummy variable\n",dummy_Transmission)
dummy_Owner_Type=pd.get_dummies(dataset["Owner_Type"],drop_first=True)
print("\n\nOwner_Type dummy variable\n",dummy_Owner_Type)

dataset=pd.concat([dataset,dummy_Fuel_Type],axis=1)
dataset=pd.concat([dataset,dummy_Transmission],axis=1)
dataset=pd.concat([dataset,dummy_Owner_Type],axis=1)
dataset.drop(['Fuel_Type'],axis=1,inplace=True)
dataset.drop(['Transmission'],axis=1,inplace=True)
dataset.drop(['Owner_Type'],axis=1,inplace=True)
print("\n\n missing value hangi columnlarda var\n\n",dataset.isnull().sum())

#215 kh :delete kh for all column
#change the name of the mileage column
dataset["Mileage"] = dataset["Mileage"].str.split(" ", expand = True) 
dataset["Mileage"] = pd.to_numeric(dataset["Mileage"], errors = 'coerce')
print("dataset Mileage",dataset["Mileage"])

#control of missing values for mileage 
print(sum(dataset["Mileage"].isnull()))
#fill the missing values with the mean of that column
dataset["Mileage"].fillna(dataset["Mileage"].astype("float64").mean(), inplace = True)
dataset["Engine"] = dataset["Engine"].str.split(" ", expand = True)
dataset["Engine"].fillna(dataset["Engine"].astype("float64").mean(), inplace = True)
print("\n\ndataset engine\n\n",dataset["Engine"])

dataset["Power"] = dataset["Power"].str.split(" ", expand = True)
dataset["Power"] = pd.to_numeric(dataset["Power"], errors = 'coerce')
dataset["Power"].fillna(dataset["Power"].astype("float64").mean(), inplace = True)
print("\n\ndataset power\n\n",dataset["Power"])

dataset["Seats"] = pd.to_numeric(dataset["Seats"], errors = 'coerce')
dataset["Seats"].fillna(dataset["Seats"].astype("float64").mean(), inplace = True)
print("\n\ndataset Seats\n\n",dataset["Seats"])

dataset.drop(["New_Price"], axis = 1, inplace = True)

#delete first column it is index
dataset = dataset.iloc[:, 1:]
plt.figure(figsize=(25, 6))
df = pd.DataFrame(dataset.groupby(['Manufacturer'])['Price'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Company Name vs Average Price')
plt.show()

dummy_Manufacturer=pd.get_dummies(dataset["Manufacturer"],drop_first=True)
print("\n\nManufacturer dummy variable\n",dataset["Manufacturer"])

#the name of the car is converted into car company name
dataset=pd.concat([dataset,dummy_Manufacturer],axis=1)
dataset.drop(["Manufacturer"], axis = 1, inplace = True)
print("\ndummy manufacturer\n",dummy_Manufacturer)

#Normalization process for dataset
scaler = MinMaxScaler()
num_vars = ['Kilometers_Driven', 'Seats', 'Power', 'Mileage', 'Year',
            'Power','Diesel','Price','Engine' , 'Diesel', 'Electric','LPG','Petrol',
            'Manual', 'Fourth & Above','Second','Third','Audi','BMW','Bentley',
            'Chevrolet','Datsun','Fiat','Force','Ford','Honda', 'Hyundai','ISUZU',
            'Isuzu','Jaguar','Jeep','Lamborghini','Land','Mahindra','Maruti', 
            'Mercedes-Benz','Mini','Mitsubishi','Nissan','Porsche','Renault','Skoda',
            'Smart','Tata','Toyota','Volkswagen','Volvo']
dataset[num_vars] = scaler.fit_transform(dataset[num_vars])

print(dataset.describe())
print("\n\ncolumns after normalization process\n\n",dataset.columns)
print("\n\n is there any missing value,if there is ,where:\n\n",dataset.isnull().sum())
print("\n\ntype of Seats\n\n",dataset["Seats"].head(1))
print("\n\nKilometers_Driven\n\n",dataset["Kilometers_Driven"].head(1))
print("\n\ntype of Power\n\n",dataset["Power"].head(1))
print("\n\nEngine\n\n",dataset["Engine"].head(1))
print("\n\nMileage\n\n",dataset["Mileage"].head(1))
print("\n\nYear\n\n",dataset["Year"].head(1))
print("\n\ntype of Power\n\n",dataset["Power"].head(1))
print("\n\ntype of Diesel\n\n",dataset["Diesel"].head(1))

#assigning the independent variable to the X
X=dataset[['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats',
       'Diesel', 'Electric', 'LPG', 'Petrol', 'Manual',
       'Fourth & Above', 'Second', 'Third', 'Audi', 'BMW', 'Bentley',
       'Chevrolet', 'Datsun', 'Fiat', 'Force', 'Ford', 'Honda', 'Hyundai',
       'ISUZU', 'Isuzu', 'Jaguar', 'Jeep', 'Lamborghini', 'Land', 'Mahindra',
       'Maruti', 'Mercedes-Benz', 'Mini', 'Mitsubishi', 'Nissan', 'Porsche',
       'Renault', 'Skoda', 'Smart', 'Tata', 'Toyota', 'Volkswagen', 'Volvo']]

#assigning the dependent variable to Y 
Y=dataset['Price']
print("X type:",type(X))#dataframe

##random state is used as a random number generation kernel and can be any integer
X_train,X_test,y_train,y_test=train_test_split(X,Y,train_size=0.7,test_size=0.3,random_state=100)
   
#burada modelleri bir listenin içerisine alıp parametreleri ile beraber tanımlıyoruz.
models = []
models.append(('Random Forest Regression',RandomForestRegressor(n_estimators=100,random_state=0))) 
models.append(('Linear  Regression', LinearRegression()))
models.append(('Multi Layer Perceptron', MLPRegressor()))

#here we compare all the results of cross validation by trying individual models through a loop.
folds=KFold(n_splits=10,shuffle=True,random_state=200)

for name,model in models:
    model=model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    visualizer = cv_scores(model, X_train, y_train, cv=folds, scoring='r2')
    visualizer.fit(X_train,y_train)
    visualizer.show()

list_linear=[]
list_preceptron=[]
list_forest=[]
X1=X
X2=X

for column_name in X1:
    if X1.isnull==True:
        break;
      
    for name, model in models:
        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        scores =cross_val_score(model,X_train,y_train, cv=folds, scoring='r2')
        #The smaller the rmse value, the closer to zero, the more perfect the model
        #measure test error
        rmsee=np.sqrt(metrics.mean_squared_error(y_test,y_pred))
        print('\nname,RMSE',name,round(rmsee*100,2))
       
        if(name=="Linear  Regression"):
            list_linear.append(round(rmsee*100,2))
        
        if(name=="Multi Layer Perceptron"):
            list_preceptron.append(round(rmsee*100,2))   
            
        if(name=="Random Forest Regression"):
            list_forest.append(round(rmsee*100,2)) 
    X_train,X_test,y_train,y_test=train_test_split(X,Y,train_size=0.7,test_size=0.3,random_state=100)
    X=X2
    X=X.drop([column_name],axis=1)
    print("columns",X.columns)
    print("column_name",column_name)
                          
X=dataset[['Year','Mileage','Engine','Power','Seats','Diesel','Fourth & Above','Second']]
X_train,X_test,y_train,y_test=train_test_split(X,Y,train_size=0.7,test_size=0.3,random_state=100)

for name,model in models:
    model=model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    rmsee=np.sqrt(metrics.mean_squared_error(y_test,y_pred))
    print('\nname,RMSE',name,round(rmsee*100,2))
  
#looking at the above results, this is the model I chose, namely "Random Forest Regressor".
regressor=RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(X_train,y_train)
reg_pred=regressor.predict(X_test)
scores =cross_val_score(regressor,X_train,y_train, cv=folds, scoring='r2')

print("metrics.r2_score(y_test,reg_pred) scoree",metrics.r2_score(y_test,reg_pred))

#looking at the distribution of error terms is acceptable
# because it looks like a normal distribution//plot
fig=plt.figure()
sns.distplot((y_test-reg_pred),bins=50)
fig.suptitle('ErrorTerms',fontsize=20)
plt.xlabel('y_test-y_pred',fontsize=18)
plt.ylabel('Index',fontsize=16)
plt.show()

#plotting of actual and predicted values: purple line: actual, blue line: predictions//plot
a=[i for i in range(1,1807,1)]
fig=plt.figure()
plt.plot(a,y_test,color="purple",linewidth=2.5,linestyle="-") #plotting actual
plt.plot(a,reg_pred,color="blue",linewidth=2.5,linestyle="-") #plotting predict
fig.suptitle('Actual(purple) and Predicted(blue)',fontsize=20)   #plot heading
plt.xlabel("Index",fontsize=18)     #X_label
plt.ylabel("Car Price",fontsize=16)     #Y_label
print(fig)
plt.show()

#Seeing the distribution of real and predicted values as plot in y_test and y_pred//plot
plt1=plt
fig=plt1.figure()
plt1.scatter(y_test,reg_pred)
fig.suptitle('y_test vs y_pred',fontsize=20)  #plot heading
plt1.xlabel('y_test',fontsize=18)       # X-label
plt1.ylabel('y_pred',fontsize=16)        #Y-label




