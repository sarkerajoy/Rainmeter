#import necessary modules
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np

# import weather data of Australia
data=pd.read_csv('weatherAUS.csv')

# drop evporation & sunshine columns since they have no value
data=data.drop(columns=['Evaporation','Sunshine'])
#drop nan from target columns
data=data.dropna(subset=['RainTomorrow'])
# reset indexing number as drop just remove value not index number
raw_data=data.reset_index(drop=True)

#take all data into a new variable
input_data=raw_data.iloc[:,:21]

# divide input_data into  categorical and numerical data
numerical=input_data.select_dtypes(include=['float64'])
Categorical=input_data.select_dtypes(include=['object'])
#taking columns names accoring to numerical & categorical columns
numerical_cols=list(numerical.columns)
categorical_cols=list(Categorical.columns)


#data imputation for numerical data
imputation=SimpleImputer(strategy='mean')
imputation.fit(numerical)
p=np.array(imputation.transform(numerical))
num_scaling=pd.DataFrame(p,columns=numerical_cols)

#one hot ecoding
cat=pd.get_dummies(Categorical.iloc[:,1:-1],dtype=int)

# making the final data
final_data=pd.concat([Categorical['Date'],num_scaling,cat,Categorical['RainTomorrow']],axis=1)

#converting date as datetime
year=pd.to_datetime(final_data.Date).dt.year

#spilt into train& test dataset
train_data=final_data[year<=2015]
test_data=final_data[year>2015]

# making input & target variable for train test 
x_train,x_test=train_data.iloc[:,1:-1],train_data.iloc[:,1:-1]
y_train,y_test=train_data['RainTomorrow'],train_data['RainTomorrow']

#import random forest & accurcy score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# fit model
model=RandomForestClassifier()
model.fit(x_train,y_train)

#predict data
y_predict=model.predict(x_test)

# accuracy test
accuracy_score(y_predict,y_test)

# accuracy is nearly 99%


