import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data={'experience':[1,2,3,4,5],
      'salary':[40000,45000,50000,55000,60000]}
df_sal=pd.DataFrame(data)

x=df_sal.iloc[:,:1].values
y=df_sal.iloc[:,1:].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
model=LinearRegression()
model.fit(x_train,y_train)
y_train_pred=model.predict(x_train)
y_test_pred=model.predict(x_test)
print(f'coefficent(Slope):{model.coef_[0]}')
print(f'intercept(y-intercept):{model.intercept_}')
plt.scatter(x_test,y_test,color='red',label='training data')
plt.scatter(x_train,y_train_pred,color='blue',label='testing data')
plt.plot(x,model.predict(x),color='green',label='regression line')
plt.title('simple linear regression')
plt.xlabel(' year of experience')
plt.ylabel('salary')
plt.legend()
plt.show()
