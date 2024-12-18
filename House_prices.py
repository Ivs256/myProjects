import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import pickle
data=pl.read_csv("real_estate_dataset.csv")
X=data["Square_Feet","Num_Bedrooms","Num_Floors","Year_Built","Distance_to_Center"]
Y=data["Price"]
x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.8)
model=LinearRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
mean=mean_squared_error(y_test,prediction)
r_score=r2_score(y_test,prediction)
print(prediction)
print("Mean-Square-Error:",mean)
print("r^2 Score:",r_score)
with open('model.pkl', 'wb') as f:
    pickle.dump(model,f)