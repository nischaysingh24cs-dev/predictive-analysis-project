import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("data/sample_sales_data.csv")
X = df[["Day"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X,y)

pred = model.predict(X)

print("MAE:", mean_absolute_error(y,pred))
print("RMSE:", np.sqrt(mean_squared_error(y,pred)))
print("R2:", r2_score(y,pred))

future = pd.DataFrame({"Day":[11,12,13,14,15]})
forecast = model.predict(future)
print("\nForecast:")
print(pd.DataFrame({"Day":future["Day"],"Predicted Sales":forecast}))

plt.scatter(df["Day"],y,label="Actual")
plt.plot(df["Day"],pred,label="Regression")
plt.legend()
plt.xlabel("Day")
plt.ylabel("Sales")
plt.title("Sales Prediction")
plt.savefig("prediction.png")
plt.show()
