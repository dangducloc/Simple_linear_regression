import numpy as np
from crawling_data import Crawler
from sklearn.linear_model import LinearRegression   
from Read_data import GetDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from visualize import Visualize

class Prediction:


    def __init__(self) -> None:
        self.dataset = self.Get_Format_dataset()
        self.model = self.Training_model()

    def Get_Format_dataset(self):
        # Load and format the dataset
        obj = GetDataset()
        datas = obj.fomated_Data()
        areas = np.array(datas[1], dtype=float).reshape(-1, 1)
        prices = np.array(datas[0], dtype=float).reshape(-1, 1)
        return [areas,prices]
    
    def Split_dataset (self):
        dataset = self.Get_Format_dataset()
        areas = dataset[0]
        prices = dataset[1]
        # Split the dataset into training and testing sets (80% train, 20% test)
        areas_train, areas_test, prices_train, prices_test = train_test_split(areas, prices, test_size=0.2, random_state=42)
        return {
            "areas_train":areas_train, 
            "areas_test":areas_test,
            "prices_train":prices_train,
            "prices_test":prices_test
            }

    def Training_model (self):
        data = self.Split_dataset()
        areas_train = data["areas_train"]
        prices_train = data["prices_train"]
        model = LinearRegression().fit(areas_train, prices_train)
        return model

    def Get_solpe_intercept (self):
        # Get the slope (m) and intercept (b)
        slope = self.model.coef_[0]
        intercept = self.model.intercept_
        return {
            "slope": slope[0],
            "intercept":intercept[0],
            "equation":f"y = {slope[0]}*x + {intercept[0]}"
        }

    def Auto_Prediction (self):
        # Make predictions using the trained model
        data = self.Split_dataset()
        areas_train = data["areas_train"]
        areas_test = data["areas_test"]
        predicted_prices_train = self.model.predict(areas_train)
        predicted_prices_test = self.model.predict(areas_test)
        return {
            "predicted_test" : predicted_prices_test,
            "predicted_train" : predicted_prices_train,
        }

    def Prediction (self,area):
        result = self.model.predict(np.array([area],dtype=float).reshape(-1, 1))
        return result[0]

    def Rsquared_MAE_MSE (self):
        data = self.Split_dataset()
        # Calculate the R-squared score (coefficient of determination) on the test set
        areas_test = data["areas_test"]
        prices_test = data["prices_test"]
        prices_train = data["prices_train"]

        test_score = self.model.score(areas_test, prices_test)
        predicted = self.Auto_Prediction()

        predicted_prices_train = predicted["predicted_train"] 
        predicted_prices_test = predicted["predicted_test"]

        # Calculate and print error metrics for both training and test sets
        mae_train = mean_absolute_error(prices_train, predicted_prices_train)
        mse_train = mean_squared_error(prices_train, predicted_prices_train)

        mae_test = mean_absolute_error(prices_test, predicted_prices_test)
        mse_test = mean_squared_error(prices_test, predicted_prices_test)

        return {
            "r2":test_score,
            "mae-train":mae_train,
            "mse-train":mse_train,
            "mae-test":mae_test,
            "mse-test":mse_test
        }

    def Visualize (self):
        data = self.Split_dataset()
        areas_test = data["areas_test"]
        areas_train = data["areas_train"]
        prices_test = data["prices_test"]
        prices_train = data["prices_train"]
        predicted_train = self.Auto_Prediction()["predicted_train"]
        Visualize().Visualzing(
            x_test=areas_test,
            x_train=areas_train,
            y_test=prices_test,
            y_train=prices_train,
            predicted_y_train=predicted_train
        )

#useage
m = Prediction()
# auto_predict = m.Auto_Prediction()
# print(auto_predict)
# print("===============================================================")
# a = float(input("new_area: "))
# predict = m.Prediction(a)
# print(predict)
# print("===============================================================")
error_metrics = m.Rsquared_MAE_MSE()
print(error_metrics)
print("===============================================================")
equation = m.Get_solpe_intercept()
print(equation)
print("===============================================================")
# m.Visualize()
