import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class HousingPricePredictor:

    def __init__(self):
        self.data = None
        self.model = None

    def load_data(self): #--Load CSV file
        try:
            self.data = pd.read_csv('housing_data.csv')

        except FileNotFoundError as e:
            print(f"Error: {e}. CSV file was not found.")

    def explore_data(self): #--Print data to console
        print(self.data.describe())

    def train_model(self): #--Train model, Display: coef., intercept, R score
        X = self.data[['Size']]
        y = self.data['Price']

        self.model = LinearRegression()
        self.model.fit(X, y)

        print(f"Coefficient: ${self.model.coef_[0]:,.2f}")
        print(f"Intercept: ${self.model.intercept_:,.2f}")

        r_squared = self.model.score(X, y)
        print(f"R² Score: {r_squared:.2f}")

    def visualize_model(self): #--Display model predictions
        predictions = self.model.predict(self.data[['Size']])
        plt.scatter(self.data['Size'], self.data['Price'], label = 'Actual Prices')
        plt.plot(self.data['Size'], predictions, label = 'Regression Line')
        plt.xlabel("House Size (sq ft)")
        plt.ylabel("Price ($)")
        plt.title("Housing Price Prediction")
        plt.legend()
        plt.grid()
        plt.show()

    def predict_price(self): #--Predict house price
        while True:
            try:
                size = int(input("Enter house size in square feet: "))

                if size <= 0:
                    print("Please enter a positive number.")
                    continue

                prediction = self.model.predict(pd.DataFrame({'Size': [size]}))
                print(f"Estimated house price for {size} sqft. house: ${prediction[0]:,.2f}")
                break

            except ValueError:
                print("Please enter a valid number.")

def main(): #--Run program
    predictor = HousingPricePredictor()
    predictor.load_data()
    predictor.explore_data()
    predictor.train_model()
    predictor.visualize_model()
    predictor.predict_price()

if __name__ == '__main__':
    main()
