import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

class AIInventoryManagementSystem:
    def __init__(self, inventory_file):
        self.inventory_file = inventory_file
        self.inventory_data = pd.DataFrame()
        self.model = None
        self._load_inventory()

    def _load_inventory(self):
        if os.path.exists(self.inventory_file):
            self.inventory_data = pd.read_csv(self.inventory_file)
        else:
            self.inventory_data = pd.DataFrame(columns=['Item', 'Stock', 'Price', 'Sales', 'Restock_Level'])

    def save_inventory(self):
        self.inventory_data.to_csv(self.inventory_file, index=False)

    def add_item(self, item_name, stock, price, sales=0, restock_level=10):
        new_item = {
            'Item': item_name,
            'Stock': stock,
            'Price': price,
            'Sales': sales,
            'Restock_Level': restock_level
        }
        self.inventory_data = pd.concat([self.inventory_data, pd.DataFrame([new_item])], ignore_index=True)
        self.save_inventory()

    def update_stock(self, item_name, stock_change):
        if item_name in self.inventory_data['Item'].values:
            self.inventory_data.loc[self.inventory_data['Item'] == item_name, 'Stock'] += stock_change
            self.save_inventory()
        else:
            print(f"Item '{item_name}' not found in inventory.")

    def train_sales_forecast_model(self):
        if len(self.inventory_data) < 5:
            print("Not enough data to train the model.")
            return

        features = self.inventory_data[['Stock', 'Price']]
        target = self.inventory_data['Sales']

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Model trained with MSE: {mse}")

    def forecast_sales(self, stock, price):
        if self.model:
            return self.model.predict([[stock, price]])[0]
        else:
            print("Model is not trained yet.")
            return None

    def recommend_restock(self):
        recommendations = self.inventory_data[self.inventory_data['Stock'] <= self.inventory_data['Restock_Level']]
        print("Items that need restocking:")
        print(recommendations)

    def view_inventory(self):
        print("Current Inventory:")
        print(self.inventory_data)

# Example usage
if __name__ == "__main__":
    inventory_file = "inventory.csv"
    inventory_system = AIInventoryManagementSystem(inventory_file)

    # Add items
    inventory_system.add_item("Laptop", 50, 1000, 30, 10)
    inventory_system.add_item("Phone", 100, 500, 70, 15)
    inventory_system.add_item("Tablet", 30, 300, 15, 5)
    inventory_system.add_item("Headphones", 200, 100, 50, 20)
    inventory_system.add_item("Monitor", 40, 200, 25, 10)

    # View inventory
    inventory_system.view_inventory()

    # Update stock
    inventory_system.update_stock("Phone", -20)

    # Train model
    inventory_system.train_sales_forecast_model()

    # Forecast sales
    forecasted_sales = inventory_system.forecast_sales(40, 600)
    if forecasted_sales is not None:
        print(f"Forecasted Sales: {forecasted_sales:.2f}")

    # Recommend restock
    inventory_system.recommend_restock()
