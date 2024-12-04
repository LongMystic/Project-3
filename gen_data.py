import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()


# Generate time series data
def generate_data(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date)
    data = []

    # Parameters for seasonality and trends
    seasonal_amplitude = 50
    trend_factor = 0.1
    warehouse_base = 10
    truck_base = 10  # Truck base is now used

    for i, date in enumerate(dates):
        # Day of the year for seasonal patterns
        day_of_year = date.timetuple().tm_yday

        # Seasonal effect
        seasonal_effect = seasonal_amplitude * np.sin(2 * np.pi * day_of_year / 365)

        # Trending effect
        trend_effect = trend_factor * i

        # Generate warehouse capacity
        warehouse_capacity = warehouse_base + seasonal_effect + trend_effect + np.random.normal(0, 10)

        # Generate truck capacity with dependency and base
        truck_capacity = truck_base + (warehouse_capacity * 0.8) + np.random.normal(0, 5)

        # Generate price (random but within a range)
        price = np.random.uniform(50, 150)

        # Append row
        data.append({
            "date": date,
            "price": round(price, 2),
            "warehouse_capacity": int(warehouse_capacity),
            "truck_capacity": int(truck_capacity),
        })

    return pd.DataFrame(data)


def main():
    # Generate and display data
    data = generate_data(start_date="2021-01-01", end_date="2024-12-31")
    data.to_csv("data.csv", index=False)


if __name__ == "__main__":
    main()



