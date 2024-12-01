from faker import Faker
import pandas as pd

# id
# price
# unit
# inventory *
# delivery *
# date


faker = Faker()

start_date = "2024-01-01"
min_capacity = 50
max_capacity = 80
min_delivery = 40
max_delivery = 70


def main():
    data = {
        "price": [],
        "warehouse_capacity": [],
        "truck_capacity": [],
        "date": []
    }

    number_of_records = 100

    for i in range(number_of_records):
        price = faker.pyfloat(left_digits=3, right_digits=2, positive=True, min_value=1, max_value=100)
        warehouse_capacity = faker.random_int(min=min_capacity, max=max_capacity)
        truck_capacity = faker.random_int(min=min_delivery, max=max_delivery)
        date = pd.to_datetime(start_date) + pd.Timedelta(days=i)

        # add to dict
        data["price"].append(price)
        data["warehouse_capacity"].append(warehouse_capacity)
        data["truck_capacity"].append(truck_capacity)
        data["date"].append(date)

    pd.DataFrame(data).to_csv("data.csv", index=False)


if __name__ == "__main__":
    main()
