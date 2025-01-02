import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta


def merge_data(order_df, whCapacities_df):
    df = pd.merge(left=order_df, right=whCapacities_df,
                  left_on="Plant Code", right_on="Plant ID",
                  how="inner")
    df = df[df['Plant Code'] == 'PLANT03']

    faker = Faker()

    truck_cnt = []
    for i in range(len(df)):
        truck_cnt.append(faker.random_int(min=5, max=35))
    df['Truck Count'] = truck_cnt
    return df


def main():
    # modify_data()
    order_df = pd.read_csv('./data/OrderList.csv')
    whCapacities_df = pd.read_csv('./data/WhCapacities.csv')
    merged_df = merge_data(order_df, whCapacities_df)
    merged_df.to_csv("cleaned_data.csv", index=False)


if __name__ == "__main__":
    main()
