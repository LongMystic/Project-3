import pymysql
import pymysql.cursors
import pandas as pd

connection = pymysql.connect(
    host='localhost',
    user='root',
    password='juggernautlong2003',
    database='prj3',
    cursorclass=pymysql.cursors.DictCursor
)


def main():
    df = pd.read_csv("data.csv")
    cursor = connection.cursor()
    for i in range(len(df)):
        sql = f"""
            INSERT INTO prj3.data (
                price, warehouse_capacity, truck_capacity, date
            )
            VALUE (
                {df.iloc[i]['price']}, 
                {df.iloc[i]['warehouse_capacity']}, 
                {df.iloc[i]['truck_capacity']}, 
                \'{df.iloc[i]['date']}\'
            );
        """
        cursor.execute(sql)
        print(f"row {i} is inserted with sql {sql}")
    connection.commit()
    connection.close()


if __name__ == "__main__":
    main()

