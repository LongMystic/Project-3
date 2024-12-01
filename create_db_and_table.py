import pymysql
import pymysql.cursors

connection = pymysql.connect(
    host='localhost',
    user='root',
    password='juggernautlong2003',
    database='prj3',
    cursorclass=pymysql.cursors.DictCursor
)


def main():
    cursor = connection.cursor()

    # create db
    cursor.execute("""
        CREATE DATABASE IF NOT EXISTS prj3;
    """)

    cursor.execute("""
        DROP TABLE IF EXISTS data;
    """)

    cursor.execute("""
        CREATE TABLE data (
            id INT NOT NULL AUTO_INCREMENT,
            price DECIMAL(5, 2) NOT NULL,
            warehouse_capacity INT NOT NULL,
            truck_capacity INT NOT NULL,
            date DATE NOT NULL,
            PRIMARY KEY (id)
        );
    """)
    print("Table created successfully!")
    connection.commit()


if __name__ == "__main__":
    main()
