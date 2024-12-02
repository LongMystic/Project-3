from datetime import datetime

start_date = "2024-01-01"
min_capacity = 50
max_capacity = 80
min_delivery = 40
max_delivery = 70


def validate_date(date):
    message = ''
    code = 0
    min_date = datetime.strptime(start_date, "%Y-%m-%d")
    try:
        date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        code = -1
        message = 'Date must be in format YYYY-MM-DD'

    if code == 0 and isinstance(date, datetime) and date < min_date:
        code = -1
        message = 'Date must be greater than or equal 2024-01-01'

    return code, message


def validate_price(price):
    message = ''
    code = 0
    try:
        price = float(price)

    except Exception as e:
        code = -1
        message = 'Price must be a float'

    if code == 0 and isinstance(price, float) and not (1 <= price <= 100):
        code = -1
        message = 'Price must be in range (1, 100)'

    return code, message


def validate_warehouse_capacity(warehouse_capacity):
    message = ''
    code = 0
    try:
        warehouse_capacity = int(warehouse_capacity)

    except Exception as e:
        code = -1
        message = 'Warehouse Capacity must be an integer'

    if code == 0 and not (min_capacity <= warehouse_capacity <= max_capacity):
        code = -1
        message = f"Warehouse Capacity must be in range ({min_capacity}, {max_capacity})"

    return code, message


def validate_truck_capacity(truck_capacity):
    message = ''
    code = 0
    try:
        truck_capacity = int(truck_capacity)

    except Exception as e:
        code = -1
        message = 'Truck Capacity must be an integer'

    if code == 0 and not (min_delivery <= truck_capacity <= max_delivery):
        code = -1
        message = f"Truck Capacity must be in range ({min_delivery}, {max_delivery})"

    return code, message
