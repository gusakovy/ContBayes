import csv
import pandas as pd


def append_dict_to_csv(data_dict: dict, file_name: str):
    """
    Appends a dictionary as a row to a csv file. If the file does not exist, it will be created.

    :param data_dict: dictionary to be appended
    :param file_name: name of csv file to append to
    """

    try:
        with open(file_name, 'x', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data_dict.keys())
            writer.writeheader()
    except FileExistsError:
        pass

    with open(file_name, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data_dict.keys())
        writer.writerow(data_dict)


def convert_string_to_value(val: any):
    """
    Converts a string to an integer or float value (if possible).

    :param val: value to be converted
    """

    if isinstance(val, str):
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                pass
    return val


def is_matching_row_in_csv(data_dict: dict, file_name: str):
    """
    Checks if there exists a row in the csv file that matches a given dictionary.

    :param data_dict: dictionary representing a row in csv file
    :param file_name: name of csv file to search
    """

    try:
        df = pd.read_csv(file_name)

        data_list = df.to_dict('records')
        for record in data_list:
            for key, value in record.items():
                record[key] = convert_string_to_value(value)

        for row in data_list:
            if data_dict == {k: row[k] for k in data_dict.keys() if k in row.keys()}:
                return True
    except FileNotFoundError:
        print(f"The file {file_name} does not exist.")

    return False
