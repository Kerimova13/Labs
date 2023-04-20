import pandas as pd


excel_file = pd.read_excel('flats.xlsx')


field_name = input("Give a filter name: ")
filter_value =input("Give a value: ")

if pd.api.types.is_numeric_dtype(excel_file[field_name]):
    filter_value = pd.to_numeric(filter_value, errors='coerce')
else:
    filter_value = filter_value

filtered_data = excel_file[excel_file[field_name] == filter_value]

print(filtered_data)

