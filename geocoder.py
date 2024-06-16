import re
import pandas as pd
import requests
import time

def format_address(city, street, house_number, building=None):
    house_number_parts = re.split(r',\s*', house_number)
    house_number = house_number_parts[0].strip()

    if len(house_number_parts) > 1:
        for part in house_number_parts[1:]:
            if part.startswith('корп'):
                building = re.sub(r'корп\.?\s*', 'к', part.strip())
            elif part.startswith('стр'):
                building = re.sub(r'стр\.?\s*', 'с', part.strip())

    if building:
        return f"{city}, {street}, {re.sub(r'[^\d\w]+', '', house_number)} {building}"
    else:
        return f"{city}, {street}, {re.sub(r'[^\d\w]+', '', house_number)}"

def geocode_address(address):
    url = "https://geocode-maps.yandex.ru/1.x/"
    params = {
        "apikey": "fd5e2468-cd6f-4be3-9698-2eb430e5daf8",
        "format": "json",
        "geocode": address,
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        coordinates_str = data["response"]["GeoObjectCollection"]["featureMember"][0]["GeoObject"]["Point"]["pos"]
        coordinates = tuple(map(float, coordinates_str.split()))
        return coordinates
    except Exception as e:
        print(f"Произошла ошибка при геокодировании адреса '{address}': {e}")
        return None

old_df = pd.read_excel("7. Схема подключений МОЭК.xlsx")  # Предыдущий DataFrame

# Создаем список для хранения результатов
result_data = []

for index, row in old_df.iterrows():
    address_parts = row["Адрес строения"].split(", д.")
    if len(address_parts) == 2:
        street = address_parts[0].strip()
        house_number = address_parts[1].strip()
    else:
        street = address_parts[0].strip()
        house_number = None

    city = "Москва"

    # Проверяем, что адрес не пустой
    if house_number is None and street.strip() == "":
        # Если адрес пустой, пропускаем текущую итерацию
        print("Пропускаем пустой адрес")
        continue

    # Добавляем проверку на None перед вызовом функции format_address
    address = format_address(city, street, house_number) if house_number else None

    if address:
        location = geocode_address(address)
        if location:
            result_data.append({'Адрес': address, 'Широта': location[1], 'Долгота': location[0]})
            print(f"Геокодирование успешно для адреса '{address}'")
            time.sleep(0.25)
        else:
            print(f"Координаты для адреса '{address}' не найдены.")
    else:
        print("Отсутствует адрес")

# Создаем DataFrame из списка результатов
result_df = pd.DataFrame(result_data)

# Объединяем исходный DataFrame с новыми данными
merged_df = pd.concat([old_df, result_df], axis=1)

# Сохраняем результаты в новый файл Excel
merged_df.to_excel("геокодированные_адреса_yandex1.xlsx", index=False)

print(f"Сохранено общее количество записей: {len(result_data)}")