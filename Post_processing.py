import pandas as pd

# Первый скрипт для чистки данных и сохранения во временный файл
def clean_and_save():
    # Загрузка файлов
    file_path1 = 'геокодированные_адреса_yandex.xlsx'
    file_path2 = '13. Адресный реестр объектов недвижимости города Москвы.xlsx'

    # Загрузка данных из файлов
    df1 = pd.read_excel(file_path1)
    df2 = pd.read_excel(file_path2)

    # Функция для преобразования адресов
    def transform_address(address):
        replacements = {
            'ул.':'улица',
            'бульв.': 'бульвар',
            'д.': 'дом ',
            'стр.': 'строение '
        }
        for key, value in replacements.items():
            address = address.replace(key, value)
        return address

    # Преобразование адресов в первом датафрейме
    df1['Преобразованный адрес'] = df1['Адрес ТП'].apply(transform_address)

    # Установка индекса для второго датафрейма по столбцу ADDRESS
    df2.set_index('ADDRESS', inplace=True)

    # Колонка в первом файле для поиска и колонки для извлечения из второго файла
    search_column = 'Преобразованный адрес'
    columns_to_extract = ['global_id', 'UNOM', 'geoData', 'geodata_center']

    # Функция для поиска и извлечения данных из второго датафрейма
    def find_and_extract(row, df2, columns_to_extract):
        try:
            result = df2.loc[df2.index.str.contains(row[search_column], na=False), columns_to_extract].iloc[0]
            return result.values
        except IndexError:
            return [None] * len(columns_to_extract)

    # Применение функции для каждого ряда первого датафрейма
    df1[columns_to_extract] = df1.apply(lambda row: find_and_extract(row, df2, columns_to_extract), axis=1, result_type="expand")

    # Удаление строк, в которых нет значения в колонке geoData
    df1 = df1.dropna(subset=['geoData'])

    # Сохранение результата во временный Excel файл
    temp_output_file_path = 'temp_cleaned_file.xlsx'
    df1.to_excel(temp_output_file_path, index=False)

    # Печать успешно добавленных записей
    successfully_added = df1[df1[columns_to_extract].notna().all(axis=1)]
    for index, row in successfully_added.iterrows():
        print(f"Запись успешно добавлена для адреса {row['Преобразованный адрес']}: global_id={row['global_id']}, UNOM={row['UNOM']}")

    print("Чистка данных завершена. Результат сохранен в файл:", temp_output_file_path)

# Второй скрипт для объединения данных и удаления строк с отсутствующими данными в столбце "Потребители"
def merge_data_and_clean():
    # Загрузка временного файла с очищенными данными
    temp_cleaned_file = 'temp_cleaned_file.xlsx'
    df = pd.read_excel(temp_cleaned_file)

    # Загрузка файла, где нужно проверить столбец "Потребители"
    file2 = '11.Выгрузка_ОДПУ_отопление_ВАО_20240522.xlsx'
    df2 = pd.read_excel(file2)

    # Проверка наличия столбца "Потребители" в df
    if 'Потребители' not in df.columns:
        raise ValueError('Столбец "Потребители" отсутствует в первом файле')

    # Фильтрация строк, где столбец "Потребители" пустой
    df = df.dropna(subset=['Потребители'])

    # Проверка наличия столбца "Потребители" в df2
    if 'Потребители' not in df2.columns:
        raise ValueError('Столбец "Потребители" отсутствует во втором файле')

    # Объединение данных по колонке UNOM с сохранением всех строк из первого файла
    merged_df = df.merge(df2.drop_duplicates(subset=['UNOM']), on='UNOM', how='left')

    # Сохранение объединенного результата в конечный Excel файл
    final_output_file = 'merged_file.xlsx'
    merged_df.to_excel(final_output_file, index=False)

    print(f'Объединённый файл сохранён как {final_output_file}')

# Вызов функций для выполнения скриптов
clean_and_save()
merge_data_and_clean()
