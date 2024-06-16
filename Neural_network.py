import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder

# Загрузка данных
file_path = '11.Выгрузка_ОДПУ_отопление_ВАО_20240522 — копия.xlsx'
data = pd.read_excel(file_path)

# Оставим только необходимые столбцы
columns_of_interest = [
    'Объём поданого теплоносителя в систему ЦО',
    'Объём обратного теплоносителя из системы ЦО',
    'Разница между подачей и обраткой(Подмес)',
    'Разница между подачей и обраткой(Утечка)',
    'Температура подачи',
    'Температура обратки',
    'Наработка часов счётчика',
    'Расход тепловой энергии ',
    'Ошибки'
]

data = data[columns_of_interest]
print(f"Data before dropping completely NA rows: {data.shape}")

# Удаление строк, где все указанные столбцы полностью пустые
data = data.dropna(how='all', subset=columns_of_interest[:-1])
print(f"Data after dropping completely NA rows: {data.shape}")

# Заполнение пропущенных значений и масштабирование
numeric_features = columns_of_interest[:-1]  # Все столбцы, кроме "Ошибки"
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

# Обучение SimpleImputer и StandardScaler
data_imputed = imputer.fit_transform(data[numeric_features])
data_scaled = scaler.fit_transform(data_imputed)

# Преобразование категориальных ошибок в числовые значения
label_encoder = LabelEncoder()
data['Ошибки'] = data['Ошибки'].fillna('Ошибки нет')  # Заполнение пустых значений как 'Ошибки нет'
y = label_encoder.fit_transform(data['Ошибки'])

# One-hot кодирование целевой переменной
onehot_encoder = OneHotEncoder(sparse_output=False)
y = y.reshape(-1, 1)
y_onehot = onehot_encoder.fit_transform(y)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(data_scaled, y_onehot, test_size=0.2, random_state=42)

# Создание и обучение модели нейронной сети
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dense(64, activation='relu'),
    layers.Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Callback-функция для отображения точности на каждой эпохе
class PrintEpochMetrics(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"Epoch {epoch + 1}, Loss: {logs['loss']}, Accuracy: {logs['accuracy']}, Val_Loss: {logs['val_loss']}, Val_Accuracy: {logs['val_accuracy']}")


# Обучение модели
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_split=0.2,
    verbose=0,
    callbacks=[PrintEpochMetrics()]
)

# Оценка модели на тестовой выборке
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Accuracy on test set: {accuracy}')

# Получение медианных значений для заполнения пропусков
medians = imputer.statistics_


# Функция для предсказания ошибки по новым значениям
def predict_error(values):
    # Заменим пропущенные значения на медианные
    input_data = np.array([medians[i] if v == '' else float(v) for i, v in enumerate(values)]).reshape(1, -1)
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)

    # Получение вероятностей для всех классов
    prediction_probabilities = model.predict(input_data)[0]
    top_class_indices = np.argsort(prediction_probabilities)[::-1]
    possible_errors = label_encoder.inverse_transform(top_class_indices)
    probabilities = prediction_probabilities[top_class_indices]

    possible_errors_with_probabilities = list(zip(possible_errors, probabilities))

    return predicted_label[0], possible_errors_with_probabilities


# Пример использования функции предсказания
new_values = [1.0, 1.0, 0.0, 0.0, '', 40.0, 1000.0, 5.0]  # Замените на свои значения
predicted_label, possible_errors_with_probabilities = predict_error(new_values)
print(f"Predicted Error: {predicted_label}")
print("Possible Errors and their probabilities:")
for error, probability in possible_errors_with_probabilities:
    print(f"{error}: {probability:.4f}")

# Пример использования функции предсказания
new_values = [1.0, 1.0, 0.0, 0.0, '', 40.0, 1000.0, 5.0]  # Замените на свои значения
predicted_label, possible_errors_with_probabilities = predict_error(new_values)
print(f"Predicted Error: {predicted_label}")
print("Possible Errors and their probabilities:")
for error, probability in possible_errors_with_probabilities:
    print(f"{error}: {probability:.4f}")

# Пример использования функции предсказания
new_values = [30.054199, 30.339844,0.285645, '', 97.1231, 46.363003, 24,1.527237]  # Замените на свои значения
predicted_label, possible_errors_with_probabilities = predict_error(new_values)
print(f"Predicted Error: {predicted_label}")
print("Possible Errors and their probabilities:")
for error, probability in possible_errors_with_probabilities:
    print(f"{error}: {probability:.4f}")
new_values = [0, 0, "", "", 20.307106, 1.22893, 0, 0]  # Замените на свои значения
predicted_label, possible_errors_with_probabilities = predict_error(new_values)
print(f"Predicted Error: {predicted_label}")
print("Possible Errors and their probabilities:")
for error, probability in possible_errors_with_probabilities:
    print(f"{error}: {probability:.4f}")
new_values = [1.0, 1.0, 0.0, 0.0, 50.0, 40.0, 1000.0, 5.0]  # Замените на свои значения
predicted_label, possible_errors_with_probabilities = predict_error(new_values)
print(f"Predicted Error: {predicted_label}")
print("Possible Errors and their probabilities:")
for error, probability in possible_errors_with_probabilities:
    print(f"{error}: {probability:.4f}")
new_values = [1.0, 1.0, 0.0, 0.0, 50.0, 40.0, 1000.0, 5.0]  # Замените на свои значения
print(predict_error(new_values))
