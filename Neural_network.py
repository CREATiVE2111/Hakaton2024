import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder

# Load data
file_path = '11.Выгрузка_ОДПУ_отопление_ВАО_20240522.xlsx'
data = pd.read_excel(file_path)

# Select relevant columns
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

# Drop rows where all specified columns are NA
data = data.dropna(how='all', subset=columns_of_interest[:-1])

# Fill missing values and scale data
numeric_features = columns_of_interest[:-1]  # All columns except 'Ошибки'
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

data_imputed = imputer.fit_transform(data[numeric_features])
data_scaled = scaler.fit_transform(data_imputed)

# Encode the target variable
label_encoder = LabelEncoder()
data['Ошибки'] = data['Ошибки'].fillna('Ошибки нет')  # Fill missing values as 'Ошибки нет'
y = label_encoder.fit_transform(data['Ошибки'])

# One-hot encoding of the target variable
onehot_encoder = OneHotEncoder(sparse_output=False)
y = y.reshape(-1, 1)
y_onehot = onehot_encoder.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data_scaled, y_onehot, test_size=0.2, random_state=42)

# Apply SMOTE with adjusted k_neighbors, skipping classes with insufficient samples
smote = SMOTE(random_state=42)

X_train_resampled = []
y_train_resampled = []

# Check the number of samples in each class
class_counts = np.sum(y_train, axis=0)
print("Class counts in training data:", class_counts)

# Resample only if a class has more than one sample
for class_idx in range(y_train.shape[1]):
    class_count = class_counts[class_idx]
    if class_count > 1:
        X_train_class = X_train[y_train[:, class_idx] == 1]
        y_train_class = y_train[y_train[:, class_idx] == 1]
        X_res, y_res = smote.fit_resample(X_train_class, y_train_class)
        X_train_resampled.append(X_res)
        y_train_resampled.append(y_res)
    else:
        X_train_resampled.append(X_train[y_train[:, class_idx] == 1])
        y_train_resampled.append(y_train[y_train[:, class_idx] == 1])

# Concatenate arrays
X_train_resampled = np.vstack(X_train_resampled)
y_train_resampled = np.vstack(y_train_resampled)

# Shuffle the resampled dataset
shuffled_indices = np.random.permutation(X_train_resampled.shape[0])
X_train_resampled = X_train_resampled[shuffled_indices]
y_train_resampled = y_train_resampled[shuffled_indices]

# Create and train the neural network model
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[X_train_resampled.shape[1]]),
    layers.Dense(64, activation='relu'),
    layers.Dense(y_onehot.shape[1], activation='softmax')
])

# Class weights for imbalance
class_weights = {i: 1.0 / np.sum(y_train_resampled == i) for i in range(y_onehot.shape[1])}

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback to print metrics after each epoch
class PrintEpochMetrics(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"Epoch {epoch + 1}, Loss: {logs['loss']}, Accuracy: {logs['accuracy']}, Val_Loss: {logs['val_loss']}, Val_Accuracy: {logs['val_accuracy']}")

# Train the model
history = model.fit(
    X_train_resampled, y_train_resampled,
    epochs=10,
    validation_split=0.2,
    verbose=0,
    class_weight=class_weights,
    callbacks=[PrintEpochMetrics()]
)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Accuracy on test set: {accuracy}')

# Save the median values used for imputing
medians = imputer.statistics_

# Function to predict error based on new data
def predict_error(values):
    input_data = np.array([medians[i] if v == '' else float(v) for i, v in enumerate(values)]).reshape(1, -1)
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)

    # Get probabilities for all classes
    prediction_probabilities = model.predict(input_data)[0]
    top_class_indices = np.argsort(prediction_probabilities)[::-1]
    possible_errors = label_encoder.inverse_transform(top_class_indices)
    probabilities = prediction_probabilities[top_class_indices]

    possible_errors_with_probabilities = list(zip(possible_errors, probabilities))

    return predicted_label[0], possible_errors_with_probabilities

# Load new dataset for prediction
new_file_path = '../нейронка мб/merged_file.xlsx'
new_data = pd.read_excel(new_file_path)

# Add predictions to new dataframe
predictions = []
for _, row in new_data.iterrows():
    values = row[columns_of_interest[:-1]].tolist()  # Exclude 'Ошибки' column
    predicted_label, _ = predict_error(values)
    predictions.append(predicted_label)

new_data['Predicted Error'] = predictions

# Save updated dataframe to new Excel file
output_file_path = '../нейронка мб/new_dataset_with_predictions1.xlsx'
new_data.to_excel(output_file_path, index=False)
print(f"Updated file saved to {output_file_path}")
