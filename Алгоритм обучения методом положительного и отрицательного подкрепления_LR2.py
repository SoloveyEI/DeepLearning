import numpy as np
import tensorflow as tf

def step_function(z, threshold):
    return 1 if z >= threshold else 0
    # Активирует нейрон, возвращая 1, если входное значение больше или равно нулю, иначе 0.

# Класс персептрона
class Perceptron:
    def __init__(self, input_dim, threshold=0):
        self.input_dim = input_dim
        self.threshold = threshold
        self.weights = np.random.randn(input_dim) * 0.01  # Малые случайные веса

    def predict(self, X):
        weighted_sum = np.dot(X, self.weights)
        return step_function(weighted_sum, self.threshold)

    def learn(self, X, t):
        y_hat = self.predict(X)
        if y_hat != t:
            # Положительное подкрепление: увеличение весов
            if y_hat == 0 and t == 1:
                self.weights += X
            # Отрицательное подкрепление: уменьшение весов
            elif y_hat == 1 and t == 0:
                self.weights -= X

    def evaluate(self, X, y):
        correct_predictions = 0
        for i in range(len(X)):
            pred = self.predict(X[i])
            if pred == y[i]:
                correct_predictions += 1
        return correct_predictions / len(X)

# 1. Подготовка данных
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Фильтруем только классы 0 и 1
idx_train = np.where((y_train == 0) | (y_train == 1))[0]
x_train = x_train[idx_train]
y_train = y_train[idx_train]

# Аналогично для теста
idx_test = np.where((y_test == 0) | (y_test == 1))[0]
x_test = x_test[idx_test]
y_test = y_test[idx_test]

# Нормализация данных (приведение пикселей к диапазону [0, 1])
x_train = x_train / 255.0
x_test = x_test / 255.0

# Преобразование изображений в векторы
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Добавляем смещение (bias) в виде дополнительной единицы
x_train = np.hstack([np.ones((len(x_train), 1)), x_train])
x_test = np.hstack([np.ones((len(x_test), 1)), x_test])

# 2. Инициализация персептрона
input_dim = x_train.shape[1]  # Размерность входных данных (785, включая смещение)
perceptron = Perceptron(input_dim=input_dim)

# 3. Параметры обучения
epochs = 100 # Количество эпох обучения
history = {'errors': [], 'accuracy': []}

# 4. Обучение методом положительного и отрицательного подкрепления
for epoch in range(epochs):
    errors = 0
    for i in range(x_train.shape[0]):
        X = x_train[i]
        t = y_train[i]
        perceptron.learn(X, t)
        y_hat = perceptron.predict(X)
        if y_hat != t:
            errors += 1

    history['errors'].append(errors)
    accuracy = perceptron.evaluate(x_train, y_train)
    history['accuracy'].append(accuracy)
    print(f'Эпоха {epoch+1}, Количество ошибок: {errors}, Точность: {accuracy:.5f}')

    # Шаг 5. Проверка завершения обучения
    if errors == 0:
        print("Обучение завершено, ошибок нет!")
        break

#5. Оценка точности на тестовых данных
test_accuracy = perceptron.evaluate(x_test, y_test)