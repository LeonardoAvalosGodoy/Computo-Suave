import numpy as np


def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    return (x > 0).astype(float)


def lineal(x):
    return x


def d_lineal(x):
    return np.ones_like(x)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def d_mse(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


class Capa:
    def __init__(self, entradas, neuronas, activacion, d_activacion):
        self.pesos = np.random.rand(entradas, neuronas)
        self.activacion = activacion
        self.d_activacion = d_activacion

        self.entrada = None
        self.N = None
        self.salida = None
        self.delta = None

    def forward(self, X):
        self.entrada = X
        self.N = X @ self.pesos
        self.salida = self.activacion(self.N)

        return self.salida

    def backward(self, lr):
        gradiente = self.entrada.T @ self.delta
        self.pesos -= lr * gradiente


class RedNeuronal:
    def __init__(self, capas, lr=0.001):
        self.capas = capas
        self.lr = lr

    def predict(self, X):
        salida = X

        for capa in self.capas:
            salida = capa.forward(salida)

        return salida

    def entrenar(self, X, y, epocas):
        for i in range(epocas):
            y_pred = self.predict(X)
            error = mse(y, y_pred)

            d_error = d_mse(y, y_pred)

            ultima = self.capas[-1]
            ultima.delta = d_error * ultima.d_activacion(ultima.N)

            for j in range(len(self.capas) - 2, -1, -1):
                actual = self.capas[j]
                siguiente = self.capas[j + 1]

                error_anterior = siguiente.delta @ siguiente.pesos.T

                actual.delta = (
                    error_anterior *
                    actual.d_activacion(actual.N)
                )

            for capa in self.capas:
                capa.backward(self.lr)

            if i % 10 == 0:
                print(f"Epoca: {i} | MSE: {error:.6f}")


X = np.ones((3, 2)) * 5
y = np.zeros((3, 2))


capa1 = Capa(2, 7, relu, d_relu)
capa2 = Capa(7, 7, relu, d_relu)
capa3 = Capa(7, 2, lineal,d_lineal)


red = RedNeuronal(
    [capa1, capa2, capa3],
    lr=0.001
)


prediccion_inicial = red.predict(X)
error_inicial = mse(y, prediccion_inicial)

print("Prediccion inicial:")
print(prediccion_inicial)

print("\nMSE inicial:")
print(error_inicial)


print("\nEntrenando")

red.entrenar(X, y, 200)


prediccion_final = red.predict(X)
error_final = mse(y, prediccion_final)


print("\nPrediccion final:")
print(prediccion_final)

print("\nMSE final:")
print(error_final)


print("\nComparacion:")
print(f"MSE inicial: {error_inicial:.6f}")
print(f"MSE final:   {error_final:.6f}")
