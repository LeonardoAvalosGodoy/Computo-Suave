import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, precision_score, roc_auc_score, recall_score, confusion_matrix

def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    return (x > 0).astype(float)


def lineal(x):
    return x


def d_lineal(x):
    return np.ones_like(x)


def sigmoide(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def mse(y, pred):
    return np.mean((y - pred) ** 2)


def d_mse(y, pred):
    return 2 * (pred - y) / y.size


class Capa:
    def __init__(self, entradas, neuronas, activacion, d_activacion, bias=False, mejorada=False):
        if mejorada:
            factor = 2 if activacion is relu else 1
            self.pesos = np.random.randn(entradas, neuronas) * np.sqrt(factor / entradas)
        else:
            self.pesos = np.random.rand(entradas, neuronas)
        self.bias = np.zeros((1, neuronas))
        self.usar_bias = bias
        self.activacion = activacion
        self.d_activacion = d_activacion

    def forward(self, X):
        self.entrada = X
        self.N = X @ self.pesos
        if self.usar_bias:
            self.N += self.bias
        self.salida = self.activacion(self.N)
        return self.salida

    def backward(self, lr):
        self.pesos -= lr * (self.entrada.T @ self.delta)
        if self.usar_bias:
            self.bias -= lr * np.sum(self.delta, axis=0, keepdims=True)


class RedNeuronal:
    def __init__(self, capas, lr=0.001):
        self.capas = capas
        self.lr = lr

    def predict(self, X):
        for capa in self.capas:
            X = capa.forward(X)
        return X

    def entrenar(self, X, y, epocas=200, pesos_clase=None):
        for i in range(epocas):
            pred = self.predict(X)
            ultima = self.capas[-1]

            if pesos_clase is None:
                error = mse(y, pred)
                ultima.delta = d_mse(y, pred) * ultima.d_activacion(ultima.N)
            else:
                p = np.clip(pred, 1e-7, 1 - 1e-7)
                error = -np.mean(pesos_clase * (y * np.log(p) + (1 - y) * np.log(1 - p)))
                ultima.delta = pesos_clase * (pred - y) / y.size

            for j in range(len(self.capas) - 2, -1, -1):
                actual = self.capas[j]
                siguiente = self.capas[j + 1]
                actual.delta = (siguiente.delta @ siguiente.pesos.T) * actual.d_activacion(actual.N)

            for capa in self.capas:
                capa.backward(self.lr)

            if i % 50 == 0:
                nombre = "MSE" if pesos_clase is None else "BCE"
                print(f"Epoca: {i} | {nombre}: {error:.6f}")
    

def crear_red(entradas, clasificacion=False, mejorar=False):
    np.random.seed(42)
    ajuste = clasificacion or mejorar
    capa1 = Capa(entradas, 7, relu, d_relu, bias=ajuste, mejorada=ajuste)
    if clasificacion:
        capa2 = Capa(7, 1, sigmoide, None, bias=True, mejorada=True)
    else:
        capa2 = Capa(7, 1, lineal, d_lineal, bias=mejorar, mejorada=mejorar)
    return RedNeuronal([capa1, capa2], lr=0.05 if ajuste else 0.001)

def ejecutar_regresion(nombre, X, y, fechas, unidad, mejorar=False):
    X_train, X_test, y_train, y_test, _, fechas_test = train_test_split(
        X, y, fechas, train_size=0.6, test_size=0.4, shuffle=False
    )

    print(f"\n{nombre}\n")
    print("\nTotal de registros:", len(X))
    print("\nEntradas X:")
    print(X.shape)
    print("\nSalidas y:")
    print(y.shape)
    print("\nEntrenamiento:", len(X_train))
    print("Prueba:", len(X_test))

    escala_X = StandardScaler()
    X_train = escala_X.fit_transform(X_train)
    X_test = escala_X.transform(X_test)

    escala_y = StandardScaler()
    y_train_norm = escala_y.fit_transform(y_train)
    y_test_norm = escala_y.transform(y_test)

    red = crear_red(X.shape[1], mejorar=mejorar)
    inicial_pred = red.predict(X_train)
    inicial = mse(y_train_norm, inicial_pred)

    print("\nPrediccion inicial:")
    print(inicial_pred[:5])
    print("\nMSE inicial normalizado:")
    print(inicial)

    print("\nEntrenando")
    red.entrenar(X_train, y_train_norm, epocas=600 if mejorar else 200)

    final = mse(y_train_norm, red.predict(X_train))
    pred_norm = red.predict(X_test)
    error_test_norm = mse(y_test_norm, pred_norm)
    pred = escala_y.inverse_transform(pred_norm)
    error_test_real = mse(y_test, pred)
    R2 = r2_score(y_test, pred)
    R = np.corrcoef(y_test.ravel(), pred.ravel())[0, 1]

    print("\nRESULTADOS FINALES")
    print(f"\nMSE inicial normalizado: {inicial:.6f}")
    print(f"MSE final entrenamiento: {final:.6f}")
    print(f"MSE prueba normalizado:  {error_test_norm:.6f}")
    print(f"MSE prueba en {unidad}: {error_test_real:.6f}")
    print(f"\nR2: {R2:.6f}")
    print(f"R:  {R:.6f}")

    print("\nCOMPARACION DE RESULTADOS")
    for i in range(min(10, len(y_test))):
        fecha = str(fechas_test[i]).replace("T", " ")[:16]
        print(f"\nRegistro {i + 1} - {fecha}")
        print(f"Real: {y_test[i, 0]:.4f}    Prediccion: {pred[i, 0]:.4f}")


def ejecutar_sismos():
    datos = pd.read_csv(CARPETA / "sismos.csv")
    cols = ["latitude", "longitude", "depth", "magnitude"]
    for c in cols:
        datos[c] = pd.to_numeric(datos[c], errors="coerce")
    datos = datos.dropna(subset=cols)

    X = datos[["latitude", "longitude", "depth"]].to_numpy(float)
    y = (datos["magnitude"] >= 5).astype(int).to_numpy().reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.6, test_size=0.4, random_state=42, stratify=y.ravel()
    )
    escala = StandardScaler()
    X_train = escala.fit_transform(X_train)
    X_test = escala.transform(X_test)

    n0, n1 = np.sum(y_train == 0), np.sum(y_train == 1)
    pesos = np.where(y_train == 1, len(y_train) / (2 * n1), len(y_train) / (2 * n0))
    red = crear_red(3, clasificacion=True)

    print(f"\nSISMOS DE MEXICO\n")
    print("\nTotal de registros:", len(X))
    print("\nEntradas X:")
    print(X.shape)
    print("\nSalidas y:")
    print(y.shape)
    print("\nEntrenamiento:", len(X_train))
    print("Prueba:", len(X_test))
    print(f"\nClase 0: {int(n0)} | Clase 1: {int(n1)} (entrenamiento)")

    print("\nEntrenando")
    red.entrenar(X_train, y_train, epocas=500, pesos_clase=pesos)

    prob_train = np.clip(red.predict(X_train), 1e-7, 1 - 1e-7)
    bce_final = -np.mean(pesos * (y_train * np.log(prob_train) + (1 - y_train) * np.log(1 - prob_train)))
    prob = red.predict(X_test).ravel()
    pred = (prob >= 0.5).astype(int)
    reales = y_test.ravel()

    print("\nRESULTADOS FINALES")
    print(f"\nBCE final entrenamiento: {bce_final:.6f}")
    print(f"Precision: {precision_score(reales, pred, zero_division=0):.6f}")
    print(f"Exactitud: {accuracy_score(reales, pred):.6f}")
    print(f"ROC-AUC: {roc_auc_score(reales, prob):.6f}")
    print(f"Recall clase 1: {recall_score(reales, pred, zero_division=0):.6f}")
    print("\nMatriz de confusion:")
    print(confusion_matrix(reales, pred, labels=[0, 1]))
    print(f"Exactitud de predecir siempre 0: {np.mean(reales == 0):.6f}")

    print("\nCOMPARACION DE RESULTADOS")
    for i in range(min(10, len(reales))):
        print(f"\nRegistro {i + 1}")
        print(f"Real: {reales[i]}    Prediccion: {pred[i]}    Probabilidad: {prob[i]:.4f}")


def ejecutar_temperaturas():
    datos = pd.read_csv(CARPETA / "temp_mx.csv")
    cols = ["Temperature(F)", "Dew Point(F)", "Humidity(%)", "Wind Speed(mph)", "Pressure(in)"]
    datos["DateTime"] = pd.to_datetime(datos["DateTime"], errors="coerce")
    for c in cols:
        datos[c] = pd.to_numeric(datos[c], errors="coerce")
    datos = datos.sort_values("DateTime").reset_index(drop=True)
    datos["siguiente"] = datos["Temperature(F)"].shift(-1)
    datos["fecha_siguiente"] = datos["DateTime"].shift(-1)
    intervalo = datos["fecha_siguiente"] - datos["DateTime"]
    datos = datos[intervalo == pd.Timedelta(hours=1)].dropna(subset=cols + ["siguiente"])

    X = datos[cols].to_numpy(float)
    y = datos[["siguiente"]].to_numpy(float)
    ejecutar_regresion("TEMPERATURAS DE CDMX", X, y, datos["fecha_siguiente"].to_numpy(), "Fahrenheit", mejorar=True)


def ejecutar_cambio():
    datos = pd.read_csv(CARPETA / "cambio.csv")
    datos["Unnamed: 0"] = pd.to_datetime(datos["Unnamed: 0"], errors="coerce")
    datos["mexican_peso"] = pd.to_numeric(datos["mexican_peso"], errors="coerce")
    datos = datos.dropna(subset=["Unnamed: 0", "mexican_peso"])
    datos = datos.sort_values("Unnamed: 0").reset_index(drop=True)
    valores = datos["mexican_peso"].to_numpy(float)
    X = np.array([valores[i-3:i] for i in range(3, len(valores))])
    y = valores[3:].reshape(-1, 1)
    fechas = datos["Unnamed: 0"].to_numpy()[3:]
    ejecutar_regresion("TIPO DE CAMBIO MXN/USD", X, y, fechas, "pesos")


CARPETA = Path(__file__).resolve().parent

if __name__ == "__main__":
    ejecutar_sismos()
    ejecutar_temperaturas()
    ejecutar_cambio()
