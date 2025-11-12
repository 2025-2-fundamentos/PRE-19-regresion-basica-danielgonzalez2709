"""Script para entrenar el modelo MLP y crear los archivos pickle necesarios."""

import pickle
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Cargar los datos
print("Cargando datos...")
dataset = pd.read_csv("files/input/auto_mpg.csv")

# Eliminar filas con valores faltantes
dataset = dataset.dropna()

# Convertir Origin a categorías
dataset["Origin"] = dataset["Origin"].map(
    {1: "USA", 2: "Europe", 3: "Japan"},
)

# Crear variables dummy para Origin
dataset = pd.get_dummies(dataset, columns=["Origin"], prefix="", prefix_sep="")

# Separar características y variable objetivo
y = dataset.pop("MPG")
X = dataset

print(f"Datos cargados: {X.shape[0]} filas, {X.shape[1]} características")
print(f"Características: {list(X.columns)}")

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear y entrenar el scaler
print("\nCreando scaler...")
features_scaler = StandardScaler()
X_train_scaled = features_scaler.fit_transform(X_train)
X_test_scaled = features_scaler.transform(X_test)

# Crear y entrenar el modelo MLP
print("Entrenando modelo MLP...")
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),  # Dos capas ocultas
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=True
)

mlp.fit(X_train_scaled, y_train)

# Evaluar el modelo
print("\nEvaluando modelo...")
y_pred_train = mlp.predict(X_train_scaled)
y_pred_test = mlp.predict(X_test_scaled)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f"MSE en entrenamiento: {mse_train:.4f}")
print(f"MSE en prueba: {mse_test:.4f}")

# Evaluar en todo el dataset (como lo hace el test)
X_all_scaled = features_scaler.transform(X)
y_pred_all = mlp.predict(X_all_scaled)
mse_all = mean_squared_error(y, y_pred_all)
print(f"MSE en dataset completo: {mse_all:.4f}")

# Guardar el modelo y el scaler
print("\nGuardando archivos pickle...")
with open("mlp.pickle", "wb") as file:
    pickle.dump(mlp, file)

with open("features_scaler.pickle", "wb") as file:
    pickle.dump(features_scaler, file)

print("\n¡Listo! Archivos guardados:")
print("- mlp.pickle")
print("- features_scaler.pickle")
print(f"\nMSE final: {mse_all:.4f} (debe ser < 7.745)")

if mse_all < 7.745:
    print("✓ El modelo cumple con el requisito!")
else:
    print("✗ El modelo NO cumple con el requisito. Intenta ajustar los hiperparámetros.")

