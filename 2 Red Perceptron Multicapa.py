import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.datasets import load_breast_cancer, fetch_california_housing
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# 1. REGRESIÓN CON DATOS DE DIABETES
def modelo_regresion():
    print("1. Modelo de Regresión - Dataset de Diabetes")
    print("-" * 50)
    
    # Cargamos el dataset de diabetes (viene incluido en scikit-learn)
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    
    # Nombres de las características
    feature_names = diabetes.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizacióón de datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    modelo = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        max_iter=2000,          
        random_state=42,
        verbose=True,
        learning_rate_init=0.001,  
        early_stopping=True,     
        validation_fraction=0.1,  
        n_iter_no_change=50      
    )
    
    print("Entrenando modelo de regresión...")
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Error Cuadrático Medio: {mse:.4f}")
    
    # Visualización de resultados
    plt.figure(figsize=(12, 6))
    
    # Gráfico de predicciones vs valores reales
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title('Predicciones vs Valores Reales')
    
    # Importancia de características
    plt.subplot(1, 2, 2)
    importances = np.abs(modelo.coefs_[0]).mean(axis=1)
    plt.bar(range(len(feature_names)), importances)
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.title('Importancia de Características')
    
    plt.tight_layout()
    plt.show()

# 2. CLASIFICACIÓN CON DATOS DE CÁNCER DE MAMA
def modelo_clasificacion_cancer():
    print("\n2. Modelo de Clasificación - Cáncer de Mama")
    print("-" * 50)
    
    # Cargamos el dataset
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    modelo = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=2000,
        random_state=42,
        verbose=True,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50
    )
    
    print("Entrenando modelo de clasificación...")
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión: {accuracy:.4f}")
    
    # Visualización de resultados
    plt.figure(figsize=(15, 5))
    
    # Matriz de confusión
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    
    # Características más importantes
    plt.subplot(1, 2, 2)
    importances = np.abs(modelo.coefs_[0]).mean(axis=1)
    top_features = np.argsort(importances)[-10:]  # Top 10 características
    plt.barh(range(10), importances[top_features])
    plt.yticks(range(10), cancer.feature_names[top_features])
    plt.title('Top 10 Características más Importantes')
    
    plt.tight_layout()
    plt.show()

# 3. PREDICCIÓN DE BITCOIN
def modelo_bitcoin_real():
    print("\n3. Predicción de Bitcoin - Datos de la Última Semana")
    print("-" * 50)
    
    # Descargamos datos de la última semana
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    print("Descargando datos de Bitcoin...")
    btc = yf.download('BTC-USD', start=start_date, end=end_date, interval='1h')
    
    if len(btc) == 0:
        print("Error: No se pudieron descargar datos de Bitcoin")
        return
    
    # Preparamos los datos
    btc['Returns'] = btc['Close'].pct_change()
    btc['Volatility'] = btc['Returns'].rolling(window=24).std()
    btc['MA24'] = btc['Close'].rolling(window=24).mean()
    btc['MA7'] = btc['Close'].rolling(window=7).mean()
    btc = btc.dropna()
    
    # Características para el modelo
    features = ['Returns', 'Volatility', 'MA24', 'MA7']
    X = btc[features].values
    y = btc['Close'].values
    
    if len(X) < 25:  # Necesitamos al menos 25 puntos de datos
        print("Error: Datos insuficientes")
        return
    
    # División temporal (últimas 24 horas para test)
    train_size = len(X) - 24
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    modelo = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        max_iter=2000,
        random_state=42,
        verbose=True,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50
    )
    
    print("Entrenando modelo de predicción de Bitcoin...")
    modelo.fit(X_train_scaled, y_train_scaled)
    y_pred_scaled = modelo.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    
    # Visualización de resultados
    plt.figure(figsize=(15, 10))
    
    # Gráfico de precios y predicciones
    plt.subplot(2, 1, 1)
    plt.plot(btc.index[-24:], y_test, label='Real', alpha=0.7)
    plt.plot(btc.index[-24:], y_pred, label='Predicción', alpha=0.7)
    plt.title('Predicción de Precios de Bitcoin - Últimas 24 horas')
    plt.xlabel('Fecha')
    plt.ylabel('Precio USD')
    plt.legend()
    
    # Gráfico de error
    plt.subplot(2, 1, 2)
    error = np.abs(y_test - y_pred.ravel())
    plt.plot(btc.index[-24:], error)
    plt.title('Error Absoluto en la Predicción')
    plt.xlabel('Fecha')
    plt.ylabel('Error USD')
    
    plt.tight_layout()
    plt.show()
    
    # Métricas
    mse = mean_squared_error(y_test, y_pred)
    print(f"Error Cuadrático Medio: {mse:.4f}")
    print(f"Error Promedio: ${np.mean(error):.2f}")
    
    # Predicción para la próxima hora
    ultima_data = scaler_X.transform(X[-1:])
    proxima_prediccion = scaler_y.inverse_transform(modelo.predict(ultima_data).reshape(-1, 1))[0][0]
    print(f"\nPredicción para la próxima hora: ${proxima_prediccion:.2f}")

if __name__ == "__main__":
    print("Iniciando análisis de datos...")
    try:
        modelo_regresion()
        modelo_clasificacion_cancer()
        modelo_bitcoin_real()
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
