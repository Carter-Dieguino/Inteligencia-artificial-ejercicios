import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generar datos sintéticos
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = 0.5 * X.ravel() + np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Valores de epsilon a probar
epsilons = [0.01, 0.1, 0.5, 1.0]
colors = ['blue', 'green', 'red', 'purple']

# Crear figura para múltiples gráficos
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
axes = [ax1, ax2, ax3, ax4]

# Puntos para la curva de predicción
X_plot = np.linspace(0, 5, 100).reshape(-1, 1)

# Almacenar métricas
metrics = []

# Crear gráficos para cada valor de epsilon
for eps, color, ax in zip(epsilons, colors, axes):
    # Entrenar modelo
    svr = SVR(kernel='rbf', C=100, epsilon=eps)
    svr.fit(X_train, y_train)
    
    # Predicciones
    y_pred = svr.predict(X_test)
    y_plot = svr.predict(X_plot)
    
    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics.append({'epsilon': eps, 'MSE': mse, 'R2': r2})
    
    # Graficar
    ax.scatter(X_train, y_train, color='gray', alpha=0.5, label='Datos de entrenamiento')
    ax.scatter(X_test, y_test, color='black', alpha=0.5, label='Datos de prueba')
    ax.plot(X_plot, y_plot, color=color, label=f'SVR (ε={eps})')
    
    # Visualizar el tubo epsilon
    ax.fill_between(X_plot.ravel(), 
                   y_plot - eps, 
                   y_plot + eps, 
                   alpha=0.2, 
                   color=color)
    
    ax.set_title(f'SVR con ε={eps}\nMSE={mse:.3f}, R²={r2:.3f}')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# Imprimir tabla de métricas
print("\nMétricas para diferentes valores de epsilon:")
print("----------------------------------------")
print("Epsilon  |    MSE    |    R²")
print("----------------------------------------")
for metric in metrics:
    print(f"{metric['epsilon']:.2f}     |  {metric['MSE']:.4f}  |  {metric['R2']:.4f}")
