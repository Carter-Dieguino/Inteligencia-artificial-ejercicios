import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import seaborn as sns

# Cargar y preparar datos Iris para clasificación
iris = load_iris()
X = iris.data
y = iris.target

# Dividir datos para clasificación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar clasificador KNN
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train_scaled, y_train)

# Hacer predicciones
y_pred = knn_clf.predict(X_test_scaled)

# Evaluar el clasificador
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del clasificador KNN:", accuracy)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Visualización de resultados de clasificación
plt.figure(figsize=(12, 5))

# Gráfico 1: Comparación de características
plt.subplot(1, 2, 1)
sns.scatterplot(data=pd.DataFrame(X, columns=iris.feature_names),
                x=iris.feature_names[0], 
                y=iris.feature_names[1],
                hue=iris.target_names[y],
                style=iris.target_names[y])
plt.title('Visualización de Clasificación Iris\n(Primeras dos características)')

# Gráfico 2: Matriz de confusión
plt.subplot(1, 2, 2)
sns.heatmap(pd.crosstab(y_test, y_pred), 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.tight_layout()
plt.show()

# Regresión KNN usando longitud del pétalo como objetivo
# Usaremos las otras características para predecir la longitud del pétalo
X_reg = np.delete(X, 2, axis=1)  # Eliminar longitud del pétalo
y_reg = X[:, 2]  # Usar longitud del pétalo como objetivo

# Dividir datos para regresión
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Escalar datos para regresión
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# Entrenar regressor KNN
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train_reg_scaled, y_train_reg)

# Hacer predicciones de regresión
y_pred_reg = knn_reg.predict(X_test_reg_scaled)

# Evaluar el regresor
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
print("\nError cuadrático medio (RMSE) de la regresión:", rmse)

# Visualización de resultados de regresión
plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Comparación de Valores Reales vs Predichos\nRegresión KNN')
plt.tight_layout()
plt.show()
