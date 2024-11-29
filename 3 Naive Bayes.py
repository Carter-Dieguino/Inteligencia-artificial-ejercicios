import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Cargar el dataset Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Crear un DataFrame para mejor manipulación
df = pd.DataFrame(X, columns=feature_names)
df['species'] = pd.Categorical.from_codes(y, target_names)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar los datoss
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train_scaled, y_train)

# Realizar predicciones
y_pred = nb_classifier.predict(X_test_scaled)

# Evaluar el modelo
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Crear matriz de confusión
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.title('Matriz de Confusión')
plt.ylabel('Verdaderos')
plt.xlabel('Predichos')
plt.show()

# Visualizar la separación de clases usando las dos características más importantes
plt.figure(figsize=(12, 5))

# Primer plot: longitud y ancho del sépalo
plt.subplot(1, 2, 1)
scatter = plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'],
                     c=y, cmap='viridis')
plt.xlabel('Longitud del Sépalo (cm)')
plt.ylabel('Ancho del Sépalo (cm)')
plt.title('Separación por Características del Sépalo')
# Corrección de la leyenda
plt.legend(scatter.legend_elements()[0], target_names)

# Segundo plot: longitud y ancho del pétalo
plt.subplot(1, 2, 2)
scatter = plt.scatter(df['petal length (cm)'], df['petal width (cm)'],
                     c=y, cmap='viridis')
plt.xlabel('Longitud del Pétalo (cm)')
plt.ylabel('Ancho del Pétalo (cm)')
plt.title('Separación por Características del Pétalo')
# Corrección de la leyenda
plt.legend(scatter.legend_elements()[0], target_names)

plt.tight_layout()
plt.show()
