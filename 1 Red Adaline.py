import numpy as np
import matplotlib.pyplot as plt

class AdalineNetwork:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors_history = []
        
    def fit(self, X, y):
        # Inicialización de pesos y bias
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = np.random.randn() * 0.01
        
        # Historial de errores
        self.errors_history = []
        
        # Entrenamiento
        for _ in range(self.epochs):
            errors_epoch = []
            for i in range(len(X)):
                # Forward pass
                x_i = X[i] if len(X.shape) > 1 else np.array([X[i]])
                net_input = np.dot(x_i, self.weights) + self.bias
                
                # Calcular error
                error = y[i] - net_input
                
                # Actualizar pesos y bias
                self.weights += self.learning_rate * error * x_i
                self.bias += self.learning_rate * error
                
                errors_epoch.append(error**2)
            
            self.errors_history.append(np.mean(errors_epoch))
    
    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return np.dot(X, self.weights) + self.bias

# Función para normalizar datos
def normalize_data(X):
    return (X - np.mean(X)) / np.std(X)

# 1. Regresión Celsius a Fahrenheit
def celsius_to_fahrenheit():
    # Datos de entrenamiento
    celsius = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    fahrenheit = np.array([33.8, 35.6, 37.4, 39.2, 41, 42.8, 44.6, 46.4, 48.2, 50])
    
    # Normalizar datos de entrada
    celsius_norm = normalize_data(celsius)
    
    # Crearr y entrenar el modelo
    model = AdalineNetwork(learning_rate=0.01, epochs=500)
    model.fit(celsius_norm, fahrenheit)
    
    # Graficar resultados
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(celsius, fahrenheit, 'bo', label='Datos reales')
    plt.plot(celsius, model.predict(celsius_norm), 'r-', label='Predicción')
    plt.xlabel('Celsius')
    plt.ylabel('Fahrenheit')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(model.errors_history)
    plt.xlabel('Época')
    plt.ylabel('Error cuadrático medio')
    plt.title('Convergencia del error')
    plt.show()
    
    return model

# 2. Compuerta OR
def train_or_gate():
    # Datos de entrenamiento
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    
    model = AdalineNetwork(learning_rate=0.1, epochs=100)
    model.fit(X, y)
    
    # Graficar resultados
    plt.plot(model.errors_history)
    plt.xlabel('Época')
    plt.ylabel('Error cuadrático medio')
    plt.title('Convergencia del error - Compuerta OR')
    plt.show()
    
    return model

# 3. Compuerta AND
def train_and_gate():
    # Datos de entrenamiento
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    model = AdalineNetwork(learning_rate=0.1, epochs=100)
    model.fit(X, y)
    
    # Graficar resultados
    plt.plot(model.errors_history)
    plt.xlabel('Época')
    plt.ylabel('Error cuadrático medio')
    plt.title('Convergencia del error - Compuerta AND')
    plt.show()
    
    return model

# 4. RMS vs lambda
def analyze_learning_rates():
    # Datos de ejemplo (usando conversión C a F)
    celsius = normalize_data(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    fahrenheit = np.array([33.8, 35.6, 37.4, 39.2, 41, 42.8, 44.6, 46.4, 48.2, 50])
    
    # Diferentes valores de lambda (learning rate)
    lambdas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9]
    final_errors = []
    
    plt.figure(figsize=(12, 6))
    for lr in lambdas:
        model = AdalineNetwork(learning_rate=lr, epochs=100)
        model.fit(celsius, fahrenheit)
        plt.plot(model.errors_history, label=f'λ={lr}')
        final_errors.append(model.errors_history[-1])
    
    plt.xlabel('Época')
    plt.ylabel('RMS')
    plt.title('RMS vs épocas para diferentes valores de λ')
    plt.legend()
    plt.show()
    
    # Graficar RMS final vs lambda
    plt.figure(figsize=(8, 6))
    plt.semilogx(lambdas, final_errors, 'bo-')
    plt.xlabel('λ (escala logarítmica)')
    plt.ylabel('RMS final')
    plt.title('RMS final vs λ')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("1. Entrenando modelo Celsius a Fahrenheit...")
    celsius_model = celsius_to_fahrenheit()
    
    print("\n2. Entrenando compuerta OR...")
    or_model = train_or_gate()
    
    print("\n3. Entrenando compuerta AND...")
    and_model = train_and_gate()
    
    print("\n4. Analizando diferentes tasas de aprendizaje...")
    analyze_learning_rates()
