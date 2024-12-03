import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

class AlgoritmoEvolutivo:
    def __init__(self, tam_poblacion=100, num_generaciones=1000, prob_mutacion=0.05):
        self.tam_poblacion = tam_poblacion
        self.num_generaciones = num_generaciones
        self.prob_mutacion = prob_mutacion
        
        # Cargar dataset Iris
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        
        # División train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Variables para tracking
        self.mejor_fitness_por_gen = []
        self.avg_fitness_por_gen = []
        self.mejor_individuo_por_gen = []
        self.poblacion_diversidad = []
        self.feature_usage = []
    
    def inicializar_poblacion(self):
        return [np.random.randint(2, size=self.X.shape[1]) for _ in range(self.tam_poblacion)]
    
    def fitness(self, individuo):
        if sum(individuo) == 0:
            return 0
        
        X_train_sel = self.X_train[:, individuo == 1]
        X_test_sel = self.X_test[:, individuo == 1]
        
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X_train_sel, self.y_train)
        y_pred = clf.predict(X_test_sel)
        return accuracy_score(self.y_test, y_pred)
    
    def seleccion(self, poblacion, fitness_scores):
        seleccionados = []
        fitness_scores = np.array(fitness_scores)
        
        for _ in range(len(poblacion)):
            idx_candidatos = np.random.choice(len(poblacion), 3, replace=False)
            fitness_candidatos = fitness_scores[idx_candidatos]
            ganador_idx = idx_candidatos[np.argmax(fitness_candidatos)]
            seleccionados.append(poblacion[ganador_idx].copy())
        
        return seleccionados
    
    def cruce(self, padre1, padre2):
        punto = np.random.randint(1, len(padre1))
        hijo1 = np.concatenate([padre1[:punto], padre2[punto:]])
        hijo2 = np.concatenate([padre2[:punto], padre1[punto:]])
        return hijo1, hijo2
    
    def mutacion(self, individuo):
        for i in range(len(individuo)):
            if np.random.random() < self.prob_mutacion:
                individuo[i] = 1 - individuo[i]
        return individuo
    
    def calcular_diversidad(self, poblacion):
        return np.mean([np.sum(np.abs(p1 - p2)) 
                       for i, p1 in enumerate(poblacion) 
                       for p2 in poblacion[i+1:]])
    
    def evolucionar(self):
        poblacion = self.inicializar_poblacion()
        
        # Añadir barra de progreso
        print("Iniciando evolución...")
        
        for gen in range(self.num_generaciones):
            if gen % 100 == 0:  # Mostrar progreso cada 100 generaciones
                print(f"Generación {gen}/{self.num_generaciones}")
            
            # Evaluar fitness
            fitness_scores = [self.fitness(ind) for ind in poblacion]
            
            # Guardar métricas
            self.mejor_fitness_por_gen.append(max(fitness_scores))
            self.avg_fitness_por_gen.append(np.mean(fitness_scores))
            mejor_idx = np.argmax(fitness_scores)
            self.mejor_individuo_por_gen.append(poblacion[mejor_idx].copy())
            self.poblacion_diversidad.append(self.calcular_diversidad(poblacion))
            self.feature_usage.append(np.mean(poblacion, axis=0))
            
            # Selección y nueva generación
            seleccionados = self.seleccion(poblacion, fitness_scores)
            nueva_poblacion = [poblacion[mejor_idx].copy()]  # Elitismo
            
            while len(nueva_poblacion) < self.tam_poblacion:
                idx_padres = np.random.choice(len(seleccionados), 2, replace=False)
                h1, h2 = self.cruce(seleccionados[idx_padres[0]], seleccionados[idx_padres[1]])
                h1, h2 = self.mutacion(h1), self.mutacion(h2)
                nueva_poblacion.extend([h1, h2])
            
            poblacion = nueva_poblacion[:self.tam_poblacion]
        
        print("Evolución completada!")
        fitness_final = [self.fitness(ind) for ind in poblacion]
        mejor_idx = np.argmax(fitness_final)
        return poblacion[mejor_idx], fitness_final[mejor_idx]
    
    def visualizar_evolucion(self):
        plt.style.use('default')
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Evolución del Fitness
        ax1 = fig.add_subplot(221)
        generaciones = range(self.num_generaciones)
        ax1.plot(generaciones, self.mejor_fitness_por_gen, 'g-', label='Mejor Fitness', linewidth=2)
        ax1.plot(generaciones, self.avg_fitness_por_gen, 'b--', label='Fitness Promedio', linewidth=2)
        ax1.set_xlabel('Generación')
        ax1.set_ylabel('Fitness (Accuracy)')
        ax1.set_title('Evolución del Fitness')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Diversidad de la Población
        ax2 = fig.add_subplot(222)
        ax2.plot(generaciones, self.poblacion_diversidad, 'r-', linewidth=2)
        ax2.set_xlabel('Generación')
        ax2.set_ylabel('Diversidad')
        ax2.set_title('Diversidad de la Población')
        ax2.grid(True)
        
        # 3. Uso de Características
        ax3 = fig.add_subplot(223)
        feature_usage_array = np.array(self.feature_usage)
        colors = ['b', 'g', 'r', 'c']
        for i, (feature, color) in enumerate(zip(self.feature_names, colors)):
            ax3.plot(generaciones, feature_usage_array[:, i], color=color, 
                    label=feature, linewidth=2)
        ax3.set_xlabel('Generación')
        ax3.set_ylabel('Frecuencia de Uso')
        ax3.set_title('Evolución del Uso de Características')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Mejores Individuos a lo largo del tiempo
        ax4 = fig.add_subplot(224)
        mejores_individuos = np.array(self.mejor_individuo_por_gen)
        im = ax4.imshow(mejores_individuos.T, aspect='auto', cmap='YlOrRd')
        ax4.set_xlabel('Generación')
        ax4.set_ylabel('Característica')
        ax4.set_yticks(range(len(self.feature_names)))
        ax4.set_yticklabels(self.feature_names)
        ax4.set_title('Evolución del Mejor Individuo')
        plt.colorbar(im)
        
        plt.tight_layout()
        return fig

# Ejecutar el algoritmo
np.random.seed(42)
ae = AlgoritmoEvolutivo(tam_poblacion=100, num_generaciones=1000, prob_mutacion=0.05)
mejor_solucion, mejor_fitness = ae.evolucionar()

# Visualizar resultados
fig = ae.visualizar_evolucion()

# Imprimir resultados
print("\nMejor solución encontrada:")
print("Características seleccionadas:", 
      [feature for feature, selected in zip(ae.feature_names, mejor_solucion) if selected])
print(f"Precisión alcanzada: {mejor_fitness:.4f}")

# Mostrar las gráficas
plt.show()
