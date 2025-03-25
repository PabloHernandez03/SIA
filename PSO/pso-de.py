import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class MishraBird:
    """
    Clase que modela la función Mishra Bird para optimización.
    """
    def __init__(self):
        # Límites de la función
        self.limites = np.array([[-10, 0], [-6.5, 0]])
    
    def funcion_objetivo(self, x):
        """
        Calcula el valor de la función Mishra Bird con penalización por restricciones.
        """
        x1, x2 = x
        term1 = np.sin(x2) * np.exp((1 - np.cos(x1))**2)
        term2 = np.cos(x1) * np.exp((1 - np.sin(x2))**2)
        term3 = (x1 - x2)**2
        valor_funcion = term1 + term2 + term3
        
        # Penalización por restricción
        restriccion = (x1 + 5)**2 + (x2 + 5)**2
        if restriccion >= 25:
            penalizacion = 1e6  # Un valor grande para penalizar
            valor_funcion += penalizacion
        
        return valor_funcion
    
    def es_factible(self, x):
        """
        Verifica si una solución es factible en base a la restricción.
        """
        restriccion = (x[0] + 5)**2 + (x[1] + 5)**2
        return restriccion

class PSO:
    """
    Implementación mejorada de PSO con:
    - Control adaptativo de inercia
    - Manejo inteligente de fronteras
    - Velocidades limitadas
    - Criterio de parada temprana
    """
    def __init__(self, problema, tamano_enjambre=50, iteraciones=200):
        self.problema = problema
        self.tamano_enjambre = tamano_enjambre
        self.iteraciones = iteraciones
        
        # Espacio de búsqueda
        self.limites = problema.limites
        self.dimensiones = 2
        
        # Parámetros adaptativos
        self.inercia_inicial = 0.9
        self.inercia_final = 0.4
        self.c1 = 1.7  # Componente cognitivo
        self.c2 = 1.7  # Componente social
        
        # Límites de velocidad (20% del rango por dimensión)
        self.vel_max = (self.limites[:,1] - self.limites[:,0]) * 0.2
        
        self.mejor_global = {
            'posicion': None,
            'costo': np.inf,
            'historial': []
        }

    def inicializar_enjambre(self):
        # Inicialización con distribución inteligente
        self.particulas = np.zeros(self.tamano_enjambre, dtype=[
            ('posicion', 'float64', (self.dimensiones,)),
            ('velocidad', 'float64', (self.dimensiones,)),
            ('mejor_local_pos', 'float64', (self.dimensiones,)),
            ('mejor_local_costo', 'float64'),
            ('costo_actual', 'float64')
        ])
        
        # Generar posiciones iniciales
        for i in range(self.tamano_enjambre):
            self.particulas[i]['posicion'] = np.random.uniform(
                self.limites[:,0], self.limites[:,1])
            
            # Inicializar mejores locales
            self.particulas[i]['mejor_local_pos'] = self.particulas[i]['posicion'].copy()
            costo = self.problema.funcion_objetivo(self.particulas[i]['posicion'])
            self.particulas[i]['mejor_local_costo'] = costo
            self.particulas[i]['costo_actual'] = costo
            
            # Actualizar mejor global
            if costo < self.mejor_global['costo']:
                self.mejor_global['posicion'] = self.particulas[i]['posicion'].copy()
                self.mejor_global['costo'] = costo
                
        self.mejor_global['historial'].append(self.mejor_global['costo'])

    def actualizar_velocidad(self, i, inercia):
        r1 = np.random.rand(self.dimensiones)
        r2 = np.random.rand(self.dimensiones)
        nueva_vel = (inercia * self.particulas[i]['velocidad'] +
                     self.c1 * r1 * (self.particulas[i]['mejor_local_pos'] - 
                                    self.particulas[i]['posicion']) +
                     self.c2 * r2 * (self.mejor_global['posicion'] - 
                                    self.particulas[i]['posicion']))
        # Aplicar límites de velocidad
        return np.clip(nueva_vel, -self.vel_max, self.vel_max)

    def actualizar_posicion(self, i):
        nueva_pos = self.particulas[i]['posicion'] + self.particulas[i]['velocidad']
        # Manejo de fronteras con reflexión
        for d in range(self.dimensiones):
            while True:
                if nueva_pos[d] < self.limites[d,0]:
                    nueva_pos[d] = 2*self.limites[d,0] - nueva_pos[d]
                    self.particulas[i]['velocidad'][d] *= -0.5
                elif nueva_pos[d] > self.limites[d,1]:
                    nueva_pos[d] = 2*self.limites[d,1] - nueva_pos[d]
                    self.particulas[i]['velocidad'][d] *= -0.5
                else:
                    break
        return nueva_pos

    def optimizar(self):
        self.inicializar_enjambre()
        
        for iteracion in range(self.iteraciones):
            inercia = self.inercia_inicial - (self.inercia_inicial - self.inercia_final) * iteracion/self.iteraciones
            for i in range(self.tamano_enjambre):
                # Actualizar velocidad y posición
                self.particulas[i]['velocidad'] = self.actualizar_velocidad(i, inercia)
                self.particulas[i]['posicion'] = self.actualizar_posicion(i)
                
                # Evaluar nueva posición
                costo_actual = self.problema.funcion_objetivo(self.particulas[i]['posicion'])
                self.particulas[i]['costo_actual'] = costo_actual
                
                # Actualizar mejor local
                if costo_actual < self.particulas[i]['mejor_local_costo']:
                    self.particulas[i]['mejor_local_pos'] = self.particulas[i]['posicion'].copy()
                    self.particulas[i]['mejor_local_costo'] = costo_actual
                    
                    # Actualizar mejor global
                    if costo_actual < self.mejor_global['costo']:
                        self.mejor_global['posicion'] = self.particulas[i]['posicion'].copy()
                        self.mejor_global['costo'] = costo_actual
            
            self.mejor_global['historial'].append(self.mejor_global['costo'])
            
            # Criterio de parada temprana
            if iteracion > 20 and (np.std(self.mejor_global['historial'][-20:]) < 0.1):
                # print(f"Convergencia alcanzada en iteración {iteracion+1}")
                break
            
            # print(f"Iter {iteracion+1}: Costo ${self.mejor_global['costo']:.10f}")

class DE:
    """
    Implementación del Algoritmo Evolutivo Diferencial (DE).
    """
    def __init__(self, problema, tamano_poblacion=50, iteraciones=200, F=0.8, CR=0.9):
        self.problema = problema
        self.tamano_poblacion = tamano_poblacion
        self.iteraciones = iteraciones
        self.F = F  # Factor de mutación
        self.CR = CR  # Tasa de recombinación
        
        # Espacio de búsqueda
        self.limites = problema.limites
        self.dimensiones = 2
        
        self.mejor_global = {
            'posicion': None,
            'costo': np.inf,
            'historial': []
        }

    def inicializar_poblacion(self):
        self.poblacion = np.random.uniform(
            self.limites[:,0], self.limites[:,1], 
            (self.tamano_poblacion, self.dimensiones))
        
        self.costos = np.array([self.problema.funcion_objetivo(ind) for ind in self.poblacion])
        
        # Inicializar mejor global
        mejor_idx = np.argmin(self.costos)
        self.mejor_global['posicion'] = self.poblacion[mejor_idx].copy()
        self.mejor_global['costo'] = self.costos[mejor_idx]
        self.mejor_global['historial'].append(self.mejor_global['costo'])

    def mutacion(self, idx):
        indices = [i for i in range(self.tamano_poblacion) if i != idx]
        a, b, c = self.poblacion[np.random.choice(indices, 3, replace=False)]
        mutante = np.clip(a + self.F * (b - c), self.limites[:,0], self.limites[:,1])
        return mutante

    def recombinacion(self, target, mutante):
        crossover = np.random.rand(self.dimensiones) < self.CR
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dimensiones)] = True
        trial = np.where(crossover, mutante, target)
        return trial

    def optimizar(self):
        self.inicializar_poblacion()
        
        for iteracion in range(self.iteraciones):
            for i in range(self.tamano_poblacion):
                mutante = self.mutacion(i)
                trial = self.recombinacion(self.poblacion[i], mutante)
                costo_trial = self.problema.funcion_objetivo(trial)
                
                if costo_trial < self.costos[i]:
                    self.poblacion[i] = trial
                    self.costos[i] = costo_trial
                    
                    if costo_trial < self.mejor_global['costo']:
                        self.mejor_global['posicion'] = trial.copy()
                        self.mejor_global['costo'] = costo_trial
            
            self.mejor_global['historial'].append(self.mejor_global['costo'])
            
            # Criterio de parada temprana
            if iteracion > 20 and (np.std(self.mejor_global['historial'][-20:]) < 0.1):
                # print(f"Convergencia alcanzada en iteración {iteracion+1}")
                break
            
            # print(f"Iter {iteracion+1}: Costo ${self.mejor_global['costo']:.10f}")

import matplotlib.pyplot as plt

# Ejecución del algoritmo PSO y DE múltiples veces
num_repeticiones = 100
resultados_pso = []
resultados_de = []
iteraciones_pso = []
iteraciones_de = []

for _ in range(num_repeticiones):
    print(f"Iteración {_+1}")
    # PSO
    problema = MishraBird()
    pso = PSO(problema, tamano_enjambre=50, iteraciones=200)
    pso.optimizar()
    resultados_pso.append(pso.mejor_global['costo'])
    iteraciones_pso.append(len(pso.mejor_global['historial']))

    # DE
    de = DE(problema, tamano_poblacion=50, iteraciones=200)
    de.optimizar()
    resultados_de.append(de.mejor_global['costo'])
    iteraciones_de.append(len(de.mejor_global['historial']))

# Resultados detallados PSO
print("\n" + "="*50)
print("Resultados PSO:")
print(f"Mejor solución encontrada: {np.min(resultados_pso):.10f}")
print(f"Promedio de soluciones: {np.mean(resultados_pso):.10f}")
print(f"Desviación estándar de soluciones: {np.std(resultados_pso):.10f}")
print(f"¿{pso.mejor_global['posicion'][0]:.15f}, {pso.mejor_global['posicion'][1]:.15f} es factible (<25)? {problema.es_factible(pso.mejor_global['posicion'])}")
print("="*50)

# Resultados detallados DE
print("\n" + "="*50)
print("Resultados DE:")
print(f"Mejor solución encontrada: {np.min(resultados_de):.10f}")
print(f"Promedio de soluciones: {np.mean(resultados_de):.10f}")
print(f"Desviación estándar de soluciones: {np.std(resultados_de):.10f}")
print(f"¿{de.mejor_global['posicion'][0]:.15f}, {de.mejor_global['posicion'][1]:.15f} es factible (<25)? {problema.es_factible(de.mejor_global['posicion'])}")
print("="*50)

# Graficar iteraciones promedio necesarias para encontrar la solución
plt.figure()
plt.bar(['PSO', 'DE'], [np.mean(iteraciones_pso), np.mean(iteraciones_de)], yerr=[np.std(iteraciones_pso), np.std(iteraciones_de)], capsize=5)
plt.xlabel('Algoritmo')
plt.ylabel('Iteraciones promedio')
plt.title('Iteraciones promedio necesarias para encontrar la solución')
plt.grid(True)

# Mostrar el número de iteraciones en la gráfica
plt.text(0, np.mean(iteraciones_pso) + 1, f'{np.mean(iteraciones_pso):.2f}', ha='center')
plt.text(1, np.mean(iteraciones_de) + 1, f'{np.mean(iteraciones_de):.2f}', ha='center')

plt.show()


# Definición de la función Mishra Bird
def mishra_bird(x):
    term1 = np.sin(x[1]) * np.exp((1 - np.cos(x[0]))**2)
    term2 = np.cos(x[0]) * np.exp((1 - np.sin(x[1]))**2)
    term3 = (x[0] - x[1])**2
    return term1 + term2 + term3

# Parámetros de visualización
limites = np.array([[-10, 0], [-10, 0]])  # Rango típico para buscar el mínimo global

# Crear figura 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Generar malla de puntos
x1 = np.linspace(limites[0][0], limites[0][1], 100)
x2 = np.linspace(limites[1][0], limites[1][1], 100)
X1, X2 = np.meshgrid(x1, x2)
Z = np.zeros_like(X1)

# Calcular valores de la función
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i,j] = mishra_bird([X1[i,j], X2[i,j]])

# Graficar superficie
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.6, edgecolor='none')

# Añadir mejores soluciones (ejemplo con valores típicos)
ax.scatter(-3.13, -1.58, -106.76, color='black', s=100, label='Mínimo global real', depthshade=False)
ax.scatter(pso.mejor_global['posicion'][0], pso.mejor_global['posicion'][1], pso.mejor_global['costo'], 
           color='blue', s=80, label='Mejor PSO', marker='^')  # Mejor resultado PSO
ax.scatter(de.mejor_global['posicion'][0], de.mejor_global['posicion'][1], de.mejor_global['costo'], 
           color='red', s=80, label='Mejor DE', marker='o')    # Mejor resultado DE

# Personalización
ax.set_xlabel('X1', fontsize=12)
ax.set_ylabel('X2', fontsize=12)
ax.set_zlabel('f(X1, X2)', fontsize=12)
ax.set_title('Función Mishra Bird y soluciones de PSO vs DE', fontsize=14)
ax.view_init(elev=30, azim=45)  # Ángulo de visualización
ax.legend()
plt.tight_layout()
plt.show()