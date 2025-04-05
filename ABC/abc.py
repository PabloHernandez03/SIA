import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HolderTableProblem:
    def __init__(self):
        self.limites = np.array([[-10, 10], [-10, 10]])
    
    def funcion_objetivo(self, x):
        term1 = -np.abs(np.sin(x[0]) * np.cos(x[1]) * \
                np.exp(np.abs(1 - (np.sqrt(x[0]**2 + x[1]**2)/np.pi))))
        return term1

class ColoniaAbejas:
    def __init__(self, problema, num_abejas=50, iteraciones=200, limite=50):
        self.problema = problema
        self.num_abejas = num_abejas
        self.iteraciones = iteraciones
        self.limite = limite
        
        self.mejor = {
            'posicion': None,
            'costo': np.inf,
            'historial': []
        }
        
        self.abejas = np.zeros(num_abejas, dtype=[
            ('posicion', 'float64', (2,)),
            ('costo', 'float64'),
            ('intentos', 'int')
        ])

    def inicializar_colonia(self):
        for i in range(self.num_abejas):
            self.abejas[i]['posicion'] = np.random.uniform(
                low=self.problema.limites[:,0],
                high=self.problema.limites[:,1]
            )
            self.abejas[i]['costo'] = self.problema.funcion_objetivo(self.abejas[i]['posicion'])
            self.abejas[i]['intentos'] = 0
            
            if self.abejas[i]['costo'] < self.mejor['costo']:
                self.mejor['posicion'] = self.abejas[i]['posicion'].copy()
                self.mejor['costo'] = self.abejas[i]['costo']
        
        self.mejor['historial'].append(self.mejor['costo'])

    def generar_nueva_posicion(self, index):
        dim = np.random.randint(0, 2)
        pareja = np.random.choice([i for i in range(self.num_abejas) if i != index])
        
        nueva_pos = self.abejas[index]['posicion'].copy()
        phi = np.random.uniform(-1, 1)
        nueva_pos[dim] += phi * (self.abejas[index]['posicion'][dim] - 
                              self.abejas[pareja]['posicion'][dim])
        
        nueva_pos = np.clip(nueva_pos, 
                          self.problema.limites[:,0], 
                          self.problema.limites[:,1])
        return nueva_pos

    def optimizar(self):
        self.inicializar_colonia()
        
        for iteracion in range(self.iteraciones):
            # Fase empleadas
            for i in range(self.num_abejas):
                nueva_pos = self.generar_nueva_posicion(i)
                nuevo_costo = self.problema.funcion_objetivo(nueva_pos)
                
                if nuevo_costo < self.abejas[i]['costo']:
                    self.abejas[i]['posicion'] = nueva_pos
                    self.abejas[i]['costo'] = nuevo_costo
                    self.abejas[i]['intentos'] = 0
                    
                    if nuevo_costo < self.mejor['costo']:
                        self.mejor['posicion'] = nueva_pos.copy()
                        self.mejor['costo'] = nuevo_costo
                else:
                    self.abejas[i]['intentos'] += 1
            
            # Fase observadoras (corregido)
            fitness = np.array([1/(1 + abs(abeja['costo'])) for abeja in self.abejas])
            prob_sum = fitness.sum()
            
            if prob_sum <= 0:
                probabilidades = np.ones_like(fitness)/len(fitness)
            else:
                probabilidades = fitness / prob_sum
            
            for _ in range(self.num_abejas):
                i = np.random.choice(range(self.num_abejas), p=probabilidades)
                nueva_pos = self.generar_nueva_posicion(i)
                nuevo_costo = self.problema.funcion_objetivo(nueva_pos)
                
                if nuevo_costo < self.abejas[i]['costo']:
                    self.abejas[i]['posicion'] = nueva_pos
                    self.abejas[i]['costo'] = nuevo_costo
                    self.abejas[i]['intentos'] = 0
                    
                    if nuevo_costo < self.mejor['costo']:
                        self.mejor['posicion'] = nueva_pos.copy()
                        self.mejor['costo'] = nuevo_costo
                else:
                    self.abejas[i]['intentos'] += 1
            
            # Fase exploradoras
            for i in range(self.num_abejas):
                if self.abejas[i]['intentos'] > self.limite:
                    self.abejas[i]['posicion'] = np.random.uniform(
                        low=self.problema.limites[:,0],
                        high=self.problema.limites[:,1]
                    )
                    self.abejas[i]['costo'] = self.problema.funcion_objetivo(self.abejas[i]['posicion'])
                    self.abejas[i]['intentos'] = 0
                    
                    if self.abejas[i]['costo'] < self.mejor['costo']:
                        self.mejor['posicion'] = self.abejas[i]['posicion'].copy()
                        self.mejor['costo'] = self.abejas[i]['costo']
            
            # Verificar convergencia con desviación estándar
            costos = [abeja['costo'] for abeja in self.abejas]
            desviacion = np.std(costos)
            if desviacion < 1e-1:  # Ajustar el umbral de convergencia
                print(f"Convergencia alcanzada en iteración {iteracion+1} con desviación estándar {desviacion:.6f}")
                break
            
            self.mejor['historial'].append(self.mejor['costo'])
            print(f"Iteración {iteracion+1}: Mejor costo = {self.mejor['costo']:.4f}")

def visualizar_abc(problema, resultados):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = problema.funcion_objetivo([X[i, j], Y[i, j]])
    
    # Figura 1: Gráfico 3D con mapa de calor y vista superior
    fig1 = plt.figure(figsize=(12, 6))
    ax1 = fig1.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    ax1.scatter(resultados[0], resultados[1], resultados[2], 
                c='red', s=200, marker='o', label='ABC')
    ax1.set_title('Función Holder Table - ABC (3D)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.legend()

    # Vista superior (mapa de calor)
    ax2 = fig1.add_subplot(122)
    heatmap = ax2.contourf(X, Y, Z, cmap='viridis', levels=50)
    ax2.scatter(resultados[0], resultados[1], c='red', s=200, marker='o', label='ABC')
    ax2.set_title('Mapa de Calor - ABC (2D)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(heatmap, ax=ax2)
    plt.legend()
    plt.show()
    
    # Figura 2: Curva de convergencia
    fig2 = plt.figure(figsize=(8, 6))
    plt.plot(resultados[3], label='ABC', color='gold')
    plt.title('Convergencia ABC')
    plt.xlabel('Iteración')
    plt.ylabel('Mejor Costo')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    problema = HolderTableProblem()
    
    # Configurar y ejecutar ABC
    abc = ColoniaAbejas(problema, 
                       num_abejas=50, 
                       iteraciones=100, 
                       limite=30)
    abc.optimizar()
    
    sol_abc = (abc.mejor['posicion'][0],
               abc.mejor['posicion'][1],
               abc.mejor['costo'],
               abc.mejor['historial'])
    
    print("\n" + "="*50)
    print("Resultados Colonia de Abejas:")
    print(f"Posición óptima: ({sol_abc[0]:.5f}, {sol_abc[1]:.5f})")
    print(f"Valor mínimo: {sol_abc[2]:.5f}")
    print(f"Iteraciones totales: {len(sol_abc[3])}")
    
    visualizar_abc(problema, sol_abc)