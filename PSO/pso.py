import numpy as np

class ProblemaEnergia:
    """
    Clase que modela el problema de optimización energética con parámetros realistas.
    """
    def __init__(self):
        # Parámetros principales
        self.demanda = 100            # Demanda total en MW
        self.capacidad_renovable = 80 # Capacidad máxima renovable
        self.min_fosil = 20           # Mínimo operativo para plantas fósiles
        
        # Costos operativos
        self.costo_renovable = 10     # USD/MW
        self.costo_fosil = 20         # USD/MW
        self.costo_fijo_fosil = 50    # Costo fijo por usar plantas fósiles
        
        # Parámetros ambientales
        self.emision_co2 = 0.5        # Ton CO2/MW fósil
        self.impuesto_co2 = 30       # USD/ton CO2
        
        # Factores de penalización
        self.penalizacion_demanda = 1000
        self.penalizacion_capacidad = 1000
        self.penalizacion_min_fosil = 500

    def funcion_objetivo(self, x):
        """
        Calcula el costo total considerando múltiples factores reales:
        1. Costos operativos variables
        2. Costos fijos por activación de plantas
        3. Impuestos por emisiones
        4. Penalizaciones por restricciones operativas
        """
        renovable = x[0]
        fosil = x[1]
        produccion_total = renovable + fosil
        
        costo = 0
        penalizacion = 0
        
        # 1. Costos operativos básicos
        costo += self.costo_renovable * renovable
        costo += self.costo_fosil * fosil
        
        # 2. Costo fijo por activación de plantas fósiles
        if fosil > 0:
            costo += self.costo_fijo_fosil
        
        # 3. Impuesto por emisiones de CO2
        costo += fosil * self.emision_co2 * self.impuesto_co2
        
        # 4. Penalizaciones por restricciones
        # Demanda no satisfecha
        if produccion_total < self.demanda:
            deficit = self.demanda - produccion_total
            penalizacion += self.penalizacion_demanda * deficit
        
        # Exceso de capacidad renovable
        if renovable > self.capacidad_renovable:
            exceso = renovable - self.capacidad_renovable
            penalizacion += self.penalizacion_capacidad * exceso
        
        # Mínimo operativo plantas fósiles
        if fosil > 0 and fosil < self.min_fosil:
            deficit = self.min_fosil - fosil
            penalizacion += self.penalizacion_min_fosil * deficit
            
        return costo + penalizacion

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
        
        # Espacio de búsqueda [renovable, fosil]
        self.limites = np.array([[0, 100], [0, 150]])
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
        
        # Generar posiciones iniciales con preferencia por solución factible
        for i in range(self.tamano_enjambre):
            renovable = np.random.uniform(0, self.problema.capacidad_renovable)
            fosil = max(self.problema.demanda - renovable, 0)
            self.particulas[i]['posicion'] = [renovable, fosil]
            
            # Aplicar perturbación aleatoria
            self.particulas[i]['posicion'] += np.random.normal(0, 5, size=2)
            
            # Asegurar límites
            self.particulas[i]['posicion'] = np.clip(
                self.particulas[i]['posicion'],
                self.limites[:,0],
                self.limites[:,1]
            )
            
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
                print(f"Convergencia alcanzada en iteración {iteracion+1}")
                break
            
            print(f"Iter {iteracion+1}: Costo ${self.mejor_global['costo']:.2f}")

# Ejecución del algoritmo
problema = ProblemaEnergia()
pso = PSO(problema, tamano_enjambre=50, iteraciones=200)
pso.optimizar()

# Resultados detallados
solucion = pso.mejor_global['posicion']
renovable = solucion[0]
fosil = solucion[1]

print("\n" + "="*50)
print("Mejor solución encontrada:")
print(f"Energía renovable: {renovable:.2f} MW")
print(f"Energía fósil:     {fosil:.2f} MW")
print(f"Producción total:  {renovable + fosil:.2f} MW")
print("\nDesglose de costos:")
print(f"- Operación renovable: ${problema.costo_renovable * renovable:.2f}")
print(f"- Operación fósil:     ${problema.costo_fosil * fosil:.2f}")
if fosil > 0:
    print(f"- Costo fijo fósil:   ${problema.costo_fijo_fosil:.2f}")
    print(f"- Impuesto CO2:       ${fosil * problema.emision_co2 * problema.impuesto_co2:.2f}")
print("\nCosto total: ${:.2f}".format(pso.mejor_global['costo']))
print("="*50)