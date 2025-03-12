import random

# Parámetros del problema
d = 2
bits_por_variable = 8
tamano_individuo = bits_por_variable * d
rango_min = -5.12
rango_max = 5.12

def binario_a_decimal(binario):
    n = int(binario, 2)
    if len(binario) == 8:
        return n if n < 128 else n - 256
    else:
        raise ValueError("El binario debe ser de 8 bits")

def binario_a_real(binario):
    x1_bin = binario[:8]
    x2_bin = binario[8:]
    x1_int = binario_a_decimal(x1_bin)
    x2_int = binario_a_decimal(x2_bin)
    x1 = x1_int * (5.12 / 128)
    x2 = x2_int * (5.12 / 128)
    return x1, x2

def funcion(x1, x2):
    return x1**2 + x2**2

def crear_poblacion(tamano):
    return [''.join(random.choice('01') for _ in range(tamano_individuo)) for _ in range(tamano)]

def evaluar_poblacion(poblacion):
    return [funcion(*binario_a_real(ind)) for ind in poblacion]

def seleccion_ruleta(poblacion, aptitudes):
    ajustadas = [1 / (apt + 1e-6) for apt in aptitudes]
    total = sum(ajustadas)
    if total == 0:
        return random.choices(poblacion, k=len(poblacion))
    return random.choices(poblacion, weights=ajustadas, k=len(poblacion))

def cruzamiento(padre1, padre2):
    punto = random.randint(1, tamano_individuo - 1)
    hijo1 = padre1[:punto] + padre2[punto:]
    hijo2 = padre2[:punto] + padre1[punto:]
    return hijo1, hijo2

def mutacion(individuo, probabilidad):
    return ''.join(bit if random.random() > probabilidad else '0' if bit == '1' else '1' for bit in individuo)

def algoritmo_genetico(tamano_poblacion, generaciones, prob_mutacion=0.01):
    poblacion = crear_poblacion(tamano_poblacion)
    mejor_aptitud = float('inf')
    mejor_individuo = None
    
    for _ in range(generaciones):
        aptitudes = evaluar_poblacion(poblacion)
        min_apt = min(aptitudes)
        if min_apt < mejor_aptitud:
            idx = aptitudes.index(min_apt)
            mejor_aptitud = min_apt
            mejor_individuo = poblacion[idx]
        
        padres = seleccion_ruleta(poblacion, aptitudes)
        nueva_poblacion = []
        
        for i in range(0, tamano_poblacion, 2):
            if i+1 >= len(padres):
                break
            padre1, padre2 = padres[i], padres[i+1]
            hijo1, hijo2 = cruzamiento(padre1, padre2)
            nueva_poblacion.append(mutacion(hijo1, prob_mutacion))
            nueva_poblacion.append(mutacion(hijo2, prob_mutacion))
        
        poblacion = nueva_poblacion[:tamano_poblacion]
    
    # Evaluar la población final
    aptitudes_finales = evaluar_poblacion(poblacion)
    idx_mejor = aptitudes_finales.index(min(aptitudes_finales))
    mejor_individuo = poblacion[idx_mejor]
    x1, x2 = binario_a_real(mejor_individuo)
    return (x1, x2), aptitudes_finales[idx_mejor]

# Ejecución del algoritmo
mejor_solucion, mejor_aptitud = algoritmo_genetico(
    tamano_poblacion=50,
    generaciones=100,
    prob_mutacion=0.01
)

print(f"Mejor solución encontrada: x1={mejor_solucion[0]:.4f}, x2={mejor_solucion[1]:.4f}")
print(f"Valor mínimo de la función Sphere: {mejor_aptitud:.4f}")

import numpy as np
import matplotlib.pyplot as plt

# Definir el rango y generar la malla
x = np.linspace(-5.12, 5.12, 100)
y = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2  # Función Sphere

# Crear la figura y el eje 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar la superficie
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

# Añadir el punto en (0,0,0)
ax.scatter(0, 0, 0, color='red', s=100, label='(0,0,0)')
ax.scatter(mejor_solucion[0], mejor_solucion[1], mejor_aptitud, color='blue', s=100, label=f"{mejor_solucion[0],mejor_solucion[1]}")

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1,x2)')
ax.set_title('Gráfica 3D de la función Sphere con punto en (0,0)')

# Añadir la barra de colores y la leyenda
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.legend()

plt.show()

