import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargar los datos
df = pd.read_csv('Best_Student_Marks.csv')

# Datos de entrada
x = df['time_study'].values
y = df['Marks'].values
m = len(y)  # Número de muestras

# Inicialización de parámetros
theta0 = 1.5  # Intercepto
theta1 = 5.1  # Pendiente
alpha = 0.3    # Tasa de aprendizaje
num_iters = 60  # Número de iteraciones
tolerance = 1e-4  # Tolerancia para detener el algoritmo

# Vector para almacenar el error en cada iteración
mse_history = np.zeros(num_iters)

# Configurar gráfica interactiva
plt.figure(figsize=(10, 6))
plt.scatter(x, y, c='blue', marker='o', label='Datos')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regresión Lineal con Gradiente Descendiente (Minimizando MSE)')
plt.grid(True)
plt.legend()
plt.ion()  # Modo interactivo
plt.show()

# Gradiente descendiente iterativo
i = 0
for k in range(num_iters):
    # Predicción con valores actuales
    y_pred = theta0 + theta1 * x[i]

    # Cálculo del error cuadrático medio (MSE)
    mse = (1/m) * np.sum((y_pred - y[i])**2)
    mse_history[k] = mse
    
    # Verificar si el error es menor que la tolerancia
    if mse < tolerance:
        print(f'El algoritmo se detuvo después de {k + 1} iteraciones con un MSE de {mse:.6f}')
        break
    
    # Cálculo de gradientes
    grad_theta0 = (2/m) * np.sum(y_pred - y[i])
    grad_theta1 = (2/m) * np.sum((y_pred - y[i]) * x[i])
    
    # Actualizar parámetros
    theta0 -= alpha * grad_theta0
    theta1 -= alpha * grad_theta1

    # Actualizar índice de muestra
    i = (i + 1) % m
    
    # Graficar cada 10 iteraciones
    if (k + 1) % 10 == 0:
        plt.plot(x, theta0 + theta1 * x, 'red')
        plt.draw()
        plt.pause(0.05)
else:
    print(f'El algoritmo completó todas las {num_iters} iteraciones.')

# Graficar recta final
y_final = theta0 + theta1 * x
plt.plot(x, y_final, 'green', linewidth=2, label='Regresión final')
plt.legend()
plt.ioff()  # Desactivar modo interactivo

# Segunda figura (resultado final)
plt.figure(figsize=(10, 6))
plt.scatter(x, y, c='blue', marker='o', label='Datos')
plt.plot(x, y_final, 'green', linewidth=2, label='Regresión final')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regresión Lineal con Gradiente Descendiente')
plt.grid(True)
plt.legend()

# Mostrar parámetros finales
print(f'Theta0 final: {theta0:.4f}')
print(f'Theta1 final: {theta1:.4f}')

# Gráfica de convergencia del MSE
plt.figure(figsize=(10, 6))
plt.plot(range(1, k + 2), mse_history[:k + 1], 'blue', linewidth=2)
plt.xlabel('Iteraciones')
plt.ylabel('MSE')
plt.title('Convergencia del Error Cuadrático Medio')
plt.grid(True)

plt.show()