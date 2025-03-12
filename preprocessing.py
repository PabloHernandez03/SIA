import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
df = pd.read_csv('Student_Marks.csv')

# 1. Análisis de correlación lineal
correlation_matrix = df.corr(numeric_only=True)
marks_correlation = correlation_matrix['Marks'].abs().sort_values(ascending=False)

# 2. Determinar la mejor característica
best_feature = marks_correlation.index[1]  # Index 0 sería la correlación con sí mismo
best_corr_value = marks_correlation[1]

print(f"La mejor característica es '{best_feature}' con una correlación absoluta de {best_corr_value:.4f}")

# 3. Crear nuevo dataframe con la mejor característica
best_df = df[[best_feature, 'Marks']]

# 4. Guardar en nuevo archivo CSV
best_df.to_csv('Best_Student_Marks.csv', index=False)
print("\nNuevo archivo guardado como: Best_Student_Marks.csv")

# 5. Visualización de relaciones
plt.figure(figsize=(12, 5))

# Gráfico para number_courses
plt.subplot(1, 2, 1)
sns.regplot(x='number_courses', y='Marks', data=df, scatter_kws={'alpha':0.5})
plt.title(f'Correlación number_courses: {correlation_matrix.loc["number_courses", "Marks"]:.2f}')

# Gráfico para time_study
plt.subplot(1, 2, 2)
sns.regplot(x='time_study', y='Marks', data=df, scatter_kws={'alpha':0.5})
plt.title(f'Correlación time_study: {correlation_matrix.loc["time_study", "Marks"]:.2f}')

plt.tight_layout()
plt.show()