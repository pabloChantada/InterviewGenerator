import matplotlib.pyplot as plt
import os

# Define output directory relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'diagrams')
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------
# Gráfico de pastel: Distribución ponderada de evaluación
# --------------------------------------------------

# Datos
labels = [
    'Similitud Semántica',
    'Validación Numérica\ny Simbólica',
    'Cobertura Conceptual',
    'Estructura de Razonamiento'
]
sizes = [15, 60, 10, 15]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
explode = (0.05, 0.05, 0.05, 0.05)

# Crear figura
fig, ax = plt.subplots(figsize=(10, 8))

# Crear pie chart
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    explode=explode,
    shadow=False,
    textprops={'fontsize': 11, 'weight': 'bold'}
)

# Personalizar porcentajes
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_weight('bold')

# Título
ax.set_title('Distribución Ponderada de Evaluación',
             fontsize=14, weight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'evaluacion_piechart.png'), dpi=300, bbox_inches='tight')
plt.show()

# --------------------------------------------------
# Gráfico de línea: Evolución de la dificultad durante la entrevista
# --------------------------------------------------


# Número de pregunta (eje X)
questions = list(range(1, 11))

# Nivel de dificultad simulado (1=Fácil, 2=Media, 3=Difícil)
difficulty = [1, 2, 3, 3, 2, 2, 3, 2, 1, 2]

plt.figure(figsize=(8, 4))
plt.plot(questions, difficulty, marker='o', linestyle='-', color='#4E79A7')

plt.yticks([1, 2, 3], ['Fácil', 'Media', 'Difícil'])
plt.xticks(questions)

plt.xlabel('Número de pregunta')
plt.ylabel('Nivel de dificultad')
plt.title('Evolución de la dificultad durante la entrevista')

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'evolucion_dificultad.png'), dpi=300)
plt.show()
