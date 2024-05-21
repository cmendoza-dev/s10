import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_blobs

# Cargar el archivo Excel
file_path = 'Data10.xlsx'
data = pd.read_excel(file_path)

# Realizar la codificación de etiquetas para las columnas 'Tipo' y 'Estado'
label_encoder_tipo = LabelEncoder()
data['Tipo'] = label_encoder_tipo.fit_transform(data['Tipo'])

label_encoder_estado = LabelEncoder()
data['Estado'] = label_encoder_estado.fit_transform(data['Estado'])

# Separar las características (features) y la variable objetivo (target)
X = data.drop(columns=['Numero', 'Fecha', 'Estado'])  # Características
y = data['Estado']  # Variable objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y entrenar el modelo SVM con kernel RBF
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Predecir sobre los datos de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Imprimir el reporte de clasificación y la matriz de confusión
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Visualizar la distribución de las clases
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Distribución de las Clases (Predicciones)')
plt.colorbar(label='Clase')
plt.show()

# Función para visualizar el hiperplano de separación y los vectores de soporte
def plot_svc_decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Crear la malla para evaluar el modelo
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # Visualizar el hiperplano y los márgenes
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # Vectores de soporte
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none', edgecolors='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# Visualizar la superficie de decisión con vectores de soporte y margen máximo
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', s=50)
plot_svc_decision_function(model)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Superficie de Decisión del Modelo SVM con Vectores de Soporte y Hiperplano de Separación')
plt.colorbar(label='Clase')
plt.show()
