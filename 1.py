import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd

# Cargar el archivo Excel
file_path = 'Data10.xlsx'
data = pd.read_excel(file_path)

# Separar las características (features) y la variable objetivo (target)
X = data.drop(columns=['Estado', 'Fecha'])  # Características
y = data['Estado']  # Variable objetivo

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo SVM
svm_model = SVC(kernel='linear')

# Entrenar el modelo
svm_model.fit(X_train, y_train)

# Función para graficar la frontera de decisión
def plot_decision_boundary(clf, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlim(X[:, 0].min(), X[:, 0].max())
    plt.ylim(X[:, 1].min(), X[:, 1].max())
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.title('Frontera de decisión de SVM')

# Graficar los datos y la frontera de decisión
plt.figure(figsize=(10, 6))
plot_decision_boundary(svm_model, X_test, y_test)
plt.show()
