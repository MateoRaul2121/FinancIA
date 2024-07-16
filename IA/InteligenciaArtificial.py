import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

data = {
    'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'expenses': [1200, 1300, 1250, 1400, 1500, 1350, 1450, 1600, 1550, 1700, 1650, 1800],
    'savings': [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750]
}

df = pd.DataFrame(data)

# Variables independientes (meses)
X = df[['month']]

# Variable dependiente (gastos)
y_expenses = df['expenses']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train_expenses, y_test_expenses = train_test_split(X, y_expenses, test_size=0.2, random_state=0)

# Crear el modelo de regresión lineal
model_expenses = LinearRegression()

# Entrenar el modelo
model_expenses.fit(X_train, y_train_expenses)

# Hacer predicciones
future_months = np.array([[13], [14], [15]])  # Predicciones para los próximos tres meses
predicted_expenses = model_expenses.predict(future_months)

print(f'Predicted Expenses for Future Months: {predicted_expenses}')

# Similarmente, se puede hacer para ahorros
y_savings = df['savings']
X_train, X_test, y_train_savings, y_test_savings = train_test_split(X, y_savings, test_size=0.2, random_state=0)
model_savings = LinearRegression()
model_savings.fit(X_train, y_train_savings)
predicted_savings = model_savings.predict(future_months)

print(f'Predicted Savings for Future Months: {predicted_savings}')
