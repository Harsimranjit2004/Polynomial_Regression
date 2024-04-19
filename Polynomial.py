import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# Polynomial regression function
def polyfit(X, y, degree):
    coeffs = np.polyfit(X, y, degree)
    return np.poly1d(coeffs)

# Generate sample data
def generate_data():
    np.random.seed(0)
    X = np.sort(5 * np.random.rand(40))
    y = np.sin(X) + np.random.normal(0, 0.1, len(X))
    return X, y

# Streamlit UI
st.title('Polynomial Regression Visualization')
st.set_option('deprecation.showPyplotGlobalUse', False)
# Sidebar inputs
degree = st.sidebar.slider('Degree of Polynomial', min_value=1, max_value=10, value=2)
# show_equation = st.sidebar.checkbox('Show Equation')

# Generate data and perform polynomial regression
X, y = generate_data()
poly_model = polyfit(X, y, degree)

# Plot
plt.scatter(X, y, color='blue', label='Data points')
x_range = np.linspace(0, 5, 100)
plt.plot(x_range, poly_model(x_range), color='red', label=f'Degree {degree} polynomial')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
st.pyplot()

# Evaluation Metrics
y_pred = poly_model(X)
r_squared = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

# Display evaluation metrics using card component
st.subheader('Evaluation Metrics')
st.write('R-squared')
st.info(f'{r_squared:.2f}')
st.write('RMSE')
st.info(f'{rmse:.2f}')
st.subheader('Instructions')
st.write("Adjust the 'Degree of Polynomial' slider on the sidebar to change the degree of the polynomial regression model.")
