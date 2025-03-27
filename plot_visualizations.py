import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl

# Try to use the widget backend if in Jupyter notebook
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('matplotlib', 'widget')
except:
    print("Not running in Jupyter notebook. Using default matplotlib backend.")

# Apply the style if the file exists
try:
    plt.style.use('./deeplearning.mplstyle')
except:
    print("Style file not found. Using default matplotlib style.")

# Generate sample data for demonstration
np.random.seed(42)
n_samples = 100
x_train = np.random.uniform(0, 100, n_samples)
y_train = 2 * x_train + 5 + np.random.normal(0, 10, n_samples)
x_val = np.random.uniform(0, 100, n_samples//2)
y_val = 2 * x_val + 5 + np.random.normal(0, 10, n_samples//2)

# Create and display all visualizations
print("Creating interactive visualizations...")

# 1. Data Distribution Plot
print("\n1. Data Distribution Plot")
fig_intuition = plt_intuition(x_train, y_train, x_val, y_val)
plt.show()

# 2. Optimal Regression Surface
print("\n2. Optimal Regression Surface")
w_optimal = 2.0  # True slope
b_optimal = 5.0  # True intercept
fig_stationary = plt_stationary(x_train, y_train, x_val, y_val, w_optimal, b_optimal)
plt.show()

# 3. Interactive Parameter Update
print("\n3. Interactive Parameter Update")
w_test = 1.5  # Test slope
b_test = 3.0  # Test intercept
fig_update = plt_update_onclick(x_train, y_train, x_val, y_val, w_test, b_test)
plt.show()

# 4. Cost Function Surface
print("\n4. Cost Function Surface")
fig_soup = soup_bowl(x_train, y_train, x_val, y_val)
plt.show()

print("\nAll visualizations have been created. You can interact with them using the matplotlib controls.") 