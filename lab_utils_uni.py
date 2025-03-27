import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D

def plt_intuition(x_train, y_train, x_val, y_val):
    """Create an intuition plot showing training and validation data."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_train, y_train, c='blue', label='Training Data')
    ax.scatter(x_val, y_val, c='red', label='Validation Data')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')
    ax.set_title('Data Distribution')
    ax.legend()
    return fig

def plt_stationary(x_train, y_train, x_val, y_val, w, b):
    """Create a plot showing the optimal regression line."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_train, y_train, c='blue', label='Training Data')
    ax.scatter(x_val, y_val, c='red', label='Validation Data')
    
    # Plot the regression line
    x_line = np.array([min(x_train), max(x_train)])
    y_line = w * x_line + b
    ax.plot(x_line, y_line, 'g-', label=f'y = {w:.2f}x + {b:.2f}')
    
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')
    ax.set_title('Optimal Regression Line')
    ax.legend()
    return fig

def plt_update_onclick(x_train, y_train, x_val, y_val, w_init, b_init):
    """Create an interactive plot where parameters can be updated."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_train, y_train, c='blue', label='Training Data')
    ax.scatter(x_val, y_val, c='red', label='Validation Data')
    
    # Initial line
    x_line = np.array([min(x_train), max(x_train)])
    y_line = w_init * x_line + b_init
    line, = ax.plot(x_line, y_line, 'g-', label=f'y = {w_init:.2f}x + {b_init:.2f}')
    
    # Add sliders
    ax_w = plt.axes([0.2, 0.02, 0.6, 0.03])
    ax_b = plt.axes([0.2, 0.06, 0.6, 0.03])
    s_w = Slider(ax_w, 'Slope', -5, 5, valinit=w_init)
    s_b = Slider(ax_b, 'Intercept', -5, 5, valinit=b_init)
    
    def update(val):
        w = s_w.val
        b = s_b.val
        line.set_ydata(w * x_line + b)
        line.set_label(f'y = {w:.2f}x + {b:.2f}')
        ax.legend()
        fig.canvas.draw_idle()
    
    s_w.on_changed(update)
    s_b.on_changed(update)
    
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')
    ax.set_title('Interactive Regression Line')
    return fig

def soup_bowl(x_train, y_train, x_val, y_val):
    """Create a 3D visualization of the cost function surface."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a grid of w and b values
    w = np.linspace(-5, 5, 100)
    b = np.linspace(-5, 5, 100)
    W, B = np.meshgrid(w, b)
    
    # Calculate cost for each combination
    cost = np.zeros_like(W)
    for i in range(len(w)):
        for j in range(len(b)):
            y_pred = W[i,j] * x_train + B[i,j]
            cost[i,j] = np.mean((y_pred - y_train) ** 2)
    
    # Plot the surface
    surf = ax.plot_surface(W, B, cost, cmap='viridis')
    ax.set_xlabel('Slope (w)')
    ax.set_ylabel('Intercept (b)')
    ax.set_zlabel('Cost')
    ax.set_title('Cost Function Surface')
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    return fig 