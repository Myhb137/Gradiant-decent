import numpy as np
import sympy as sp
import plotly.graph_objects as go
import os
import webbrowser
from typing import Tuple, List


def get_user_input() -> Tuple[str, float, float, float, int]:
    """Get function and gradient descent parameters from user."""
    func_str = input("Enter function (e.g., x**2 + y**2): ")
    
    # Input validation for floats
    while True:
        try:
            start_x = float(input("Start X: "))
            start_y = float(input("Start Y: "))
            lr = float(input("Learning Rate (>0): "))
            if lr <= 0:
                raise ValueError("Learning rate must be positive.")
            iterations = int(input("Iterations (>0): "))
            if iterations <= 0:
                raise ValueError("Iterations must be positive.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")
    
    return func_str, start_x, start_y, lr, iterations


def compute_gradients(func_str: str) -> Tuple[sp.Expr, callable, callable, callable]:
    """Compute symbolic gradients and convert to numeric functions."""
    x_sym, y_sym = sp.symbols('x y')
    try:
        f_sym = sp.sympify(func_str)
    except Exception as e:
        raise ValueError(f"Error parsing function: {e}. Use valid Python syntax.")
    
    grad_x_sym = sp.diff(f_sym, x_sym)
    grad_y_sym = sp.diff(f_sym, y_sym)
    
    # Convert to numeric functions
    f_num = sp.lambdify((x_sym, y_sym), f_sym, 'numpy')
    grad_x_num = sp.lambdify((x_sym, y_sym), grad_x_sym, 'numpy')
    grad_y_num = sp.lambdify((x_sym, y_sym), grad_y_sym, 'numpy')
    
    return f_sym, f_num, grad_x_num, grad_y_num


def run_gradient_descent(f_num: callable, grad_x_num: callable, grad_y_num: callable,
                         start_x: float, start_y: float, lr: float, iterations: int
                         ) -> Tuple[List[float], List[float], List[float]]:
    """Perform gradient descent and return the path of X, Y, Z."""
    x_curr, y_curr = start_x, start_y
    path_x, path_y, path_z = [x_curr], [y_curr], [f_num(x_curr, y_curr)]
    
    for _ in range(iterations):
        gx = grad_x_num(x_curr, y_curr)
        gy = grad_y_num(x_curr, y_curr)
        x_curr -= lr * gx
        y_curr -= lr * gy
        path_x.append(x_curr)
        path_y.append(y_curr)
        path_z.append(f_num(x_curr, y_curr))
    
    print(f"\n--- Optimization Result ---")
    print(f"Final Minimum Found at: X={x_curr:.4f}, Y={y_curr:.4f}")
    print(f"Cost at Minimum: Z={path_z[-1]:.4f}")
    
    return path_x, path_y, path_z


def plot_gradient_descent(f_num: callable, path_x: List[float], path_y: List[float],
                          path_z: List[float], func_str: str) -> str:
    """Plot the surface, contours, and descent path using Plotly and save HTML."""
    # Determine dynamic plot range
    all_x = path_x + [0]
    all_y = path_y + [0]
    max_abs_coord = max(max(np.abs(all_x)), max(np.abs(all_y)))
    range_val = max(1.0, max_abs_coord * 1.2)
    
    axis_range = np.linspace(-range_val, range_val, 150)
    X, Y = np.meshgrid(axis_range, axis_range)
    Z = f_num(X, Y)
    
    fig = go.Figure()
    
    # Surface
    fig.add_trace(go.Surface(
        z=Z, x=axis_range, y=axis_range,
        colorscale='Plasma', opacity=0.85,
        name='Cost Surface',
        hovertemplate='<b>X:</b> %{x}<br><b>Y:</b> %{y}<br><b>Cost:</b> %{z:.4f}<extra></extra>'
    ))
    
    # Contour
    fig.add_trace(go.Contour(
        z=Z, x=axis_range, y=axis_range,
        colorscale='Greys',
        showscale=False,
        contours_coloring='lines',
        line_width=1,
        opacity=0.7
    ))
    
    # Descent path
    fig.add_trace(go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        mode='lines+markers',
        marker=dict(size=3, color='red', opacity=0.8),
        line=dict(color='black', width=4),
        name='Gradient Descent Path'
    ))
    
    # Start & end points
    fig.add_trace(go.Scatter3d(
        x=[path_x[0]], y=[path_y[0]], z=[path_z[0]],
        mode='markers',
        marker=dict(size=6, color='green', symbol='circle', line=dict(width=1, color='black')),
        name='Start Point'
    ))
    fig.add_trace(go.Scatter3d(
        x=[path_x[-1]], y=[path_y[-1]], z=[path_z[-1]],
        mode='markers',
        marker=dict(size=6, color='orange', symbol='diamond', line=dict(width=1, color='black')),
        name='End Point'
    ))
    
    # Layout
    fig.update_layout(
        title={'text': f"<b>Gradient Descent:</b> <i>{func_str}</i>", 'x':0.5, 'y':0.95,
               'xanchor':'center','yanchor':'top','font':{'size':24,'color':'darkblue'}},
        scene=dict(
            xaxis_title='<b>Parameter X</b>',
            yaxis_title='<b>Parameter Y</b>',
            zaxis_title='<b>Cost Function Value</b>',
            bgcolor='rgb(230,230,230)',
            camera=dict(eye=dict(x=1.8, y=1.8, z=0.8)),
            aspectmode='auto'
        ),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)',
                    bordercolor='rgba(0,0,0,0.5)', borderwidth=1),
        margin=dict(l=0, r=0, b=0, t=50),
        width=1400, height=900,
        hovermode='closest'
    )
    
    # Save HTML
    folder_name = "images"
    os.makedirs(folder_name, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in func_str)
    filename = os.path.join(folder_name, f"gpu_descent_{safe_name}.html")
    fig.write_html(filename)
    webbrowser.open(filename)
    
    print(f"\n[SUCCESS] Interactive plot saved and opened: {filename}")
    return filename


def main():
    func_str, start_x, start_y, lr, iterations = get_user_input()
    try:
        f_sym, f_num, grad_x_num, grad_y_num = compute_gradients(func_str)
    except ValueError as e:
        print(e)
        return
    
    path_x, path_y, path_z = run_gradient_descent(f_num, grad_x_num, grad_y_num,
                                                  start_x, start_y, lr, iterations)
    
    plot_gradient_descent(f_num, path_x, path_y, path_z, func_str)


if __name__ == "__main__":
    main()
