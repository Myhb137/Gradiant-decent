import numpy as np
import sympy as sp
import plotly.graph_objects as go
import os

def gradient_descent_gpu():
    print("--- GPU ACCELERATED GRADIENT DESCENT (Plotly) ---")
    
    func_str = input("Enter function (e.g., x**2 + y**2): ")
    start_x = float(input("Start X: "))
    start_y = float(input("Start Y: "))
    lr = float(input("Learning Rate: "))
    iterations = int(input("Iterations: "))

    # --- MATH SECTION (CPU) ---
    x_sym, y_sym = sp.symbols('x y')
    try:
        f_sym = sp.sympify(func_str)
    except:
        print("Error parsing function.")
        return

    grad_x_sym = sp.diff(f_sym, x_sym)
    grad_y_sym = sp.diff(f_sym, y_sym)

    f_num = sp.lambdify((x_sym, y_sym), f_sym, 'numpy')
    grad_x_num = sp.lambdify((x_sym, y_sym), grad_x_sym, 'numpy')
    grad_y_num = sp.lambdify((x_sym, y_sym), grad_y_sym, 'numpy')

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

    print(f"Minimum: x={x_curr:.4f}, y={y_curr:.4f}, Cost={path_z[-1]:.4f}")

    # --- GPU PLOTTING SECTION (Plotly) ---
    
    # 1. Create Grid
    all_x = path_x + [0]
    all_y = path_y + [0]
    range_val = max(max(np.abs(all_x)), max(np.abs(all_y))) * 1.2
    axis_range = np.linspace(-range_val, range_val, 100)
    X, Y = np.meshgrid(axis_range, axis_range)
    Z = f_num(X, Y)

    # 2. Build the Figure
    fig = go.Figure()

    # The Mountain (Surface) - Uses GPU
    fig.add_trace(go.Surface(z=Z, x=axis_range, y=axis_range, colorscale='Viridis', opacity=0.8, name='Surface'))

    # The Path (Line)
    fig.add_trace(go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        mode='markers+lines',
        marker=dict(size=4, color='red'),
        line=dict(color='black', width=5),
        name='Descent Path'
    ))

    # Start Point (Green)
    fig.add_trace(go.Scatter3d(
        x=[path_x[0]], y=[path_y[0]], z=[path_z[0]],
        mode='markers', marker=dict(size=8, color='green'), name='Start'
    ))

    # End Point (Yellow)
    fig.add_trace(go.Scatter3d(
        x=[path_x[-1]], y=[path_y[-1]], z=[path_z[-1]],
        mode='markers', marker=dict(size=8, color='yellow'), name='End'
    ))

    fig.update_layout(
        title=f"Gradient Descent: {func_str}",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Cost'
        ),
        width=1200, height=800
    )

    # --- SAVE INTERACTIVE FILE ---
    folder_name = "images"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    safe_name = "".join(c for c in func_str if c.isalnum())
    filename = f"{folder_name}/gpu_descent_{safe_name}.html"
    
    fig.write_html(filename)
    print(f"\n[SUCCESS] Interactive GPU plot saved to: {filename}")
    
    # Automatically open in browser
    fig.show()

if __name__ == "__main__":
    gradient_descent_gpu()