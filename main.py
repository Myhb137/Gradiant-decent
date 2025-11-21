import numpy as np
import sympy as sp
import plotly.graph_objects as go
import os
import webbrowser

def gradient_descent_gpu():
    print("--- GPU ACCELERATED GRADIENT DESCENT (Plotly) ---")
    
    # 1. USER INPUT
    func_str = input("Enter function (e.g., x**2 + y**2): ")
    start_x = float(input("Start X: "))
    start_y = float(input("Start Y: "))
    lr = float(input("Learning Rate: "))
    iterations = int(input("Iterations: "))

    # 2. MATH & GRADIENT
    x_sym, y_sym = sp.symbols('x y')
    try:
        f_sym = sp.sympify(func_str)
    except Exception as e:
        print(f"Error parsing function: {e}. Remember to use pure Python syntax.")
        return

    grad_x_sym = sp.diff(f_sym, x_sym)
    grad_y_sym = sp.diff(f_sym, y_sym)

    f_num = sp.lambdify((x_sym, y_sym), f_sym, 'numpy')
    grad_x_num = sp.lambdify((x_sym, y_sym), grad_x_sym, 'numpy')
    grad_y_num = sp.lambdify((x_sym, y_sym), grad_y_sym, 'numpy')

    # 3. ALGORITHM
    x_curr, y_curr = start_x, start_y
    path_x, path_y, path_z = [x_curr], [y_curr], [f_num(x_curr, y_curr)]

    for i in range(iterations): # Changed loop var to i for potential future use
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

    # 4. PLOTTING (GPU Accelerated with Plotly)
    
    # Determine plot range dynamically
    all_x = path_x + [0] # Include 0 to ensure range covers origin if needed
    all_y = path_y + [0]
    max_abs_coord = max(max(np.abs(all_x)), max(np.abs(all_y)))
    range_val = max(1.0, max_abs_coord * 1.2) # Ensure a minimum range for very small paths
    
    axis_range = np.linspace(-range_val, range_val, 150) # Increased resolution
    X, Y = np.meshgrid(axis_range, axis_range)
    Z = f_num(X, Y)

    fig = go.Figure()

    # --- Surface Plot (The "Mountain") ---
    fig.add_trace(go.Surface(
        z=Z, x=axis_range, y=axis_range,
        colorscale='Plasma', # Changed colorscale for better contrast
        cmin=Z.min(), cmax=Z.max(), # Fix colorscale to full range of Z
        opacity=0.85,
        name='Cost Surface',
        hovertemplate='<b>X:</b> %{x}<br><b>Y:</b> %{y}<br><b>Cost:</b> %{z:.4f}<extra></extra>' # Custom hover info
    ))

    # --- Contour Projection on the base ---
    fig.add_trace(go.Contour(
        z=Z, x=axis_range, y=axis_range,
        colorscale='Greys', # Use a grey scale for contours
        showscale=False,
        contours_coloring='lines',
        line_width=1,
        opacity=0.7,
        zmin=Z.min(), zmax=Z.max(),
        name='Contours'
    ))

    # --- Descent Path ---
    fig.add_trace(go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        mode='lines+markers',
        marker=dict(size=3, color='rgb(255,0,0)', opacity=0.8), # Red path markers
        line=dict(color='black', width=4), # Bold black line
        name='Gradient Descent Path',
        hovertemplate='<b>Step:</b> %{pointdata.index}<br><b>X:</b> %{x:.4f}<br><b>Y:</b> %{y:.4f}<br><b>Cost:</b> %{z:.4f}<extra></extra>'
    ))

    # --- Start Point ---
    fig.add_trace(go.Scatter3d(
        x=[path_x[0]], y=[path_y[0]], z=[path_z[0]],
        mode='markers',
        marker=dict(size=6, color='rgb(0,200,0)', symbol='circle', line=dict(width=1, color='black')), # Green start
        name='Start Point'
    ))

    # --- End Point ---
    fig.add_trace(go.Scatter3d(
        x=[path_x[-1]], y=[path_y[-1]], z=[path_z[-1]],
        mode='markers',
        marker=dict(size=6, color='rgb(255,165,0)', symbol='diamond', line=dict(width=1, color='black')), # Orange end
        name='End Point'
    ))

    # --- Layout Enhancements ---
    fig.update_layout(
        title={
            'text': f"<b>Gradient Descent Visualization:</b> <i>{func_str}</i>",
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': {'size': 24, 'color': 'darkblue'}
        },
        scene=dict(
            xaxis_title='<b>Parameter X</b>',
            yaxis_title='<b>Parameter Y</b>',
            zaxis_title='<b>Cost Function Value</b>',
            bgcolor='rgb(230, 230, 230)', # Light grey background
            camera=dict( # Set a nice default camera angle
                eye=dict(x=1.8, y=1.8, z=0.8) 
            ),
            aspectmode='auto' # Allows aspect ratio to adjust
        ),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor='rgba(0,0,0,0.5)',
            borderwidth=1
        ),
        margin=dict(l=0, r=0, b=0, t=50), # Reduce margins
        width=1400, height=900, # Slightly larger default window
        hovermode='closest' # Better hover behavior
    )

    # 5. SAVE & OPEN
    folder_name = "images"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    safe_name = "".join(c for c in func_str if c.isalnum())
    filename = os.path.join(folder_name, f"gpu_descent_{safe_name}.html")
    
    fig.write_html(filename)
    
    webbrowser.open(filename)
    print(f"\n[SUCCESS] Interactive GPU plot saved and opened: {filename}")
    

if __name__ == "__main__":
    gradient_descent_gpu()