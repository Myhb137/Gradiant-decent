# 📉 Gradient Descent Visualization Tool (GPU Accelerated)

This project provides an interactive Python script to visualize the **Gradient Descent** optimization algorithm in 3D. It allows the user to define any two-variable function ($f(x, y)$), automatically calculates the gradient, runs the optimization, and plots the results using a **GPU-accelerated** engine (Plotly/WebGL).

## ✨ Features

* **Dynamic Function Input:** Define any differentiable function of $x$ and $y$ at runtime (e.g., `x**2 + y**2` or `x**2 - y**2 + sin(y)`).
* **Automatic Gradient Calculation:** Uses the `SymPy` library to calculate the partial derivatives ($\nabla f(x, y)$) automatically.
* **GPU Accelerated 3D Plotting:** Uses **Plotly** to render a smooth, interactive 3D surface plot and contour map.
* **Path Tracking:** Visualizes the step-by-step path the optimizer takes from the starting point to the minimum.
* **HTML Export:** Saves the interactive 3D plot as an HTML file in the `./images` directory.

## ⚙️ Requirements

You need a working Python 3 environment. The script relies on the following packages:

| Package | Description |
| :--- | :--- |
| `numpy` | High-performance array and numerical computation. |
| `sympy` | Symbolic mathematics (required for automatic differentiation). |
| `plotly` | Interactive, WebGL-based plotting (GPU acceleration). |
| `os` | Standard library for handling directories. |

## 🚀 Installation and Setup

1.  **Clone the Repository (or save the files):**
    Ensure all files (`gradient_descent_tool.py` and `requirements.txt`) are in the same folder.

2.  **Install Dependencies:**
    Use `pip` to install all required libraries from the provided `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ How to Run the Script

1.  Open your terminal or command prompt.
2.  Navigate to the project directory.
3.  Run the Python file:

    ```bash
    python gradient_descent_tool.py
    ```

### Example Input

The script will prompt you for four settings. Use Python's syntax for the function (e.g., `**` for power, `*` for multiplication).

| Setting | Example Input | Purpose |
| :--- | :--- | :--- |
| **Enter function...** | `x**2 + y**2` | The cost function to minimize. |
| **Start X:** | `4` | The starting coordinate on the x-axis. |
| **Start Y:** | `3` | The starting coordinate on the y-axis. |
| **Learning Rate:** | `0.1` | The step size ($\alpha$). Try `1.1` to see divergence! |
| **Iterations:** | `20` | How many steps to take. |

## 💾 Output and Visualization

Upon successful execution:

1.  The console will print the automatically calculated gradient formulas ($\frac{\partial f}{\partial x}$ and $\frac{\partial f}{\partial y}$) and the final coordinates.
2.  A new folder named **`images`** will be created in the project directory.
3.  A fully **interactive HTML file** (e.g., `images/gpu_descent_x2y2.html`) will be saved in that folder. The file will automatically open in your web browser.