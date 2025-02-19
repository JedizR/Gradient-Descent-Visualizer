import numpy as np
from flask import Flask, render_template, request, jsonify
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import json
import os

app = Flask(__name__)

class GradientDescentVisualizer:
    def __init__(self, function, gradient, x_range=(-5, 5), y_range=(-5, 5), 
                 resolution=100, learning_rate=0.1, max_iterations=100, 
                 convergence_threshold=1e-6, start_point=None):
        self.function = function
        self.gradient = gradient
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = self.function_vectorized(self.X, self.Y)
        
        if start_point is None:
            self.start_point = [np.random.uniform(x_range[0], x_range[1]),
                                np.random.uniform(y_range[0], y_range[1])]
        else:
            self.start_point = list(start_point)
            
        self.path_history = [self.start_point.copy()]
        self.function_values = [self.function(self.start_point[0], self.start_point[1])]
        
    def function_vectorized(self, x, y):
        result = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                result[i, j] = self.function(x[i, j], y[i, j])
        return result
    
    def perform_gradient_descent(self):
        current_point = self.start_point.copy()
        
        for i in range(self.max_iterations):
            grad = self.gradient(current_point[0], current_point[1])
            
            new_point = [
                current_point[0] - self.learning_rate * grad[0],
                current_point[1] - self.learning_rate * grad[1]
            ]
            
            if (new_point[0] < self.x_range[0] or new_point[0] > self.x_range[1] or
                new_point[1] < self.y_range[0] or new_point[1] > self.y_range[1]):
                break
                
            function_value = self.function(new_point[0], new_point[1])
            self.function_values.append(function_value)
            
            if np.linalg.norm(np.array(new_point) - np.array(current_point)) < self.convergence_threshold:
                current_point = new_point
                self.path_history.append(current_point.copy())
                break
                
            current_point = new_point
            self.path_history.append(current_point.copy())
            
        return self.path_history, self.function_values

    def create_preview_visualization(self):
        """Create a preview visualization of the function without gradient descent path"""
        fig = Figure(figsize=(12, 6), dpi=100, facecolor='#ffffff')
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], figure=fig)
        
        # 3D Surface Plot
        ax1 = fig.add_subplot(gs[0], projection='3d')
        surf = ax1.plot_surface(self.X, self.Y, self.Z, cmap='viridis', alpha=0.8)
        
        # Mark starting point
        start_z = self.function(self.start_point[0], self.start_point[1])
        ax1.scatter([self.start_point[0]], [self.start_point[1]], [start_z], color='red', s=100, label='Start')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('f(X, Y)')
        ax1.set_title('3D Surface')
        
        # Contour Plot
        ax2 = fig.add_subplot(gs[1])
        contour = ax2.contour(self.X, self.Y, self.Z, levels=20, cmap='viridis')
        ax2.contourf(self.X, self.Y, self.Z, levels=20, cmap='viridis', alpha=0.6)
        fig.colorbar(contour, ax=ax2)
        
        # Mark starting point
        ax2.scatter([self.start_point[0]], [self.start_point[1]], color='red', s=100, label='Start')
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Contour Plot')
        ax2.legend()
        
        fig.tight_layout()
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def create_visualization_frame(self, iteration):
        """Create a visualization frame for the given iteration"""
        fig = Figure(figsize=(12, 6), dpi=100, facecolor='#ffffff')
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], figure=fig)
        
        # 3D Surface Plot
        ax1 = fig.add_subplot(gs[0], projection='3d')
        ax1.plot_surface(self.X, self.Y, self.Z, cmap='viridis', alpha=0.8)
        
        # Plot path and current point in 3D
        path = np.array(self.path_history[:iteration+1])
        if len(path) > 0:
            path_z = [self.function(x, y) for x, y in path]
            ax1.plot(path[:, 0], path[:, 1], path_z, 'r-', linewidth=2, label='Path')
            ax1.scatter([path[-1, 0]], [path[-1, 1]], [path_z[-1]], color='blue', s=100)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('f(X, Y)')
        ax1.set_title('3D Surface with Gradient Descent Path')
        
        # Contour Plot
        ax2 = fig.add_subplot(gs[1])
        contour = ax2.contour(self.X, self.Y, self.Z, levels=20, cmap='viridis')
        ax2.contourf(self.X, self.Y, self.Z, levels=20, cmap='viridis', alpha=0.6)
        fig.colorbar(contour, ax=ax2)
        
        # Plot path and current point in 2D
        if len(path) > 0:
            ax2.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, label='Path')
            ax2.scatter([path[-1, 0]], [path[-1, 1]], color='blue', s=100)
            
            if iteration < len(self.function_values):
                current_value = self.function_values[iteration]
                iter_text = f'Iteration: {iteration}\nValue: {current_value:.4f}\nCoords: ({path[-1, 0]:.4f}, {path[-1, 1]:.4f})'
                ax2.text(0.02, 0.95, iter_text, transform=ax2.transAxes,
                      bbox=dict(facecolor='white', alpha=0.7))
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Contour Plot with Gradient Descent Path')
        ax2.legend()
        
        fig.tight_layout()
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

def get_preset_functions():
    preset_functions = {
        "Simple Quadratic": {
            "function": lambda x, y: x**2 + y**2,
            "gradient": lambda x, y: [2*x, 2*y],
            "range": (-5, 5),
            "start": (4, 4),
            "description": "Basic convex function with a single global minimum at (0,0)"
        },
        "Rosenbrock": {
            "function": lambda x, y, a=1, b=100: (a - x)**2 + b * (y - x**2)**2,
            "gradient": lambda x, y, a=1, b=100: [-2 * (a - x) - 4 * b * x * (y - x**2), 
                                                  2 * b * (y - x**2)],
            "range": (-2, 2),
            "start": (-1, 2),
            "description": "Classic banana-shaped function with a narrow curved valley"
        },
        "Himmelblau": {
            "function": lambda x, y: (x**2 + y - 11)**2 + (x + y**2 - 7)**2,
            "gradient": lambda x, y: [4*x*(x**2 + y - 11) + 2*(x + y**2 - 7),
                                      2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)],
            "range": (-5, 5),
            "start": (0, 0),
            "description": "Function with four identical local minima"
        },
        "Rastrigin": {
            "function": lambda x, y, A=10: 2*A + (x**2 - A*np.cos(2*np.pi*x)) + (y**2 - A*np.cos(2*np.pi*y)),
            "gradient": lambda x, y, A=10: [2*x + A*2*np.pi*np.sin(2*np.pi*x),
                                          2*y + A*2*np.pi*np.sin(2*np.pi*y)],
            "range": (-5.12, 5.12),
            "start": (4.5, 4.5),
            "description": "Highly multimodal function with many regularly spaced local minima"
        },
        "Ackley": {
            "function": lambda x, y: -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - 
                                    np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + np.e + 20,
            "gradient": lambda x, y: [
                4 * x * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) / np.sqrt(0.5 * (x**2 + y**2)) +
                np.pi * np.sin(2*np.pi*x) * np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))),
                
                4 * y * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) / np.sqrt(0.5 * (x**2 + y**2)) +
                np.pi * np.sin(2*np.pi*y) * np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))
            ],
            "range": (-5, 5),
            "start": (3, 3),
            "description": "Function with nearly flat outer region and deep hole at center"
        },
        "Three-Hump Camel": {
            "function": lambda x, y: 2*x**2 - 1.05*x**4 + x**6/6 + x*y + y**2,
            "gradient": lambda x, y: [4*x - 4.2*x**3 + x**5 + y, x + 2*y],
            "range": (-2, 2),
            "start": (0.5, 0.5),
            "description": "Function with three local minima of different depths"
        },
        "Easom": {
            "function": lambda x, y: -np.cos(x) * np.cos(y) * np.exp(-(x-np.pi)**2 - (y-np.pi)**2),
            "gradient": lambda x, y: [
                -np.cos(y) * (-np.sin(x) * np.exp(-(x-np.pi)**2 - (y-np.pi)**2) - 
                              np.cos(x) * np.exp(-(x-np.pi)**2 - (y-np.pi)**2) * (-2*(x-np.pi))),
                -np.cos(x) * (-np.sin(y) * np.exp(-(x-np.pi)**2 - (y-np.pi)**2) - 
                              np.cos(y) * np.exp(-(x-np.pi)**2 - (y-np.pi)**2) * (-2*(y-np.pi)))
            ],
            "range": (-10, 10),
            "start": (2, 2),
            "description": "Function with a single minimum in large search space - needle in haystack"
        },
        "Multimodal Function": {
            "function": lambda x, y: (0.5 * (x**2 + y**2) + 
                                     3 * (np.sin(x) * np.sin(y)) +
                                     -2 * np.exp(-((x-3)**2 + (y-3)**2)/2) + 
                                     -3 * np.exp(-((x+3)**2 + (y+3)**2)/3) +
                                     -2.5 * np.exp(-((x-3)**2 + (y+2)**2)/2) +
                                     -1.5 * np.exp(-((x+2)**2 + (y-2)**2)/1.3)),
            "gradient": lambda x, y: [
                x + 3 * np.cos(x) * np.sin(y) + 
                -2 * np.exp(-((x-3)**2 + (y-3)**2)/2) * (-(x-3)/2) +
                -3 * np.exp(-((x+3)**2 + (y+3)**2)/3) * (-(x+3)/3) +
                -2.5 * np.exp(-((x-3)**2 + (y+2)**2)/2) * (-(x-3)/2) +
                -1.5 * np.exp(-((x+2)**2 + (y-2)**2)/1.3) * (-(x+2)/1.3),
                
                y + 3 * np.sin(x) * np.cos(y) +
                -2 * np.exp(-((x-3)**2 + (y-3)**2)/2) * (-(y-3)/2) +
                -3 * np.exp(-((x+3)**2 + (y+3)**2)/3) * (-(y+3)/3) +
                -2.5 * np.exp(-((x-3)**2 + (y+2)**2)/2) * (-(y+2)/2) +
                -1.5 * np.exp(-((x+2)**2 + (y-2)**2)/1.3) * (-(y-2)/1.3)
            ],
            "range": (-5, 5),
            "start": (2, 2),
            "description": "Custom function with multiple local minima at various locations"
        }
    }
    return preset_functions

@app.route('/')
def index():
    preset_functions = get_preset_functions()
    # Generate initial preview for the default function (Simple Quadratic)
    default_function = "Simple Quadratic"
    function_data = preset_functions[default_function]
    
    visualizer = GradientDescentVisualizer(
        function=function_data["function"],
        gradient=function_data["gradient"],
        x_range=function_data["range"],
        y_range=function_data["range"],
        resolution=100,
        start_point=function_data["start"]
    )
    
    preview_image = visualizer.create_preview_visualization()
    
    return render_template('index.html', 
                          preset_functions=preset_functions,
                          default_function=default_function,
                          default_preview=preview_image,
                          default_description=function_data["description"],
                          default_start=function_data["start"],
                          default_range=function_data["range"])

@app.route('/function_preview', methods=['POST'])
def function_preview():
    """Generate a preview visualization for a selected function"""
    data = request.json
    preset_functions = get_preset_functions()
    function_name = data.get('function', 'Simple Quadratic')
    
    if function_name not in preset_functions:
        return jsonify({'error': 'Function not found'}), 404
    
    function_data = preset_functions[function_name]
    
    # Get parameters or use defaults from the function definition
    resolution = data.get('resolution', 100)
    start_x = data.get('start_x', function_data["start"][0])
    start_y = data.get('start_y', function_data["start"][1])
    x_range = y_range = data.get('range', function_data["range"])
    
    visualizer = GradientDescentVisualizer(
        function=function_data["function"],
        gradient=function_data["gradient"],
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        start_point=(start_x, start_y)
    )
    
    preview_image = visualizer.create_preview_visualization()
    
    return jsonify({
        'preview_image': preview_image,
        'description': function_data['description'],
        'range': function_data['range'],
        'start': function_data['start']
    })

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    data = request.json
    preset_functions = get_preset_functions()
    function_name = data.get('function', 'Simple Quadratic')
    
    if function_name not in preset_functions:
        return jsonify({'error': 'Function not found'}), 404
    
    function_data = preset_functions[function_name]
    function = function_data["function"]
    gradient = function_data["gradient"]
    x_range = y_range = data.get('range', function_data["range"])
    learning_rate = data.get('learning_rate', 0.1)
    max_iterations = data.get('max_iterations', 100)
    resolution = data.get('resolution', 100)
    start_x = data.get('start_x', function_data["start"][0])
    start_y = data.get('start_y', function_data["start"][1])
    
    visualizer = GradientDescentVisualizer(
        function=function,
        gradient=gradient,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        convergence_threshold=1e-6,
        start_point=(start_x, start_y)
    )
    
    path, values = visualizer.perform_gradient_descent()
    
    # Create frames for animation
    frames = []
    for i in range(len(path)):
        frame = visualizer.create_visualization_frame(i)
        frames.append({
            'iteration': i,
            'image': frame,
            'value': float(values[i]) if i < len(values) else None,
            'coords': {'x': float(path[i][0]), 'y': float(path[i][1])}
        })
    
    result = {
        'total_iterations': len(path),
        'frames': frames,
        'final_value': float(values[-1]) if values else None,
        'final_coords': {'x': float(path[-1][0]), 'y': float(path[-1][1])} if path else None
    }
    
    return jsonify(result)

@app.route('/get_function_info', methods=['POST'])
def get_function_info():
    data = request.json
    function_name = data.get('function', 'Simple Quadratic')
    preset_functions = get_preset_functions()
    
    if function_name in preset_functions:
        return jsonify({
            'description': preset_functions[function_name]['description'],
            'range': preset_functions[function_name]['range'],
            'start': preset_functions[function_name]['start']
        })
    else:
        return jsonify({'error': 'Function not found'}), 404

# Utility route for handling parameter changes that affect the visualization preview
@app.route('/update_preview', methods=['POST'])
def update_preview():
    data = request.json
    preset_functions = get_preset_functions()
    function_name = data.get('function', 'Simple Quadratic')
    
    if function_name not in preset_functions:
        return jsonify({'error': 'Function not found'}), 404
    
    function_data = preset_functions[function_name]
    resolution = data.get('resolution', 100)
    start_x = data.get('start_x', function_data["start"][0])
    start_y = data.get('start_y', function_data["start"][1])
    
    visualizer = GradientDescentVisualizer(
        function=function_data["function"],
        gradient=function_data["gradient"],
        x_range=function_data["range"],
        y_range=function_data["range"],
        resolution=resolution,
        start_point=(start_x, start_y)
    )
    
    preview_image = visualizer.create_preview_visualization()
    
    return jsonify({
        'preview_image': preview_image
    })

if __name__ == '__main__':
    app.run(debug=True)