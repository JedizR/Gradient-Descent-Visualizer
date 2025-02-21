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
import random

app = Flask(__name__)

class GradientDescentVisualizer:
    def __init__(self, function, gradient, x_range=(-5, 5), y_range=(-5, 5), 
                 resolution=100, learning_rate=0.1, max_iterations=100, 
                 convergence_threshold=1e-6, start_point=None, 
                 lr_scheduler=None, optimizer='vanilla', elevate=30, azimuth=30):
        self.function = function
        self.gradient = gradient
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.elevate = elevate
        self.azimuth = azimuth
        self.convergence_threshold = convergence_threshold
        
        # Momentum parameters
        self.beta1 = 0.9  # For momentum and Adam
        self.beta2 = 0.999  # For Adam
        self.epsilon = 1e-8  # For Adam
        
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
    
    def get_learning_rate(self, iteration):
        if self.lr_scheduler is None:
            return self.learning_rate
        
        if self.lr_scheduler == 'step':
            # Step decay: reduce by half every 20 iterations
            return self.learning_rate * (0.5 ** (iteration // 20))
        
        elif self.lr_scheduler == 'exponential':
            # Exponential decay
            return self.learning_rate * np.exp(-0.01 * iteration)
        
        elif self.lr_scheduler == 'cosine':
            # Cosine annealing
            return self.learning_rate * (1 + np.cos(np.pi * iteration / self.max_iterations)) / 2
        
        else:
            return self.learning_rate
    
    def perform_gradient_descent(self):
        current_point = self.start_point.copy()
        
        # Optimizer-specific variables
        v_dw = [0, 0]  # Momentum and Adam
        v_db = [0, 0]  # Adam
        
        for i in range(self.max_iterations):
            grad = self.gradient(current_point[0], current_point[1])
            current_lr = self.get_learning_rate(i)
            
            if self.optimizer == 'vanilla':
                # Standard gradient descent
                new_point = [
                    current_point[0] - current_lr * grad[0],
                    current_point[1] - current_lr * grad[1]
                ]
            
            elif self.optimizer == 'momentum':
                # Gradient descent with momentum
                v_dw[0] = self.beta1 * v_dw[0] + (1 - self.beta1) * grad[0]
                v_dw[1] = self.beta1 * v_dw[1] + (1 - self.beta1) * grad[1]
                
                new_point = [
                    current_point[0] - current_lr * v_dw[0],
                    current_point[1] - current_lr * v_dw[1]
                ]
            
            elif self.optimizer == 'adam':
                # Adam optimizer
                v_dw[0] = self.beta1 * v_dw[0] + (1 - self.beta1) * grad[0]
                v_dw[1] = self.beta1 * v_dw[1] + (1 - self.beta1) * grad[1]
                
                v_db[0] = self.beta2 * v_db[0] + (1 - self.beta2) * (grad[0] ** 2)
                v_db[1] = self.beta2 * v_db[1] + (1 - self.beta2) * (grad[1] ** 2)
                
                # Bias correction
                v_dw_corrected = [v_dw[0] / (1 - self.beta1 ** (i + 1)), 
                                  v_dw[1] / (1 - self.beta1 ** (i + 1))]
                v_db_corrected = [v_db[0] / (1 - self.beta2 ** (i + 1)), 
                                  v_db[1] / (1 - self.beta2 ** (i + 1))]
                
                new_point = [
                    current_point[0] - current_lr * v_dw_corrected[0] / (np.sqrt(v_db_corrected[0]) + self.epsilon),
                    current_point[1] - current_lr * v_dw_corrected[1] / (np.sqrt(v_db_corrected[1]) + self.epsilon)
                ]
            
            else:  # Default to vanilla if invalid optimizer
                new_point = [
                    current_point[0] - current_lr * grad[0],
                    current_point[1] - current_lr * grad[1]
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
        
        # Set view angle
        ax1.view_init(elev=self.elevate, azim=self.azimuth)
        
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
        
        # Set view angle
        ax1.view_init(elev=self.elevate, azim=self.azimuth)
        
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
                current_lr = self.get_learning_rate(iteration)
                current_value = self.function_values[iteration]
                iter_text = f'Iteration: {iteration}\nValue: {current_value:.4f}\nCoords: ({path[-1, 0]:.4f}, {path[-1, 1]:.4f})\nLR: {current_lr:.4f}'
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

def generate_rbf_function(complexity=5, amplitude=1.0, seed=None, epsilon=0.5):
    """
    Generate a Radial Basis Function interpolation surface with clear minima
    
    Parameters:
    complexity (int): Number of RBF centers (more centers = more complex surface)
    amplitude (float): Amplitude of the RBF peaks
    seed (int): Random seed for reproducibility
    epsilon (float): Width parameter for RBF functions
    
    Returns:
    function, gradient
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Number of control points based on complexity
    num_control_points = 15 + complexity * 3
    
    # Generate control points
    x_range = y_range = (-4, 4)
    x_points = np.random.uniform(x_range[0], x_range[1], num_control_points)
    y_points = np.random.uniform(y_range[0], y_range[1], num_control_points)
    
    # Generate z values (lower = minima)
    z_base = np.random.uniform(-10, -2, num_control_points) * amplitude
    
    # Ensure we have proper minima
    num_minima = 2 + complexity // 2
    minima_indices = np.random.choice(num_control_points, num_minima, replace=False)
    z_base[minima_indices] = np.random.uniform(-20, -15, num_minima) * amplitude
    
    # Create hills/maxima
    num_hills = 2 + complexity // 3
    hills_indices = np.random.choice(
        [i for i in range(num_control_points) if i not in minima_indices],
        num_hills, 
        replace=False
    )
    z_base[hills_indices] = np.random.uniform(-5, -2, num_hills) * amplitude
    
    # Create the centers and weights for our explicit RBF implementation
    centers = np.column_stack((x_points, y_points))
    weights = z_base
    
    # Width parameters for each RBF
    widths = np.ones(num_control_points) * epsilon
    
    # Add quadratic term to ensure convergence at large distances
    def quadratic_term(x, y):
        dist_factor = 0.05 * (complexity / 5)
        return dist_factor * (x**2 + y**2)
    
    def rbf_function(x, y):
        point = np.array([x, y])
        result = 0
        for i in range(num_control_points):
            center = centers[i]
            dist = np.linalg.norm(point - center)
            result += weights[i] * np.exp(-widths[i] * dist**2)
        # Add quadratic term for stability
        return result + quadratic_term(x, y)
    
    def rbf_gradient(x, y):
        point = np.array([x, y])
        grad_x = 0
        grad_y = 0
        
        for i in range(num_control_points):
            center = centers[i]
            diff = point - center
            dist_squared = np.sum(diff**2)
            exp_term = np.exp(-widths[i] * dist_squared)
            
            grad_x += weights[i] * (-2 * widths[i] * diff[0]) * exp_term
            grad_y += weights[i] * (-2 * widths[i] * diff[1]) * exp_term
        
        # Add gradient of quadratic term
        dist_factor = 0.05 * (complexity / 5)
        grad_x += 2 * dist_factor * x
        grad_y += 2 * dist_factor * y
            
        return [grad_x, grad_y]
    
    return rbf_function, rbf_gradient

def get_preset_functions():
    preset_functions = {
        "RBF Interpolation": {
            "function": None,  # Will be generated dynamically
            "gradient": None,  # Will be generated dynamically
            "range": (-5, 5),
            "start": (3, 3),
            "description": "Radial Basis Function interpolation with randomly placed centers",
            "is_rbf": True,
            "complexity": 5,
            "amplitude": 1.0,
            "seed": 42
        },
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
    # Generate initial preview for the default function (RBF Interpolation)
    default_function = "RBF Interpolation"
    function_data = preset_functions[default_function]
    
    # Generate RBF function
    if function_data.get("is_rbf", False):
        rbf_function, rbf_gradient = generate_rbf_function(
            complexity=function_data["complexity"],
            amplitude=function_data["amplitude"],
            seed=function_data["seed"]
        )
        function_data["function"] = rbf_function
        function_data["gradient"] = rbf_gradient
    
    visualizer = GradientDescentVisualizer(
        function=function_data["function"],
        gradient=function_data["gradient"],
        x_range=function_data["range"],
        y_range=function_data["range"],
        resolution=100,
        start_point=function_data["start"]
    )
    
    preview_image = visualizer.create_preview_visualization()
    
    # Prepare optimizer options
    optimizer_options = {
        "vanilla": "Vanilla Gradient Descent",
        "momentum": "Momentum",
        "adam": "Adam"
    }
    
    # Prepare learning rate scheduler options
    lr_scheduler_options = {
        "constant": "Constant",
        "step": "Step Decay",
        "exponential": "Exponential Decay",
        "cosine": "Cosine Annealing"
    }
    
    return render_template('index.html', 
                          preset_functions=preset_functions,
                          default_function=default_function,
                          default_preview=preview_image,
                          default_description=function_data["description"],
                          default_start=function_data["start"],
                          default_range=function_data["range"],
                          optimizer_options=optimizer_options,
                          lr_scheduler_options=lr_scheduler_options,
                          default_rbf_complexity=function_data.get("complexity", 5),
                          default_rbf_amplitude=function_data.get("amplitude", 1.0),
                          default_rbf_seed=function_data.get("seed", 42))

@app.route('/generate_rbf', methods=['POST'])
def generate_rbf():
    """Generate a new RBF function and provide a preview"""
    data = request.json
    complexity = data.get('complexity', 5)
    amplitude = data.get('amplitude', 1.0)
    seed = data.get('seed')
    
    if seed is not None:
        seed = int(seed)
    
    rbf_function, rbf_gradient = generate_rbf_function(
        complexity=complexity,
        amplitude=amplitude,
        seed=seed
    )
    
    # Get parameters or use defaults
    resolution = data.get('resolution', 100)
    start_x = data.get('start_x', 3.0)
    start_y = data.get('start_y', 3.0)
    x_range = y_range = (-5, 5)
    
    visualizer = GradientDescentVisualizer(
        function=rbf_function,
        gradient=rbf_gradient,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        start_point=(start_x, start_y)
    )
    
    preview_image = visualizer.create_preview_visualization()
    
    return jsonify({
        'preview_image': preview_image,
        'description': f"Custom RBF with {complexity} centers and amplitude {amplitude}"
    })

@app.route('/function_preview', methods=['POST'])
def function_preview():
    """Generate a preview visualization for a selected function"""
    data = request.json
    preset_functions = get_preset_functions()
    function_name = data.get('function', 'RBF Interpolation')
    
    if function_name not in preset_functions:
        return jsonify({'error': 'Function not found'}), 404
    
    function_data = preset_functions[function_name]
    
    # Generate RBF function if needed
    if function_data.get("is_rbf", False):
        complexity = data.get('complexity', function_data.get("complexity", 5))
        amplitude = data.get('amplitude', function_data.get("amplitude", 1.0))
        seed = data.get('seed', function_data.get("seed", None))
        
        if seed is not None:
            seed = int(seed)
        
        rbf_function, rbf_gradient = generate_rbf_function(
            complexity=complexity,
            amplitude=amplitude,
            seed=seed
        )
        function_data["function"] = rbf_function
        function_data["gradient"] = rbf_gradient
    
    # Get parameters or use defaults from the function definition
    resolution = data.get('resolution', 100)
    start_x = data.get('start_x', function_data["start"][0])
    start_y = data.get('start_y', function_data["start"][1])
    x_range = y_range = data.get('range', function_data["range"])
    elevate = data.get('elevate', 30)
    azimuth = data.get('azimuth', 30)
    
    visualizer = GradientDescentVisualizer(
        function=function_data["function"],
        gradient=function_data["gradient"],
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        start_point=(start_x, start_y),
        elevate=elevate,
        azimuth=azimuth
    )
    
    preview_image = visualizer.create_preview_visualization()
    
    return jsonify({
        'preview_image': preview_image,
        'description': function_data['description'],
        'range': function_data['range'],
        'start': function_data['start'],
        'is_rbf': function_data.get('is_rbf', False),
        'complexity': function_data.get('complexity', 5),
        'amplitude': function_data.get('amplitude', 1.0),
        'seed': function_data.get('seed', 42)
    })

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    data = request.json
    preset_functions = get_preset_functions()
    function_name = data.get('function', 'RBF Interpolation')
    
    if function_name not in preset_functions:
        return jsonify({'error': 'Function not found'}), 404
    
    function_data = preset_functions[function_name]
    
    # Generate RBF function if needed
    if function_data.get("is_rbf", False):
        complexity = data.get('complexity', function_data.get("complexity", 5))
        amplitude = data.get('amplitude', function_data.get("amplitude", 1.0))
        seed = data.get('seed', function_data.get("seed", None))
        
        if seed is not None:
            seed = int(seed)
        
        rbf_function, rbf_gradient = generate_rbf_function(
            complexity=complexity,
            amplitude=amplitude,
            seed=seed
        )
        function_data["function"] = rbf_function
        function_data["gradient"] = rbf_gradient
    
    function = function_data["function"]
    gradient = function_data["gradient"]
    
    # Get optimization parameters
    x_range = y_range = data.get('range', function_data["range"])
    learning_rate = data.get('learning_rate', 0.1)
    max_iterations = data.get('max_iterations', 100)
    resolution = data.get('resolution', 100)
    start_x = data.get('start_x', function_data["start"][0])
    start_y = data.get('start_y', function_data["start"][1])
    
    # Get optimizer and learning rate scheduler
    optimizer = data.get('optimizer', 'vanilla')
    lr_scheduler = data.get('lr_scheduler', None)
    
    # Get 3D view angles
    elevate = data.get('elevate', 30)
    azimuth = data.get('azimuth', 30)
    
    # Get the convergence threshold parameter
    convergence_threshold = data.get('convergence_threshold', 1e-6)
    
    visualizer = GradientDescentVisualizer(
        function=function,
        gradient=gradient,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        start_point=(start_x, start_y),
        lr_scheduler=lr_scheduler,
        optimizer=optimizer,
        elevate=elevate,
        azimuth=azimuth
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
            'coords': {'x': float(path[i][0]), 'y': float(path[i][1])},
            'learning_rate': float(visualizer.get_learning_rate(i))
        })
    
    result = {
        'total_iterations': len(path),
        'frames': frames,
        'final_value': float(values[-1]) if values else None,
        'final_coords': {'x': float(path[-1][0]), 'y': float(path[-1][1])} if path else None,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler
    }
    
    return jsonify(result)

@app.route('/get_function_info', methods=['POST'])
def get_function_info():
    data = request.json
    function_name = data.get('function', 'RBF Interpolation')
    preset_functions = get_preset_functions()
    
    if function_name in preset_functions:
        function_data = preset_functions[function_name]
        return jsonify({
            'description': function_data['description'],
            'range': function_data['range'],
            'start': function_data['start'],
            'is_rbf': function_data.get('is_rbf', False),
            'complexity': function_data.get('complexity', 5),
            'amplitude': function_data.get('amplitude', 1.0),
            'seed': function_data.get('seed', 42)
        })
    else:
        return jsonify({'error': 'Function not found'}), 404

@app.route('/update_preview', methods=['POST'])
def update_preview():
    data = request.json
    preset_functions = get_preset_functions()
    function_name = data.get('function', 'RBF Interpolation')
    
    if function_name not in preset_functions:
        return jsonify({'error': 'Function not found'}), 404
    
    function_data = preset_functions[function_name]
    
    # Generate RBF function if needed
    if function_data.get("is_rbf", False):
        complexity = data.get('complexity', function_data.get("complexity", 5))
        amplitude = data.get('amplitude', function_data.get("amplitude", 1.0))
        seed = data.get('seed', function_data.get("seed", None))
        
        if seed is not None:
            seed = int(seed)
        
        rbf_function, rbf_gradient = generate_rbf_function(
            complexity=complexity,
            amplitude=amplitude,
            seed=seed
        )
        function_data["function"] = rbf_function
        function_data["gradient"] = rbf_gradient
    
    resolution = data.get('resolution', 100)
    start_x = data.get('start_x', function_data["start"][0])
    start_y = data.get('start_y', function_data["start"][1])
    
    # Get 3D view angles
    elevate = data.get('elevate', 30)
    azimuth = data.get('azimuth', 30)
    
    visualizer = GradientDescentVisualizer(
        function=function_data["function"],
        gradient=function_data["gradient"],
        x_range=function_data["range"],
        y_range=function_data["range"],
        resolution=resolution,
        start_point=(start_x, start_y),
        elevate=elevate,
        azimuth=azimuth
    )
    
    preview_image = visualizer.create_preview_visualization()
    
    return jsonify({
        'preview_image': preview_image
    })

@app.route('/rotate_view', methods=['POST'])
def rotate_view():
    """Update the 3D plot view angle"""
    data = request.json
    preset_functions = get_preset_functions()
    function_name = data.get('function', 'RBF Interpolation')
    
    if function_name not in preset_functions:
        return jsonify({'error': 'Function not found'}), 404
    
    function_data = preset_functions[function_name]
    
    # Generate RBF function if needed
    if function_data.get("is_rbf", False):
        complexity = data.get('complexity', function_data.get("complexity", 5))
        amplitude = data.get('amplitude', function_data.get("amplitude", 1.0))
        seed = data.get('seed', function_data.get("seed", None))
        
        if seed is not None:
            seed = int(seed)
        
        rbf_function, rbf_gradient = generate_rbf_function(
            complexity=complexity,
            amplitude=amplitude,
            seed=seed
        )
        function_data["function"] = rbf_function
        function_data["gradient"] = rbf_gradient
    
    resolution = data.get('resolution', 100)
    start_x = data.get('start_x', function_data["start"][0])
    start_y = data.get('start_y', function_data["start"][1])
    
    # Get the current view angles
    current_elevate = data.get('elevate', 30)
    current_azimuth = data.get('azimuth', 30)
    
    # Apply the rotation direction
    direction = data.get('direction', 'up')
    
    if direction == 'up':
        elevate = current_elevate + 15
        azimuth = current_azimuth
    elif direction == 'down':
        elevate = current_elevate - 15
        azimuth = current_azimuth
    elif direction == 'left':
        elevate = current_elevate
        azimuth = current_azimuth - 15
    elif direction == 'right':
        elevate = current_elevate
        azimuth = current_azimuth + 15
    else:
        elevate = current_elevate
        azimuth = current_azimuth
    
    # Ensure angles are within valid ranges
    elevate = max(0, min(90, elevate))  # Elevation between 0 and 90 degrees
    azimuth = azimuth % 360  # Azimuth wraps around 0-360
    
    visualizer = GradientDescentVisualizer(
        function=function_data["function"],
        gradient=function_data["gradient"],
        x_range=function_data["range"],
        y_range=function_data["range"],
        resolution=resolution,
        start_point=(start_x, start_y),
        elevate=elevate,
        azimuth=azimuth
    )
    
    preview_image = visualizer.create_preview_visualization()
    
    return jsonify({
        'preview_image': preview_image,
        'elevate': elevate,
        'azimuth': azimuth
    })


if __name__ == '__main__':
    app.run(debug=True)