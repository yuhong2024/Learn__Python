import pygame
import numpy as np
import sys
from pygame.locals import *
import math
import random
import time

# Color definitions
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)
LIGHT_BLUE = (173, 216, 230)
DARK_GREEN = (0, 100, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
FINISH_COLOR = (255, 0, 128)  # Pink color for finish line

# Game parameters
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Basic car parameters
CAR_WIDTH = 20
CAR_HEIGHT = 10
SENSOR_LENGTH = 150
NUM_SENSORS = 5

class Car:
    def __init__(self, x, y, angle, brain=None):
        self.x = x
        self.y = y
        self.angle = angle  # angle, 0 means facing right
        self.speed = 0
        self.max_speed = 5
        self.acceleration = 0.1
        self.turn_speed = 0.1
        
        # Sensors
        self.sensors = []
        self.sensor_values = [0] * NUM_SENSORS
        
        # Genome - Neural network weights
        if brain is None:
            self.brain = np.random.uniform(-1, 1, (NUM_SENSORS + 1, 2))  # +1 for bias
        else:
            # Copy or mutate existing brain
            self.brain = brain.copy()
        
        # Fitness
        self.fitness = 0
        self.alive = True
        self.distance_traveled = 0
        self.time_alive = 0
        
        # Checkpoints
        self.next_checkpoint = 0
        self.checkpoints_passed = 0
        
        # Add race completion attributes
        self.position_history = []
        self.max_history_length = 1000
        self.reached_goal = False
        self.completion_time = 0
        self.path_length = 0
        self.best_path = []
        
        # Improved performance tracking
        self.average_speed = 0
        self.top_speed = 0
        self.turn_count = 0  # Track how many directional changes car makes
        self.path_efficiency = 0  # Will be calculated as straight-line/actual path
    
    def think(self):
        # Simple neural network:
        # Input: sensor values
        # Output: [acceleration, steering]
        inputs = np.append(self.sensor_values, 1)  # Add bias
        outputs = np.dot(inputs, self.brain)
        
        # Use tanh activation function to limit output range between -1 and 1
        outputs = np.tanh(outputs)
        
        # Apply outputs
        self.speed += outputs[0] * self.acceleration
        self.speed = max(0, min(self.max_speed, self.speed))  # Limit speed range
        self.angle += outputs[1] * self.turn_speed
    
    def update_sensors(self, walls):
        self.sensors = []
        self.sensor_values = []
        
        # Calculate sensor angles
        angles = []
        for i in range(NUM_SENSORS):
            # Fan-out sensor distribution
            angle = self.angle + math.radians(-90 + 180 * i / (NUM_SENSORS - 1))
            angles.append(angle)
        
        # Calculate collisions for each sensor
        for angle in angles:
            # Sensor start point
            start_x = self.x
            start_y = self.y
            
            # Sensor end point
            end_x = start_x + SENSOR_LENGTH * math.cos(angle)
            end_y = start_y + SENSOR_LENGTH * math.sin(angle)
            
            # Check sensor collision with walls
            closest_point = None
            min_distance = SENSOR_LENGTH
            
            for wall in walls:
                # Detect line segment intersection
                intersection = self.line_intersection(
                    (start_x, start_y), (end_x, end_y),
                    (wall[0], wall[1]), (wall[2], wall[3])
                )
                
                if intersection:
                    # Calculate distance to intersection point
                    dx = intersection[0] - start_x
                    dy = intersection[1] - start_y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = intersection
            
            # Save sensor information
            if closest_point:
                self.sensors.append((start_x, start_y, closest_point[0], closest_point[1]))
                # Normalize sensor value to 0-1 range
                self.sensor_values.append(min_distance / SENSOR_LENGTH)
            else:
                self.sensors.append((start_x, start_y, end_x, end_y))
                self.sensor_values.append(1.0)  # No collision
    
    def update(self, walls, checkpoints, finish_line=None):
        if not self.alive:
            return
            
        self.time_alive += 1
        
        # Optimization: Early stall detection
        # If car hasn't moved significantly in the last 30 frames, consider it stalled
        if len(self.position_history) > 30:
            recent_positions = self.position_history[-30:]
            total_distance = 0
            
            for i in range(1, len(recent_positions)):
                dx = recent_positions[i][0] - recent_positions[i-1][0]
                dy = recent_positions[i][1] - recent_positions[i-1][1]
                total_distance += math.sqrt(dx*dx + dy*dy)
            
            # If car barely moved in 30 frames, consider it stalled
            if total_distance < 10:  # Threshold for considering a car stalled
                self.alive = False
                return
        
        # Save previous position and angle for analysis
        prev_x, prev_y = self.x, self.y
        prev_angle = self.angle
        
        # Think and move
        self.think()
        
        # Update position based on angle and speed
        dx = self.speed * math.cos(self.angle)
        dy = self.speed * math.sin(self.angle)
        self.x += dx
        self.y += dy
        
        # Track performance metrics
        distance_moved = math.sqrt(dx*dx + dy*dy)
        self.distance_traveled += distance_moved
        self.path_length += distance_moved
        
        # Track top speed
        if self.speed > self.top_speed:
            self.top_speed = self.speed
            
        # Track average speed
        self.average_speed = self.distance_traveled / max(1, self.time_alive)
        
        # Track significant direction changes (turns)
        angle_change = abs(self.angle - prev_angle)
        if angle_change > 0.05:  # Consider this a significant turn
            self.turn_count += 1
        
        # Record position in history
        self.position_history.append((self.x, self.y))
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
        
        # Update sensors
        self.update_sensors(walls)
        
        # Check wall collisions
        if self.check_collision(walls):
            self.alive = False
        
        # Check if car reached finish line with improved detection
        if finish_line and not self.reached_goal:
            if self.check_finish_line_collision(finish_line):
                self.reached_goal = True
                self.completion_time = self.time_alive
                
                # Calculate path efficiency (straight-line distance / actual path length)
                if len(self.position_history) > 0:
                    start_pos = self.position_history[0]
                    end_pos = (self.x, self.y)
                    straight_line = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
                    self.path_efficiency = straight_line / max(1, self.path_length)
                
                self.best_path = self.position_history.copy()
        
        # Check checkpoints (for guiding the learning process)
        if self.next_checkpoint < len(checkpoints):
            checkpoint = checkpoints[self.next_checkpoint]
            # Simple checkpoint: if car is close to checkpoint center
            dx = self.x - checkpoint[0]
            dy = self.y - checkpoint[1]
            distance = math.sqrt(dx*dx + dy*dy)
            if distance < checkpoint[2]:  # checkpoint[2] is checkpoint radius
                self.next_checkpoint += 1
                self.checkpoints_passed += 1
        
        # Improved fitness calculation - encourage efficient paths
        if self.reached_goal:
            # Base reward for reaching goal
            finish_reward = 10000
            
            # Time efficiency bonus (faster is better)
            time_factor = 5000 / max(1, self.completion_time)
            
            # Path efficiency bonus (straighter is better)
            path_efficiency_bonus = 2000 * self.path_efficiency
            
            # Speed consistency bonus
            speed_bonus = 1000 * (self.average_speed / max(1, self.max_speed))
            
            # Penalize too many turns
            turn_penalty = min(500, 10 * self.turn_count)
            
            self.fitness = finish_reward + time_factor + path_efficiency_bonus + speed_bonus - turn_penalty
        else:
            # Progressive fitness for cars still trying
            checkpoint_progress = self.checkpoints_passed * 500
            distance_progress = self.distance_traveled
            
            # Encourage cars moving at good speed
            speed_factor = self.average_speed * 50
            
            self.fitness = checkpoint_progress + distance_progress + speed_factor
            
            if not self.alive:
                self.fitness *= 0.8  # Penalty for dead cars
    
    def check_collision(self, walls):
        # More efficient collision detection
        # First do a quick boundary check
        if (self.x < 0 or self.x > SCREEN_WIDTH or 
            self.y < 0 or self.y > SCREEN_HEIGHT):
            return True
        
        # Then check collisions with walls
        car_radius = (CAR_WIDTH + CAR_HEIGHT) / 4
        
        for wall in walls:
            # Calculate squared distance from car to wall segment (more efficient)
            closest_point = self.closest_point_on_line(
                (self.x, self.y), 
                (wall[0], wall[1]), 
                (wall[2], wall[3])
            )
            
            if closest_point:
                dx = self.x - closest_point[0]
                dy = self.y - closest_point[1]
                distance_squared = dx*dx + dy*dy
                
                if distance_squared < car_radius * car_radius:
                    return True
        
        return False
    
    def check_finish_line_collision(self, finish_line):
        """Check if car has crossed the finish line"""
        line_start = (finish_line[0], finish_line[1])
        line_end = (finish_line[2], finish_line[3])
        
        # Check if last movement crossed the finish line
        if len(self.position_history) >= 2:
            prev_pos = self.position_history[-2]
            current_pos = (self.x, self.y)
            
            # Check if movement line intersects with finish line
            intersection = self.line_intersection(
                prev_pos, current_pos,
                line_start, line_end
            )
            
            return intersection is not None
        
        return False
    
    def draw(self, screen, show_trajectory=False):
        # Draw car
        car_color = GREEN if self.alive else RED
        if self.reached_goal:
            car_color = FINISH_COLOR  # Cars that finished get a special color
            
        pygame.draw.circle(screen, car_color, (int(self.x), int(self.y)), 5)
        
        # Draw direction indicator
        end_x = self.x + 15 * math.cos(self.angle)
        end_y = self.y + 15 * math.sin(self.angle)
        pygame.draw.line(screen, BLUE, (self.x, self.y), (end_x, end_y), 2)
        
        # Draw sensors
        for sensor in self.sensors:
            # Calculate opacity based on sensor value (closer = more opaque)
            sensor_idx = self.sensors.index(sensor)
            if sensor_idx < len(self.sensor_values):
                opacity = int(255 * (1 - self.sensor_values[sensor_idx]))
                sensor_color = (255, 255, 0, opacity)  # Semi-transparent yellow
            else:
                sensor_color = YELLOW
                
            pygame.draw.line(screen, sensor_color, (sensor[0], sensor[1]), (sensor[2], sensor[3]), 1)
        
        # Draw trajectory if requested and car has a history
        if show_trajectory and len(self.position_history) > 1:
            for i in range(1, len(self.position_history)):
                # Use gradient color to show time progression
                alpha = int(200 * (i / len(self.position_history)))
                if self.reached_goal:
                    color = ORANGE  # Successful paths in orange
                else:
                    color = LIGHT_BLUE  # Active cars in light blue
                    
                pygame.draw.line(
                    screen,
                    color,
                    (int(self.position_history[i-1][0]), int(self.position_history[i-1][1])),
                    (int(self.position_history[i][0]), int(self.position_history[i][1])),
                    1
                )
    
    @staticmethod
    def line_intersection(line1_start, line1_end, line2_start, line2_end):
        # Line segment intersection detection
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
            
        A = line1_start
        B = line1_end
        C = line2_start
        D = line2_end
        
        # Quick check if segments intersect
        if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
            # Calculate intersection point
            xdiff = (A[0] - B[0], C[0] - D[0])
            ydiff = (A[1] - B[1], C[1] - D[1])
            
            def det(a, b):
                return a[0] * b[1] - a[1] * b[0]
                
            div = det(xdiff, ydiff)
            if div == 0:
                return None
                
            d = (det(A, B), det(C, D))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            return (x, y)
            
        return None
    
    @staticmethod
    def closest_point_on_line(point, line_start, line_end):
        # Calculate closest point from a point to a line segment
        x1, y1 = line_start
        x2, y2 = line_end
        x0, y0 = point
        
        # Vector direction
        dx = x2 - x1
        dy = y2 - y1
        
        # If line segment degenerates to a point
        if dx == 0 and dy == 0:
            return line_start
            
        # Calculate projection position
        t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
        
        # Restrict to line segment
        t = max(0, min(1, t))
        
        # Calculate closest point coordinates
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        return (closest_x, closest_y)

class GeneticAlgorithm:
    def __init__(self, population_size=30, elite_size=5, mutation_rate=0.1, crossover_rate=0.7):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generation = 0
        self.population = []
        self.best_fitness = 0
        self.best_car = None
    
    def initialize_population(self, start_x, start_y, start_angle):
        self.population = []
        for _ in range(self.population_size):
            self.population.append(Car(start_x, start_y, start_angle))
    
    def evaluate_fitness(self, walls, checkpoints):
        # All cars' fitness is already calculated in the update method
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Update best fitness
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_car = self.population[0]
    
    def select_parents(self):
        # Use tournament selection
        parents = []
        
        # Elites are directly selected
        for i in range(self.elite_size):
            parents.append(self.population[i])
        
        # Remaining use tournament selection
        for _ in range(self.population_size - self.elite_size):
            # Randomly select 3 individuals
            tournament = random.sample(self.population, 3)
            # Select the best one
            tournament.sort(key=lambda x: x.fitness, reverse=True)
            parents.append(tournament[0])
        
        return parents
    
    def crossover(self, parent1, parent2):
        # Enhanced crossover with multiple methods
        if random.random() < self.crossover_rate:
            child_brain = parent1.brain.copy()
            
            # Choose crossover method randomly
            method = random.choice(['single_point', 'two_point', 'uniform'])
            
            if method == 'single_point':
                # Single-point crossover (existing method)
                crossover_point = random.randint(0, parent1.brain.shape[0] * parent1.brain.shape[1] - 1)
                row = crossover_point // parent1.brain.shape[1]
                col = crossover_point % parent1.brain.shape[1]
                
                for r in range(row, parent1.brain.shape[0]):
                    start_col = 0
                    if r == row:
                        start_col = col
                        
                    for c in range(start_col, parent1.brain.shape[1]):
                        child_brain[r, c] = parent2.brain[r, c]
                        
            elif method == 'two_point':
                # Two-point crossover
                size = parent1.brain.shape[0] * parent1.brain.shape[1]
                point1 = random.randint(0, size - 2)
                point2 = random.randint(point1 + 1, size - 1)
                
                flat_brain1 = parent1.brain.flatten()
                flat_brain2 = parent2.brain.flatten()
                flat_child = np.copy(flat_brain1)
                
                flat_child[point1:point2] = flat_brain2[point1:point2]
                child_brain = flat_child.reshape(parent1.brain.shape)
                
            elif method == 'uniform':
                # Uniform crossover
                for r in range(parent1.brain.shape[0]):
                    for c in range(parent1.brain.shape[1]):
                        if random.random() < 0.5:
                            child_brain[r, c] = parent2.brain[r, c]
            
            return child_brain
        else:
            # No crossover, return more fit parent
            if parent1.fitness > parent2.fitness:
                return parent1.brain.copy()
            else:
                return parent2.brain.copy()
    
    def mutate(self, brain):
        # Enhanced mutation with adaptive rate
        # Determine if this is a successful individual
        is_successful = self.best_car is not None and np.array_equal(brain, self.best_car.brain)
        
        # Lower mutation rate for successful individuals
        effective_rate = self.mutation_rate * 0.5 if is_successful else self.mutation_rate
        
        # Apply mutation
        for i in range(brain.shape[0]):
            for j in range(brain.shape[1]):
                if random.random() < effective_rate:
                    mutation_strength = random.choice([
                        random.uniform(-0.1, 0.1),  # Small change
                        random.uniform(-0.5, 0.5),  # Medium change
                        random.uniform(-1.0, 1.0)   # Large change
                    ])
                    brain[i, j] += mutation_strength
                    # Limit to -1 to 1 range
                    brain[i, j] = max(-1, min(1, brain[i, j]))
        
        return brain
    
    def evolve(self, start_x, start_y, start_angle, walls, checkpoints):
        # Evaluate current population
        self.evaluate_fitness(walls, checkpoints)
        
        # More clearly track evolution progress
        avg_fitness = sum(car.fitness for car in self.population) / len(self.population)
        print(f"Generation {self.generation}: Best fitness = {self.best_fitness}, Avg fitness = {avg_fitness}")
        
        # Track improvement over generations
        improvement_ratio = 0
        if self.generation > 0 and hasattr(self, 'last_best_fitness'):
            improvement_ratio = (self.best_fitness - self.last_best_fitness) / max(1, self.last_best_fitness)
            print(f"Improvement: {improvement_ratio*100:.2f}%")
        
        self.last_best_fitness = self.best_fitness
        
        # Select parents
        parents = self.select_parents()
        
        # Create next generation
        next_generation = []
        
        # Elite preservation
        for i in range(self.elite_size):
            # Create new car but keep original brain
            elite_car = Car(start_x, start_y, start_angle, parents[i].brain.copy())
            next_generation.append(elite_car)
        
        # Remaining are produced by crossover and mutation
        while len(next_generation) < self.population_size:
            # Randomly select two parents
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # Crossover
            child_brain = self.crossover(parent1, parent2)
            
            # Mutation
            child_brain = self.mutate(child_brain)
            
            # Create child
            child = Car(start_x, start_y, start_angle, child_brain)
            next_generation.append(child)
        
        # Update population
        self.population = next_generation
        self.generation += 1
        
        return self.population

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Smart Car Evolution Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        
        # Game state
        self.running = True
        self.paused = False
        self.editing = True  # Start in edit mode
        self.show_all_cars = True
        
        # Editor related
        self.walls = []
        self.checkpoints = []
        self.start_point = (100, 400)
        self.edit_mode = "wall"  # 'wall' or 'checkpoint'
        self.temp_wall = None
        
        # Genetic algorithm
        self.ga = GeneticAlgorithm(
            population_size=30,
            elite_size=5,
            mutation_rate=0.1,
            crossover_rate=0.7
        )
        
        # Initialize population
        self.restart_simulation()
        
        # Create a sidebar layout for better organization
        sidebar_width = 250
        self.sidebar_rect = pygame.Rect(SCREEN_WIDTH - sidebar_width, 0, sidebar_width, SCREEN_HEIGHT)
        self.simulation_rect = pygame.Rect(0, 0, SCREEN_WIDTH - sidebar_width, SCREEN_HEIGHT)
        
        # Sidebar controls
        button_width = 230
        button_height = 30
        button_spacing = 10
        sidebar_margin = 10
        
        # Group buttons by function
        group_y = 20
        self.buttons = []
        
        # Track editing group
        self.buttons.append({"rect": pygame.Rect(SCREEN_WIDTH - sidebar_width + sidebar_margin, group_y, button_width, button_height), 
                             "text": "Wall Mode", "action": "wall_mode", "group": "edit"})
        group_y += button_height + 5
        self.buttons.append({"rect": pygame.Rect(SCREEN_WIDTH - sidebar_width + sidebar_margin, group_y, button_width, button_height), 
                             "text": "Checkpoint Mode", "action": "checkpoint_mode", "group": "edit"})
        group_y += button_height + 5
        self.buttons.append({"rect": pygame.Rect(SCREEN_WIDTH - sidebar_width + sidebar_margin, group_y, button_width, button_height), 
                             "text": "Set Start", "action": "set_start", "group": "edit"})
        group_y += button_height + 5
        self.buttons.append({"rect": pygame.Rect(SCREEN_WIDTH - sidebar_width + sidebar_margin, group_y, button_width, button_height), 
                             "text": "Set Finish", "action": "set_finish", "group": "edit"})
        
        # Add Clear Track button to the Track Editor group
        self.buttons.append({"rect": pygame.Rect(SCREEN_WIDTH - sidebar_width + sidebar_margin, group_y + 5, button_width, button_height), 
                             "text": "Clear Track", "action": "clear_track", "group": "edit"})
        group_y += button_height + 5
        
        # Add Save/Load feature buttons
        self.buttons.append({"rect": pygame.Rect(SCREEN_WIDTH - sidebar_width + sidebar_margin, group_y, button_width//2 - 5, button_height), 
                             "text": "Save Track", "action": "save_track", "group": "edit"})
        self.buttons.append({"rect": pygame.Rect(SCREEN_WIDTH - sidebar_width + sidebar_margin + button_width//2 + 5, group_y, button_width//2 - 5, button_height), 
                             "text": "Load Track", "action": "load_track", "group": "edit"})
        
        # Simulation control group
        group_y += button_height + 20  # Add extra space between groups
        self.buttons.append({"rect": pygame.Rect(SCREEN_WIDTH - sidebar_width + sidebar_margin, group_y, button_width, button_height), 
                             "text": "Start/Pause", "action": "toggle_pause", "group": "sim"})
        group_y += button_height + 5
        self.buttons.append({"rect": pygame.Rect(SCREEN_WIDTH - sidebar_width + sidebar_margin, group_y, button_width, button_height), 
                             "text": "Reset Simulation", "action": "restart", "group": "sim"})
        group_y += button_height + 5
        self.buttons.append({"rect": pygame.Rect(SCREEN_WIDTH - sidebar_width + sidebar_margin, group_y, button_width, button_height), 
                             "text": "Show All/Best Cars", "action": "toggle_view", "group": "sim"})
        
        # Speed and visualization controls
        group_y += button_height + 20  # Add extra space between groups
        speed_btn_width = (button_width - 10) // 2
        self.buttons.append({"rect": pygame.Rect(SCREEN_WIDTH - sidebar_width + sidebar_margin, group_y, speed_btn_width, button_height), 
                             "text": "Speed Up (+)", "action": "speed_up", "group": "speed"})
        self.buttons.append({"rect": pygame.Rect(SCREEN_WIDTH - sidebar_width + sidebar_margin + speed_btn_width + 10, group_y, speed_btn_width, button_height), 
                             "text": "Slow Down (-)", "action": "slow_down", "group": "speed"})
        
        group_y += button_height + 5
        self.buttons.append({"rect": pygame.Rect(SCREEN_WIDTH - sidebar_width + sidebar_margin, group_y, speed_btn_width, button_height), 
                             "text": "Toggle Trail", "action": "toggle_trajectory", "group": "vis"})
        self.buttons.append({"rect": pygame.Rect(SCREEN_WIDTH - sidebar_width + sidebar_margin + speed_btn_width + 10, group_y, speed_btn_width, button_height), 
                             "text": "Clear Trail", "action": "clear_trajectory", "group": "vis"})
        
        # Preset track buttons
        group_y += button_height + 20  # Add extra space between groups
        self.buttons.append({"rect": pygame.Rect(SCREEN_WIDTH - sidebar_width + sidebar_margin, group_y, button_width, button_height), 
                             "text": "Simple Oval Track", "action": "preset_simple", "group": "presets"})
        group_y += button_height + 5
        self.buttons.append({"rect": pygame.Rect(SCREEN_WIDTH - sidebar_width + sidebar_margin, group_y, button_width, button_height), 
                             "text": "Figure 8 Track", "action": "preset_figure8", "group": "presets"})
        group_y += button_height + 5
        self.buttons.append({"rect": pygame.Rect(SCREEN_WIDTH - sidebar_width + sidebar_margin, group_y, button_width, button_height), 
                             "text": "Race Track", "action": "preset_race", "group": "presets"})
        
        # Sliders for genetic algorithm parameters
        group_y += button_height + 30
        slider_width = button_width
        slider_spacing = 40
        
        self.sliders = [
            {"x": SCREEN_WIDTH - sidebar_width + sidebar_margin, "y": group_y, "width": slider_width, 
             "value": self.ga.mutation_rate, "min": 0, "max": 0.5, "text": "Mutation Rate"},
            
            {"x": SCREEN_WIDTH - sidebar_width + sidebar_margin, "y": group_y + slider_spacing, "width": slider_width, 
             "value": self.ga.crossover_rate, "min": 0, "max": 1, "text": "Crossover Rate"},
            
            {"x": SCREEN_WIDTH - sidebar_width + sidebar_margin, "y": group_y + 2 * slider_spacing, "width": slider_width, 
             "value": self.ga.population_size, "min": 10, "max": 100, "step": 5, "text": "Population Size"},
            
            {"x": SCREEN_WIDTH - sidebar_width + sidebar_margin, "y": group_y + 3 * slider_spacing, "width": slider_width, 
             "value": self.ga.elite_size, "min": 1, "max": 10, "step": 1, "text": "Elite Count"},
        ]
        
        # Timer related variables
        self.start_time = time.time()
        self.generation_start_time = time.time()
        self.total_runtime = 0
        self.generation_time = 0
        
        # Speed control
        self.simulation_speed = 3  # Start at 3x speed instead of 1x
        self.max_simulation_speed = 10  # Maximum speed multiplier
        
        # Trajectory display controls
        self.show_best_trajectory = True
        self.best_trajectory = []
        
        # Best performance tracking
        self.best_completion_time = float('inf')
        self.best_path_length = float('inf')
        self.best_path = []
        
        # Current generation stats
        self.cars_finished = 0
        self.best_time_this_gen = float('inf')
        
        # Add finish line
        self.finish_line = None
        
        # Add adaptive evolution parameters
        self.adaptive_evolution = True
        self.generation_timeout_base = 1000  # Base timeout value
        self.min_generation_time = 200  # Minimum frames before allowing evolution
        self.early_termination_threshold = 0.8  # Terminate generation early if 80% cars died/finished
        
        # Fix initialization issue: active_slider not defined
        self.active_slider = None
    
    def restart_simulation(self):
        # Re-initialize population
        start_angle = 0  # Facing right
        self.ga.initialize_population(self.start_point[0], self.start_point[1], start_angle)
        self.ga.generation = 0
        self.ga.best_fitness = 0
        self.ga.best_car = None
        self.generation_start_time = time.time()
        self.generation_time = 0
        self.best_trajectory = []
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                elif event.key == K_SPACE:
                    self.paused = not self.paused
                elif event.key == K_e:
                    self.editing = not self.editing
                    if not self.editing:
                        self.restart_simulation()
                # 新增键盘快捷键
                elif event.key == K_PLUS or event.key == K_EQUALS:  # + 键加速
                    self.increase_simulation_speed()
                elif event.key == K_MINUS:  # - 键减速
                    self.decrease_simulation_speed()
                elif event.key == K_t:  # t 键切换轨迹显示
                    self.show_best_trajectory = not self.show_best_trajectory
                elif event.key == K_c:  # c 键清除轨迹
                    self.best_trajectory = []
            
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Left button
                    mouse_pos = pygame.mouse.get_pos()
                    
                    # Check button clicks
                    for button in self.buttons:
                        if button["rect"].collidepoint(mouse_pos):
                            self.handle_button_action(button["action"])
                            break
                    else:
                        # Check slider clicks
                        for i, slider in enumerate(self.sliders):
                            slider_rect = pygame.Rect(slider["x"], slider["y"], slider["width"], 20)
                            if slider_rect.collidepoint(mouse_pos):
                                self.active_slider = i
                                # Update slider value
                                self.update_slider_value(i, mouse_pos[0])
                                break
                        else:
                            # Handle editor clicks
                            if self.editing:
                                if self.edit_mode == "wall":
                                    if self.temp_wall is None:
                                        self.temp_wall = (mouse_pos[0], mouse_pos[1])
                                    else:
                                        self.walls.append((self.temp_wall[0], self.temp_wall[1], 
                                                          mouse_pos[0], mouse_pos[1]))
                                        self.temp_wall = None
                                elif self.edit_mode == "checkpoint":
                                    # Add checkpoint (x, y, radius)
                                    self.checkpoints.append((mouse_pos[0], mouse_pos[1], 20))
                                elif self.edit_mode == "start":
                                    self.start_point = (mouse_pos[0], mouse_pos[1])
                                elif self.edit_mode == "finish":
                                    # Setting finish line (two-point line like walls)
                                    if self.temp_wall is None:
                                        self.temp_wall = (mouse_pos[0], mouse_pos[1])
                                    else:
                                        # Create finish line
                                        self.finish_line = (self.temp_wall[0], self.temp_wall[1], 
                                                           mouse_pos[0], mouse_pos[1])
                                        self.temp_wall = None
            
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:  # Left button release
                    self.active_slider = None
            
            elif event.type == MOUSEMOTION:
                # If dragging a slider
                if self.active_slider is not None:
                    self.update_slider_value(self.active_slider, event.pos[0])
    
    def update_slider_value(self, slider_idx, x_pos):
        slider = self.sliders[slider_idx]
        
        # Calculate value corresponding to slider position
        ratio = max(0, min(1, (x_pos - slider["x"]) / slider["width"]))
        value_range = slider["max"] - slider["min"]
        
        if "step" in slider:
            # Discrete value
            steps = value_range / slider["step"]
            step_idx = round(ratio * steps)
            value = slider["min"] + step_idx * slider["step"]
        else:
            # Continuous value
            value = slider["min"] + ratio * value_range
        
        # Update slider value
        slider["value"] = value
        
        # Update genetic algorithm parameters
        if slider_idx == 0:
            self.ga.mutation_rate = value
        elif slider_idx == 1:
            self.ga.crossover_rate = value
        elif slider_idx == 2:
            self.ga.population_size = int(value)
        elif slider_idx == 3:
            self.ga.elite_size = int(value)
    
    def handle_button_action(self, action):
        if action == "wall_mode":
            self.edit_mode = "wall"
            self.temp_wall = None
        elif action == "checkpoint_mode":
            self.edit_mode = "checkpoint"
        elif action == "set_start":
            self.edit_mode = "start"
        elif action == "set_finish":
            self.edit_mode = "finish"
        elif action == "toggle_pause":
            self.paused = not self.paused
        elif action == "restart":
            self.restart_simulation()
        elif action == "toggle_view":
            self.show_all_cars = not self.show_all_cars
        elif action == "speed_up":
            self.increase_simulation_speed()
        elif action == "slow_down":
            self.decrease_simulation_speed()
        elif action == "toggle_trajectory":
            self.show_best_trajectory = not self.show_best_trajectory
        elif action == "clear_trajectory":
            self.best_trajectory = []
        elif action == "preset_simple" or action == "preset_figure8" or action == "preset_race":
            # Automatically enter simulation mode after selecting a preset track
            self.editing = False
            if action == "preset_simple":
                self.create_simple_track()
            elif action == "preset_figure8":
                self.create_figure8_track()
            elif action == "preset_race":
                self.create_race_track()
            
            # Clear previous best trajectory when loading a new track
            self.best_trajectory = []
            self.best_path = []
            self.best_completion_time = float('inf')
            self.best_path_length = float('inf')
        elif action == "clear_track":
            self.walls = []
            self.checkpoints = []
            self.finish_line = None
            self.start_point = (100, 400)
            self.best_path = []
            self.best_trajectory = []
            self.best_completion_time = float('inf')
            self.best_path_length = float('inf')
        elif action == "save_track":
            # Implement saving track to file
            try:
                import json
                import os
                
                # Create track data dictionary
                track_data = {
                    "walls": self.walls,
                    "checkpoints": self.checkpoints,
                    "start_point": self.start_point,
                    "finish_line": self.finish_line
                }
                
                # Create tracks directory if it doesn't exist
                if not os.path.exists("tracks"):
                    os.makedirs("tracks")
                
                # Save to file
                with open(f"tracks/track_{time.strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
                    json.dump(track_data, f)
            except Exception as e:
                print(f"Error saving track: {e}")
        elif action == "load_track":
            # Implement loading track from file
            try:
                import json
                import os
                import tkinter as tk
                from tkinter import filedialog
                
                # Create minimal tkinter root window
                root = tk.Tk()
                root.withdraw()
                
                # Open file dialog
                filepath = filedialog.askopenfilename(
                    initialdir="tracks",
                    title="Select Track File",
                    filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
                )
                
                if filepath:
                    with open(filepath, "r") as f:
                        track_data = json.load(f)
                    
                    # Load track data
                    self.walls = track_data.get("walls", [])
                    self.checkpoints = track_data.get("checkpoints", [])
                    self.start_point = track_data.get("start_point", (100, 400))
                    self.finish_line = track_data.get("finish_line", None)
                    
                    # Reset best paths
                    self.best_path = []
                    self.best_trajectory = []
                    self.best_completion_time = float('inf')
                    self.best_path_length = float('inf')
            except Exception as e:
                print(f"Error loading track: {e}")
    
    def increase_simulation_speed(self):
        self.simulation_speed = min(self.simulation_speed + 1, self.max_simulation_speed)
    
    def decrease_simulation_speed(self):
        self.simulation_speed = max(1, self.simulation_speed - 1)
    
    def update(self):
        if not self.paused and not self.editing:
            # Add visualization of evolution progress
            generation_data = []
            if not hasattr(self, 'fitness_history'):
                self.fitness_history = []
            
            # Reset stats for this generation
            if self.cars_finished == 0:  # Only reset at beginning of simulation step
                self.cars_finished = 0
                self.best_time_this_gen = float('inf')
            
            # Optimization: Use adaptive simulation speed
            effective_speed = self.simulation_speed
            if self.adaptive_evolution and self.ga.generation > 5:
                # Increase effective speed for more advanced generations
                effective_speed = min(self.simulation_speed + 2, self.max_simulation_speed)
            
            # Run multiple updates based on simulation speed
            for _ in range(effective_speed):
                # Track cars still racing and their status
                cars_still_racing = 0
                total_cars = len(self.ga.population)
                dead_or_finished_cars = 0
                
                for car in self.ga.population:
                    # Only update cars that are still racing
                    if car.alive and not car.reached_goal:
                        cars_still_racing += 1
                        car.update(self.walls, self.checkpoints, self.finish_line)
                        
                        # Track cars that reach the finish line
                        if car.reached_goal:
                            self.cars_finished += 1
                            dead_or_finished_cars += 1
                            if car.completion_time < self.best_time_this_gen:
                                self.best_time_this_gen = car.completion_time
                            
                            # Check for all-time best performance
                            is_better = False
                            if car.completion_time < self.best_completion_time:
                                is_better = True
                            elif car.completion_time == self.best_completion_time and car.path_length < self.best_path_length:
                                is_better = True
                                
                            if is_better:
                                self.best_completion_time = car.completion_time
                                self.best_path_length = car.path_length
                                self.best_path = car.best_path.copy()
                    else:
                        # Car is already dead or finished
                        dead_or_finished_cars += 1
                
                # Adaptive timeout based on generation
                adaptive_timeout = max(self.min_generation_time, 
                                      self.generation_timeout_base // (1 + min(self.ga.generation//5, 5)))
                
                # Smart termination conditions
                time_limit_reached = self.ga.population[0].time_alive > adaptive_timeout
                early_termination = (dead_or_finished_cars / total_cars) > self.early_termination_threshold and self.ga.population[0].time_alive > self.min_generation_time
                no_cars_racing = cars_still_racing == 0
                
                # Evolution decision
                if no_cars_racing or time_limit_reached or early_termination:
                    # Record fitness statistics for this generation
                    current_gen_stats = {
                        'generation': self.ga.generation,
                        'best_fitness': self.ga.best_fitness,
                        'avg_fitness': sum(car.fitness for car in self.ga.population) / len(self.ga.population),
                        'cars_finished': self.cars_finished,
                        'completion_time': self.best_time_this_gen if self.best_time_this_gen < float('inf') else None
                    }
                    self.fitness_history.append(current_gen_stats)
                    
                    # Ensure we keep only recent history to avoid memory issues
                    if len(self.fitness_history) > 100:
                        self.fitness_history = self.fitness_history[-100:]
                    
                    # Save best car's trajectory before evolving
                    if self.ga.best_car and self.ga.best_car.reached_goal:
                        self.best_trajectory = self.ga.best_car.position_history.copy()
                    
                    # Update generation time statistics
                    self.generation_time = time.time() - self.generation_start_time
                    self.generation_start_time = time.time()
                    
                    # Evolve to next generation with optimized parameters
                    if self.ga.generation > 10 and self.cars_finished / total_cars < 0.3:
                        # If few cars finish after many generations, increase mutation rate temporarily
                        original_mutation = self.ga.mutation_rate
                        self.ga.mutation_rate = min(0.3, self.ga.mutation_rate * 1.5)
                        self.ga.evolve(self.start_point[0], self.start_point[1], 0, self.walls, self.checkpoints)
                        self.ga.mutation_rate = original_mutation
                    else:
                        # Normal evolution
                        self.ga.evolve(self.start_point[0], self.start_point[1], 0, self.walls, self.checkpoints)
                    
                    # Reset counters for next generation
                    self.cars_finished = 0
                    break
            
            # Update total runtime
            self.total_runtime = time.time() - self.start_time
    
    def draw(self):
        # Fill main simulation area with white
        self.screen.fill(WHITE, self.simulation_rect)
        
        # Fill sidebar with a light gray to differentiate it
        self.screen.fill((240, 240, 240), self.sidebar_rect)
        
        # Draw separator line between sidebar and simulation area
        pygame.draw.line(self.screen, BLACK, 
                         (self.simulation_rect.width, 0), 
                         (self.simulation_rect.width, SCREEN_HEIGHT), 2)
        
        # Draw track elements in the simulation area
        # Draw walls
        for wall in self.walls:
            pygame.draw.line(self.screen, BLACK, (wall[0], wall[1]), (wall[2], wall[3]), 2)
        
        # Draw temporary wall
        if self.temp_wall is not None:
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.line(self.screen, GRAY, self.temp_wall, mouse_pos, 2)
        
        # Draw checkpoints
        for i, checkpoint in enumerate(self.checkpoints):
            pygame.draw.circle(self.screen, (0, 100, 255), (int(checkpoint[0]), int(checkpoint[1])), checkpoint[2], 1)
            # Draw number
            text = self.font.render(str(i), True, (0, 100, 255))
            self.screen.blit(text, (checkpoint[0] - 5, checkpoint[1] - 8))
        
        # Draw start point
        pygame.draw.circle(self.screen, GREEN, self.start_point, 10)
        pygame.draw.circle(self.screen, BLACK, self.start_point, 10, 1)
        
        # Draw finish line if it exists
        if self.finish_line:
            pygame.draw.line(self.screen, FINISH_COLOR, 
                             (self.finish_line[0], self.finish_line[1]),
                             (self.finish_line[2], self.finish_line[3]), 4)
            # Draw little flags on both ends
            pygame.draw.polygon(self.screen, FINISH_COLOR, [
                (self.finish_line[0], self.finish_line[1]),
                (self.finish_line[0] - 10, self.finish_line[1] - 5),
                (self.finish_line[0] - 10, self.finish_line[1] + 5)
            ])
            pygame.draw.polygon(self.screen, FINISH_COLOR, [
                (self.finish_line[2], self.finish_line[3]),
                (self.finish_line[2] - 10, self.finish_line[3] - 5),
                (self.finish_line[2] - 10, self.finish_line[3] + 5)
            ])
        
        # Draw best path if enabled
        if self.show_best_trajectory and self.best_path and len(self.best_path) > 1:
            for i in range(1, len(self.best_path)):
                pygame.draw.line(
                    self.screen,
                    PURPLE,  # Best all-time path in purple
                    (int(self.best_path[i-1][0]), int(self.best_path[i-1][1])),
                    (int(self.best_path[i][0]), int(self.best_path[i][1])),
                    2
                )
            
            # Mark the best path end
            if len(self.best_path) > 0:
                end_point = self.best_path[-1]
                pygame.draw.circle(self.screen, PURPLE, (int(end_point[0]), int(end_point[1])), 7, 2)
        
        # Draw cars with trajectories if enabled
        if not self.editing:
            if self.show_all_cars:
                for car in self.ga.population:
                    car.draw(self.screen, self.show_best_trajectory)
            else:
                # Only show best car
                if self.ga.best_car:
                    self.ga.best_car.draw(self.screen, True)  # Always show trajectory for best car
                else:
                    self.ga.population[0].draw(self.screen, True)
        
        # Draw UI elements in sidebar
        # Draw group headers
        headers = {
            "edit": "Track Editor",
            "sim": "Simulation Control",
            "speed": "Speed Control",
            "presets": "Preset Tracks",
            "params": "Genetic Parameters"
        }
        
        current_group = None
        for button in self.buttons:
            # Draw group header if this is a new group
            if button["group"] != current_group:
                current_group = button["group"]
                if current_group in headers:
                    header_y = button["rect"].y - 20
                    header_text = self.font.render(headers[current_group], True, BLACK)
                    self.screen.blit(header_text, (self.sidebar_rect.x + 10, header_y))
                    pygame.draw.line(self.screen, BLACK, 
                                   (self.sidebar_rect.x + 10, header_y + 18), 
                                   (self.sidebar_rect.right - 10, header_y + 18), 1)
            
            # Highlight active mode button
            button_color = GRAY
            if (button["action"] == "wall_mode" and self.edit_mode == "wall") or \
               (button["action"] == "checkpoint_mode" and self.edit_mode == "checkpoint") or \
               (button["action"] == "set_start" and self.edit_mode == "start") or \
               (button["action"] == "set_finish" and self.edit_mode == "finish") or \
               (button["action"] == "toggle_pause" and not self.paused and not self.editing) or \
               (button["action"] == "toggle_trajectory" and self.show_best_trajectory):
                button_color = (160, 200, 160)  # Light green for active buttons
            
            # Draw button
            pygame.draw.rect(self.screen, button_color, button["rect"])
            pygame.draw.rect(self.screen, BLACK, button["rect"], 1)
            text = self.font.render(button["text"], True, BLACK)
            text_rect = text.get_rect(center=button["rect"].center)
            self.screen.blit(text, text_rect)
        
        # Draw sliders header
        slider_header_y = self.sliders[0]["y"] - 20
        slider_header = self.font.render(headers["params"], True, BLACK)
        self.screen.blit(slider_header, (self.sidebar_rect.x + 10, slider_header_y))
        pygame.draw.line(self.screen, BLACK, 
                       (self.sidebar_rect.x + 10, slider_header_y + 18), 
                       (self.sidebar_rect.right - 10, slider_header_y + 18), 1)
        
        # Draw sliders
        for slider in self.sliders:
            # Slider text
            text = self.font.render(f"{slider['text']}", True, BLACK)
            self.screen.blit(text, (slider["x"], slider["y"] - 20))
            
            # Slider value
            value_text = self.font.render(f"{slider['value']:.2f}", True, BLACK)
            value_rect = value_text.get_rect(right=slider["x"] + slider["width"])
            self.screen.blit(value_text, (value_rect.x, slider["y"] - 20))
            
            # Slider background
            pygame.draw.rect(self.screen, GRAY, (slider["x"], slider["y"], slider["width"], 10))
            
            # Slider position
            ratio = (slider["value"] - slider["min"]) / (slider["max"] - slider["min"])
            handle_pos = slider["x"] + ratio * slider["width"]
            pygame.draw.circle(self.screen, BLACK, (int(handle_pos), slider["y"] + 5), 8)
        
        # Draw stats at the bottom of simulation area
        stats_y = self.simulation_rect.height - 90
        
        if not self.editing:
            # Draw status panel background
            stats_rect = pygame.Rect(0, stats_y - 10, self.simulation_rect.width, 100)
            pygame.draw.rect(self.screen, (240, 240, 240), stats_rect)
            pygame.draw.line(self.screen, BLACK, (0, stats_y - 10), (self.simulation_rect.width, stats_y - 10), 1)
            
            # Left column: Generation info
            self.screen.blit(self.font.render(f"Generation: {self.ga.generation}", True, BLACK), (20, stats_y))
            self.screen.blit(self.font.render(f"Best Fitness: {self.ga.best_fitness:.2f}", True, BLACK), (20, stats_y + 25))
            self.screen.blit(self.font.render(f"Cars Finished: {self.cars_finished}/{len(self.ga.population)}", True, BLACK), (20, stats_y + 50))
            
            # Middle column: Performance info
            middle_x = self.simulation_rect.width // 3
            if self.best_completion_time < float('inf'):
                self.screen.blit(self.font.render(f"Best Time: {self.best_completion_time} frames", True, BLACK), (middle_x, stats_y))
                self.screen.blit(self.font.render(f"Path Length: {self.best_path_length:.1f} px", True, BLACK), (middle_x, stats_y + 25))
            
            # Right column: Time info
            right_x = 2 * self.simulation_rect.width // 3
            self.screen.blit(self.font.render(f"Total Runtime: {self.total_runtime:.1f} sec", True, BLACK), (right_x, stats_y))
            self.screen.blit(self.font.render(f"Last Gen Time: {self.generation_time:.2f} sec", True, BLACK), (right_x, stats_y + 25))
            self.screen.blit(self.font.render(f"Sim Speed: {self.simulation_speed}x", True, BLACK), (right_x, stats_y + 50))
            
            # Add generation progress bar
            if not self.editing and not self.paused:
                # Calculate progress as percentage of timeout
                progress_width = 150
                progress_height = 15
                
                # Adaptive timeout calculation
                adaptive_timeout = max(self.min_generation_time, 
                                      self.generation_timeout_base // (1 + min(self.ga.generation//5, 5)))
                
                progress = min(1.0, self.ga.population[0].time_alive / adaptive_timeout)
                
                # Draw progress bar background
                progress_bar_bg = pygame.Rect(20, stats_y - 30, progress_width, progress_height)
                pygame.draw.rect(self.screen, (200, 200, 200), progress_bar_bg, 0, 5)
                
                # Draw progress bar fill
                progress_bar = pygame.Rect(20, stats_y - 30, int(progress_width * progress), progress_height)
                progress_color = (100, 200, 100)  # Green normally
                
                # Change color when approaching timeout
                if progress > 0.8:
                    progress_color = (200, 200, 100)  # Yellow when nearly done
                
                pygame.draw.rect(self.screen, progress_color, progress_bar, 0, 5)
                
                # Add label
                progress_text = f"Gen {self.ga.generation}: {int(progress*100)}%"
                progress_label = self.font.render(progress_text, True, BLACK)
                self.screen.blit(progress_label, (progress_width + 30, stats_y - 30))
            
            # Add a "waiting for evolution" indicator when needed
            if not self.editing and not self.paused and self.ga.generation > 0:
                cars_alive = sum(1 for car in self.ga.population if car.alive and not car.reached_goal)
                if cars_alive == 0:
                    evolving_text = self.font.render("EVOLVING TO NEXT GENERATION...", True, (200, 50, 50))
                    text_rect = evolving_text.get_rect(center=(self.simulation_rect.width//2, SCREEN_HEIGHT//2))
                    
                    # Add background for better visibility
                    bg_rect = text_rect.copy()
                    bg_rect.inflate_ip(20, 10)
                    pygame.draw.rect(self.screen, (240, 240, 240), bg_rect)
                    pygame.draw.rect(self.screen, BLACK, bg_rect, 1)
                    
                    self.screen.blit(evolving_text, text_rect)
        else:
            # Edit mode info
            mode_text = self.font.render(f"Edit Mode: {self.edit_mode.capitalize()}", True, BLACK)
            self.screen.blit(mode_text, (20, self.simulation_rect.height - 30))
            
            # Help text
            help_text = self.font.render("Press E to exit edit mode and start simulation", True, BLACK)
            self.screen.blit(help_text, (250, self.simulation_rect.height - 30))
        
        # Draw evolution progress graph if we have history
        if not self.editing and hasattr(self, 'fitness_history') and len(self.fitness_history) > 1:
            # Set up the graph area
            graph_rect = pygame.Rect(10, 100, 300, 150)
            graph_margin = 20
            graph_width = graph_rect.width - 2 * graph_margin
            graph_height = graph_rect.height - 2 * graph_margin
            
            # Draw background
            pygame.draw.rect(self.screen, (240, 240, 240), graph_rect)
            pygame.draw.rect(self.screen, BLACK, graph_rect, 1)
            
            # Draw title
            title = self.font.render("Fitness Across Generations", True, BLACK)
            self.screen.blit(title, (graph_rect.x + (graph_rect.width - title.get_width()) // 2, graph_rect.y + 5))
            
            # Calculate scale based on data
            max_fitness = max(stats['best_fitness'] for stats in self.fitness_history)
            if max_fitness <= 0:
                max_fitness = 1  # Avoid division by zero
            
            # Draw axis labels
            y_label = self.font.render(f"Max: {max_fitness:.0f}", True, BLACK)
            self.screen.blit(y_label, (graph_rect.x + 5, graph_rect.y + graph_margin))
            
            x_label = self.font.render(f"Generations: {len(self.fitness_history)}", True, BLACK)
            self.screen.blit(x_label, (graph_rect.right - x_label.get_width() - 5, graph_rect.bottom - 20))
            
            # Plot the best fitness line
            points = []
            for i, stats in enumerate(self.fitness_history):
                x = graph_rect.x + graph_margin + (i / (len(self.fitness_history)-1)) * graph_width if len(self.fitness_history) > 1 else 0
                y = graph_rect.bottom - graph_margin - (stats['best_fitness'] / max_fitness) * graph_height
                points.append((x, y))
            
            if len(points) > 1:
                # Draw connecting lines
                pygame.draw.lines(self.screen, (200, 0, 0), False, points, 2)
                
                # Add dots for each generation
                for point in points:
                    pygame.draw.circle(self.screen, (200, 0, 0), (int(point[0]), int(point[1])), 3)
            
            # Also plot average fitness
            avg_points = []
            for i, stats in enumerate(self.fitness_history):
                x = graph_rect.x + graph_margin + (i / (len(self.fitness_history)-1)) * graph_width if len(self.fitness_history) > 1 else 0
                y = graph_rect.bottom - graph_margin - (stats['avg_fitness'] / max_fitness) * graph_height
                avg_points.append((x, y))
            
            if len(avg_points) > 1:
                pygame.draw.lines(self.screen, (0, 100, 200), False, avg_points, 2)
        
        pygame.display.flip()
    
    def run(self):
        self.start_time = time.time()  # 记录开始时间
        while self.running:
            # 根据模拟速度调整帧率限制
            target_fps = FPS * min(2, self.simulation_speed)  # 在高速模式下提高帧率上限
            self.clock.tick(target_fps)
            
            self.handle_events()
            self.update()
            self.draw()
        
        pygame.quit()
        sys.exit()

    def create_simple_track(self):
        # Reset current track
        self.walls = []
        self.checkpoints = []
        self.finish_line = None
        
        # Create a simple oval track with proper width
        center_x = self.simulation_rect.width // 2
        center_y = SCREEN_HEIGHT // 2
        width = min(500, self.simulation_rect.width - 100)  # Adjust width to available space
        height = 300
        track_width = 80  # Wider track for easier navigation
        
        # Start point - clear position on left side
        self.start_point = (center_x - width//2 + 50, center_y)
        
        # Calculate the path points for smoother corners
        num_points = 40
        oval_points = []
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            # Oval equation
            x = center_x + (width//2 - track_width//2) * math.cos(angle)
            y = center_y + (height//2 - track_width//2) * math.sin(angle)
            oval_points.append((x, y))
        
        # Create outer and inner walls
        for i in range(num_points):
            # Outer walls
            p1 = oval_points[i]
            p2 = oval_points[(i + 1) % num_points]
            self.walls.append((p1[0], p1[1], p2[0], p2[1]))
            
            # Inner walls - smaller oval
            inner_x1 = center_x + (width//2 - track_width*1.5) * math.cos(2 * math.pi * i / num_points)
            inner_y1 = center_y + (height//2 - track_width*1.5) * math.sin(2 * math.pi * i / num_points)
            inner_x2 = center_x + (width//2 - track_width*1.5) * math.cos(2 * math.pi * (i + 1) / num_points)
            inner_y2 = center_y + (height//2 - track_width*1.5) * math.sin(2 * math.pi * (i + 1) / num_points)
            self.walls.append((inner_x1, inner_y1, inner_x2, inner_y2))
        
        # Add strategic checkpoints for better guidance - evenly spaced
        for i in range(4):  # 4 checkpoints around the track
            angle = 2 * math.pi * i / 4
            checkpoint_x = center_x + (width//2 - track_width) * math.cos(angle)
            checkpoint_y = center_y + (height//2 - track_width) * math.sin(angle)
            self.checkpoints.append((checkpoint_x, checkpoint_y, 20))
        
        # Add finish line at right side (3/4 of the way around from start)
        finish_angle = 3 * math.pi / 2  # Left side
        finish_x = center_x + (width//2 - track_width//2) * math.cos(finish_angle)
        
        # Calculate normal vector for proper finish line orientation
        normal_x = -math.sin(finish_angle)
        normal_y = math.cos(finish_angle)
        
        # Place finish line with proper orientation
        self.finish_line = (
            finish_x - normal_x * track_width//2, 
            center_y + normal_y * track_width//2,
            finish_x + normal_x * track_width//2, 
            center_y - normal_y * track_width//2
        )
        
        # Restart simulation after creating track
        self.restart_simulation()
    
    def create_figure8_track(self):
        # Reset current track
        self.walls = []
        self.checkpoints = []
        self.finish_line = None
        
        # Create a figure-8 track
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        radius = 150
        inner_radius = radius - 60
        
        # Start point - left side of left circle
        self.start_point = (center_x - radius - 20, center_y)
        
        # Create smoother circles with more points
        segments = 36  # 10° segments
        
        # Left circle (outer)
        left_center = (center_x - radius, center_y)
        for i in range(segments):
            angle1 = 2 * math.pi * i / segments
            angle2 = 2 * math.pi * (i + 1) / segments
            x1 = left_center[0] + math.cos(angle1) * radius
            y1 = left_center[1] + math.sin(angle1) * radius
            x2 = left_center[0] + math.cos(angle2) * radius
            y2 = left_center[1] + math.sin(angle2) * radius
            self.walls.append((x1, y1, x2, y2))
        
        # Right circle (outer)
        right_center = (center_x + radius, center_y)
        for i in range(segments):
            angle1 = 2 * math.pi * i / segments
            angle2 = 2 * math.pi * (i + 1) / segments
            x1 = right_center[0] + math.cos(angle1) * radius
            y1 = right_center[1] + math.sin(angle1) * radius
            x2 = right_center[0] + math.cos(angle2) * radius
            y2 = right_center[1] + math.sin(angle2) * radius
            self.walls.append((x1, y1, x2, y2))
        
        # Left circle (inner)
        for i in range(segments):
            angle1 = 2 * math.pi * i / segments
            angle2 = 2 * math.pi * (i + 1) / segments
            x1 = left_center[0] + math.cos(angle1) * inner_radius
            y1 = left_center[1] + math.sin(angle1) * inner_radius
            x2 = left_center[0] + math.cos(angle2) * inner_radius
            y2 = left_center[1] + math.sin(angle2) * inner_radius
            self.walls.append((x1, y1, x2, y2))
        
        # Right circle (inner)
        for i in range(segments):
            angle1 = 2 * math.pi * i / segments
            angle2 = 2 * math.pi * (i + 1) / segments
            x1 = right_center[0] + math.cos(angle1) * inner_radius
            y1 = right_center[1] + math.sin(angle1) * inner_radius
            x2 = right_center[0] + math.cos(angle2) * inner_radius
            y2 = right_center[1] + math.sin(angle2) * inner_radius
            self.walls.append((x1, y1, x2, y2))
        
        # Add checkpoints around the track in sequence
        checkpoint_radius = 20
        # Top half
        self.checkpoints.append((center_x - radius, center_y - radius//2, checkpoint_radius))
        self.checkpoints.append((center_x, center_y - radius//2, checkpoint_radius))
        self.checkpoints.append((center_x + radius, center_y - radius//2, checkpoint_radius))
        # Bottom half
        self.checkpoints.append((center_x + radius, center_y + radius//2, checkpoint_radius))
        self.checkpoints.append((center_x, center_y + radius//2, checkpoint_radius))
        self.checkpoints.append((center_x - radius, center_y + radius//2, checkpoint_radius))
        
        # Add finish line - place it on the right side of the right circle
        # This forces cars to complete nearly a full figure-8 before finishing
        self.finish_line = (center_x + radius * 2 - 20, center_y - 40, center_x + radius * 2 - 20, center_y + 40)
        
        # Restart simulation after creating track
        self.restart_simulation()
    
    def create_race_track(self):
        # Reset current track
        self.walls = []
        self.checkpoints = []
        self.finish_line = None
        
        # Create a more balanced race track
        margin = 70
        track_width = 100  # Wider track for easier navigation
        
        # Adjust to available space
        available_width = self.simulation_rect.width - 2*margin
        available_height = SCREEN_HEIGHT - 2*margin
        
        # Define track centerline points
        points = [
            (margin, SCREEN_HEIGHT//2),                           # Start
            (margin + available_width//4, SCREEN_HEIGHT//2),      # First straight
            (margin + available_width//2, margin + available_height//4),  # First curve
            (margin + 3*available_width//4, margin + available_height//4),  # Second straight
            (margin + available_width - track_width//2, SCREEN_HEIGHT//2),  # Second curve
            (margin + 3*available_width//4, margin + 3*available_height//4),  # Third curve
            (margin + available_width//4, margin + 3*available_height//4),  # Fourth straight
            (margin, SCREEN_HEIGHT//2)                           # Back to start
        ]
        
        # Set start point at the beginning of the track
        self.start_point = (margin + 40, SCREEN_HEIGHT//2)
        
        # Generate smooth corner points
        smooth_points = []
        
        # For each segment, add intermediate points for smoother curves
        for i in range(len(points)-1):
            p1 = points[i]
            p2 = points[i+1]
            
            # Add the current point
            smooth_points.append(p1)
            
            # If moving to a corner, add intermediate points for smoother curve
            if (i == 0 or i == 2 or i == 4 or i == 6):
                # Add 5 intermediate points
                for j in range(1, 6):
                    t = j / 6.0
                    # Quadratic Bezier curve for smoother corners
                    if i == 0:  # First straight to first curve
                        control_x = p1[0] + (p2[0] - p1[0])
                        control_y = p1[1]
                    elif i == 2:  # First curve to second straight
                        control_x = p2[0]
                        control_y = p1[1]
                    elif i == 4:  # Second curve to third curve
                        control_x = p1[0]
                        control_y = p2[1]
                    else:  # Fourth straight back to start
                        control_x = p1[0]
                        control_y = p1[1] + (p2[1] - p1[1])
                    
                    # Interpolate point along bezier curve
                    interp_x = (1-t)*(1-t)*p1[0] + 2*(1-t)*t*control_x + t*t*p2[0]
                    interp_y = (1-t)*(1-t)*p1[1] + 2*(1-t)*t*control_y + t*t*p2[1]
                    smooth_points.append((interp_x, interp_y))
        
        # Add the last point to close the loop
        smooth_points.append(points[-1])
        
        # Create walls by offsetting centerline
        def get_track_edges(p1, p2, width):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length < 0.0001:
                return None
            
            # Normal vector (perpendicular)
            nx = dy / length
            ny = -dx / length
            
            # Create the two edges of the track
            edge1 = (p1[0] + nx * width/2, p1[1] + ny * width/2)
            edge2 = (p1[0] - nx * width/2, p1[1] - ny * width/2)
            
            return edge1, edge2
        
        # Generate the track walls
        inner_points = []
        outer_points = []
        
        for i in range(len(smooth_points)-1):
            p1 = smooth_points[i]
            p2 = smooth_points[i+1]
            
            edges = get_track_edges(p1, p2, track_width)
            if edges:
                outer_edge, inner_edge = edges
                outer_points.append(outer_edge)
                inner_points.append(inner_edge)
        
        # Create the walls from the edge points
        for i in range(len(outer_points)-1):
            # Outer wall
            self.walls.append((outer_points[i][0], outer_points[i][1], 
                              outer_points[i+1][0], outer_points[i+1][1]))
            # Inner wall
            self.walls.append((inner_points[i][0], inner_points[i][1], 
                              inner_points[i+1][0], inner_points[i+1][1]))
        
        # Add checkpoints along the track - more frequent for better guidance
        checkpoint_count = 8
        for i in range(checkpoint_count):
            idx = (i * len(smooth_points)) // checkpoint_count
            if idx < len(smooth_points):
                p = smooth_points[idx]
                self.checkpoints.append((p[0], p[1], 20))
        
        # Add finish line at 3/4 of the track
        finish_idx = (3 * len(smooth_points)) // 4
        if finish_idx < len(smooth_points)-1:
            p1 = smooth_points[finish_idx]
            p2 = smooth_points[finish_idx+1]
            edges = get_track_edges(p1, p2, track_width*1.2)  # Slightly wider
            if edges:
                outer_edge, inner_edge = edges
                self.finish_line = (
                    outer_edge[0], outer_edge[1],
                    inner_edge[0], inner_edge[1]
                )
        
        # Restart simulation after creating track
        self.restart_simulation()

if __name__ == "__main__":
    game = Game()
    game.run() 