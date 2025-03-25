import pygame
import numpy as np
import sys
from pygame.locals import *
import math
import random

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)

# 游戏参数
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# 基本小车参数
CAR_WIDTH = 20
CAR_HEIGHT = 10
SENSOR_LENGTH = 150
NUM_SENSORS = 5

class Car:
    def __init__(self, x, y, angle, brain=None):
        self.x = x
        self.y = y
        self.angle = angle  # 角度，0表示向右
        self.speed = 0
        self.max_speed = 5
        self.acceleration = 0.1
        self.turn_speed = 0.1
        
        # 传感器
        self.sensors = []
        self.sensor_values = [0] * NUM_SENSORS
        
        # 基因组 - 神经网络权重
        if brain is None:
            self.brain = np.random.uniform(-1, 1, (NUM_SENSORS + 1, 2))  # +1是偏置
        else:
            # 复制或变异现有大脑
            self.brain = brain.copy()
        
        # 适应度
        self.fitness = 0
        self.alive = True
        self.distance_traveled = 0
        self.time_alive = 0
        
        # 检查点
        self.next_checkpoint = 0
        self.checkpoints_passed = 0
    
    def think(self):
        # 简单神经网络:
        # 输入: 传感器值
        # 输出: [加速度, 转向]
        inputs = np.append(self.sensor_values, 1)  # 添加偏置
        outputs = np.dot(inputs, self.brain)
        
        # 使用tanh激活函数限制输出范围在-1到1之间
        outputs = np.tanh(outputs)
        
        # 应用输出
        self.speed += outputs[0] * self.acceleration
        self.speed = max(0, min(self.max_speed, self.speed))  # 限制速度范围
        self.angle += outputs[1] * self.turn_speed
    
    def update_sensors(self, walls):
        self.sensors = []
        self.sensor_values = []
        
        # 计算传感器角度
        angles = []
        for i in range(NUM_SENSORS):
            # 扇形分布传感器
            angle = self.angle + math.radians(-90 + 180 * i / (NUM_SENSORS - 1))
            angles.append(angle)
        
        # 计算每个传感器的碰撞
        for angle in angles:
            # 传感器起点
            start_x = self.x
            start_y = self.y
            
            # 传感器终点
            end_x = start_x + SENSOR_LENGTH * math.cos(angle)
            end_y = start_y + SENSOR_LENGTH * math.sin(angle)
            
            # 检查传感器与墙壁的碰撞
            closest_point = None
            min_distance = SENSOR_LENGTH
            
            for wall in walls:
                # 检测线段相交
                intersection = self.line_intersection(
                    (start_x, start_y), (end_x, end_y),
                    (wall[0], wall[1]), (wall[2], wall[3])
                )
                
                if intersection:
                    # 计算与交点的距离
                    dx = intersection[0] - start_x
                    dy = intersection[1] - start_y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = intersection
            
            # 保存传感器信息
            if closest_point:
                self.sensors.append((start_x, start_y, closest_point[0], closest_point[1]))
                # 归一化传感器值到0-1范围
                self.sensor_values.append(min_distance / SENSOR_LENGTH)
            else:
                self.sensors.append((start_x, start_y, end_x, end_y))
                self.sensor_values.append(1.0)  # 没有碰撞
    
    def update(self, walls, checkpoints):
        if not self.alive:
            return
            
        self.time_alive += 1
        
        # 思考并移动
        self.think()
        
        # 根据角度和速度更新位置
        dx = self.speed * math.cos(self.angle)
        dy = self.speed * math.sin(self.angle)
        self.x += dx
        self.y += dy
        
        # 计算移动距离
        self.distance_traveled += math.sqrt(dx*dx + dy*dy)
        
        # 更新传感器
        self.update_sensors(walls)
        
        # 检查碰撞
        if self.check_collision(walls):
            self.alive = False
        
        # 检查是否经过检查点
        if self.next_checkpoint < len(checkpoints):
            checkpoint = checkpoints[self.next_checkpoint]
            # 简单检查点：如果小车靠近检查点中心
            dx = self.x - checkpoint[0]
            dy = self.y - checkpoint[1]
            distance = math.sqrt(dx*dx + dy*dy)
            if distance < checkpoint[2]:  # checkpoint[2]是检查点半径
                self.next_checkpoint += 1
                self.checkpoints_passed += 1
        
        # 计算适应度
        self.fitness = (self.checkpoints_passed * 1000) + self.distance_traveled
        if not self.alive:
            self.fitness *= 0.8  # 对死亡的车辆适应度略有惩罚
    
    def check_collision(self, walls):
        # 简化为点碰撞检测
        for wall in walls:
            # 检查小车中心是否在离墙壁很近的距离内
            closest_point = self.closest_point_on_line(
                (self.x, self.y), 
                (wall[0], wall[1]), 
                (wall[2], wall[3])
            )
            
            if closest_point:
                dx = self.x - closest_point[0]
                dy = self.y - closest_point[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < (CAR_WIDTH + CAR_HEIGHT) / 4:  # 简化的碰撞半径
                    return True
        
        return False
    
    def draw(self, screen):
        # 绘制小车
        car_color = GREEN if self.alive else RED
        pygame.draw.circle(screen, car_color, (int(self.x), int(self.y)), 5)
        
        # 绘制方向指示线
        end_x = self.x + 15 * math.cos(self.angle)
        end_y = self.y + 15 * math.sin(self.angle)
        pygame.draw.line(screen, BLUE, (self.x, self.y), (end_x, end_y), 2)
        
        # 绘制传感器
        for sensor in self.sensors:
            pygame.draw.line(screen, YELLOW, (sensor[0], sensor[1]), (sensor[2], sensor[3]), 1)
    
    @staticmethod
    def line_intersection(line1_start, line1_end, line2_start, line2_end):
        # 线段相交检测
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
            
        A = line1_start
        B = line1_end
        C = line2_start
        D = line2_end
        
        # 快速检查线段是否相交
        if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
            # 计算交点
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
        # 计算点到线段的最近点
        x1, y1 = line_start
        x2, y2 = line_end
        x0, y0 = point
        
        # 向量方向
        dx = x2 - x1
        dy = y2 - y1
        
        # 如果线段退化为点
        if dx == 0 and dy == 0:
            return line_start
            
        # 计算投影位置
        t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
        
        # 限制在线段上
        t = max(0, min(1, t))
        
        # 计算最近点坐标
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
        # 所有车辆的适应度已在update方法中计算
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # 更新最佳适应度
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_car = self.population[0]
    
    def select_parents(self):
        # 使用锦标赛选择
        parents = []
        
        # 精英直接选择
        for i in range(self.elite_size):
            parents.append(self.population[i])
        
        # 剩余使用锦标赛选择
        for _ in range(self.population_size - self.elite_size):
            # 随机选择3个个体
            tournament = random.sample(self.population, 3)
            # 选择最佳的一个
            tournament.sort(key=lambda x: x.fitness, reverse=True)
            parents.append(tournament[0])
        
        return parents
    
    def crossover(self, parent1, parent2):
        # 单点交叉
        if random.random() < self.crossover_rate:
            # 创建子代
            child_brain = parent1.brain.copy()
            
            # 随机交叉点
            crossover_point = random.randint(0, parent1.brain.shape[0] * parent1.brain.shape[1] - 1)
            row = crossover_point // parent1.brain.shape[1]
            col = crossover_point % parent1.brain.shape[1]
            
            # 交换交叉点后的数据
            for r in range(row, parent1.brain.shape[0]):
                start_col = 0
                if r == row:
                    start_col = col
                    
                for c in range(start_col, parent1.brain.shape[1]):
                    child_brain[r, c] = parent2.brain[r, c]
            
            return child_brain
        else:
            # 不交叉，直接返回父代1的基因组
            return parent1.brain.copy()
    
    def mutate(self, brain):
        # 变异
        for i in range(brain.shape[0]):
            for j in range(brain.shape[1]):
                if random.random() < self.mutation_rate:
                    # 添加随机扰动
                    brain[i, j] += random.uniform(-0.5, 0.5)
                    # 限制在-1到1范围内
                    brain[i, j] = max(-1, min(1, brain[i, j]))
        
        return brain
    
    def evolve(self, start_x, start_y, start_angle, walls, checkpoints):
        # 评估当前种群
        self.evaluate_fitness(walls, checkpoints)
        
        # 选择父代
        parents = self.select_parents()
        
        # 创建下一代
        next_generation = []
        
        # 精英保留
        for i in range(self.elite_size):
            # 创建新车但保留原有大脑
            elite_car = Car(start_x, start_y, start_angle, parents[i].brain.copy())
            next_generation.append(elite_car)
        
        # 剩余通过交叉和变异产生
        while len(next_generation) < self.population_size:
            # 随机选择两个父代
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # 交叉
            child_brain = self.crossover(parent1, parent2)
            
            # 变异
            child_brain = self.mutate(child_brain)
            
            # 创建子代
            child = Car(start_x, start_y, start_angle, child_brain)
            next_generation.append(child)
        
        # 更新种群
        self.population = next_generation
        self.generation += 1
        
        return self.population

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("智能小车进化模拟器")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        
        # 游戏状态
        self.running = True
        self.paused = False
        self.editing = True  # 开始在编辑模式
        self.show_all_cars = True
        
        # 编辑器相关
        self.walls = []
        self.checkpoints = []
        self.start_point = (100, 400)
        self.edit_mode = "wall"  # 'wall' 或 'checkpoint'
        self.temp_wall = None
        
        # 遗传算法
        self.ga = GeneticAlgorithm(
            population_size=30,
            elite_size=5,
            mutation_rate=0.1,
            crossover_rate=0.7
        )
        
        # 初始化种群
        self.restart_simulation()
        
        # UI元素
        self.buttons = [
            {"rect": pygame.Rect(10, 10, 120, 30), "text": "墙壁模式", "action": "wall_mode"},
            {"rect": pygame.Rect(140, 10, 120, 30), "text": "检查点模式", "action": "checkpoint_mode"},
            {"rect": pygame.Rect(270, 10, 120, 30), "text": "设置起点", "action": "set_start"},
            {"rect": pygame.Rect(400, 10, 120, 30), "text": "开始/暂停", "action": "toggle_pause"},
            {"rect": pygame.Rect(530, 10, 120, 30), "text": "重置模拟", "action": "restart"},
            {"rect": pygame.Rect(660, 10, 120, 30), "text": "显示所有/最佳", "action": "toggle_view"},
        ]
        
        # 滑块控件(位置X, 位置Y, 宽度, 当前值, 最小值, 最大值, 文本)
        self.sliders = [
            {"x": 800, "y": 15, "width": 100, "value": self.ga.mutation_rate, "min": 0, "max": 0.5, "text": "变异率"},
            {"x": 800, "y": 45, "width": 100, "value": self.ga.crossover_rate, "min": 0, "max": 1, "text": "交叉率"},
            {"x": 800, "y": 75, "width": 100, "value": self.ga.population_size, "min": 10, "max": 100, "step": 5, "text": "种群大小"},
            {"x": 800, "y": 105, "width": 100, "value": self.ga.elite_size, "min": 1, "max": 10, "step": 1, "text": "精英数量"},
        ]
        self.active_slider = None
    
    def restart_simulation(self):
        # 重新初始化种群
        start_angle = 0  # 向右
        self.ga.initialize_population(self.start_point[0], self.start_point[1], start_angle)
        self.ga.generation = 0
        self.ga.best_fitness = 0
        self.ga.best_car = None
    
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
            
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键
                    mouse_pos = pygame.mouse.get_pos()
                    
                    # 检查按钮点击
                    for button in self.buttons:
                        if button["rect"].collidepoint(mouse_pos):
                            self.handle_button_action(button["action"])
                            break
                    else:
                        # 检查滑块点击
                        for i, slider in enumerate(self.sliders):
                            slider_rect = pygame.Rect(slider["x"], slider["y"], slider["width"], 20)
                            if slider_rect.collidepoint(mouse_pos):
                                self.active_slider = i
                                # 更新滑块值
                                self.update_slider_value(i, mouse_pos[0])
                                break
                        else:
                            # 处理编辑器点击
                            if self.editing:
                                if self.edit_mode == "wall":
                                    if self.temp_wall is None:
                                        self.temp_wall = (mouse_pos[0], mouse_pos[1])
                                    else:
                                        self.walls.append((self.temp_wall[0], self.temp_wall[1], 
                                                          mouse_pos[0], mouse_pos[1]))
                                        self.temp_wall = None
                                elif self.edit_mode == "checkpoint":
                                    # 添加检查点 (x, y, 半径)
                                    self.checkpoints.append((mouse_pos[0], mouse_pos[1], 20))
                                elif self.edit_mode == "start":
                                    self.start_point = (mouse_pos[0], mouse_pos[1])
                    
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:  # 左键释放
                    self.active_slider = None
            
            elif event.type == MOUSEMOTION:
                # 如果正在拖动滑块
                if self.active_slider is not None:
                    self.update_slider_value(self.active_slider, event.pos[0])
    
    def update_slider_value(self, slider_idx, x_pos):
        slider = self.sliders[slider_idx]
        
        # 计算滑块位置对应的值
        ratio = max(0, min(1, (x_pos - slider["x"]) / slider["width"]))
        value_range = slider["max"] - slider["min"]
        
        if "step" in slider:
            # 离散值
            steps = value_range / slider["step"]
            step_idx = round(ratio * steps)
            value = slider["min"] + step_idx * slider["step"]
        else:
            # 连续值
            value = slider["min"] + ratio * value_range
        
        # 更新滑块值
        slider["value"] = value
        
        # 更新遗传算法参数
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
        elif action == "toggle_pause":
            self.paused = not self.paused
        elif action == "restart":
            self.restart_simulation()
        elif action == "toggle_view":
            self.show_all_cars = not self.show_all_cars
    
    def update(self):
        if not self.paused and not self.editing:
            # 更新所有车辆
            all_dead = True
            for car in self.ga.population:
                if car.alive:
                    all_dead = False
                    car.update(self.walls, self.checkpoints)
            
            # 如果所有车辆都死亡或超过一定时间，进化到下一代
            if all_dead or self.ga.population[0].time_alive > 1000:
                self.ga.evolve(self.start_point[0], self.start_point[1], 0, self.walls, self.checkpoints)
    
    def draw(self):
        self.screen.fill(WHITE)
        
        # 绘制墙壁
        for wall in self.walls:
            pygame.draw.line(self.screen, BLACK, (wall[0], wall[1]), (wall[2], wall[3]), 2)
        
        # 绘制临时墙壁
        if self.temp_wall is not None:
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.line(self.screen, GRAY, self.temp_wall, mouse_pos, 2)
        
        # 绘制检查点
        for i, checkpoint in enumerate(self.checkpoints):
            pygame.draw.circle(self.screen, (0, 100, 255), (int(checkpoint[0]), int(checkpoint[1])), checkpoint[2], 1)
            # 绘制编号
            text = self.font.render(str(i), True, (0, 100, 255))
            self.screen.blit(text, (checkpoint[0] - 5, checkpoint[1] - 8))
        
        # 绘制起点
        pygame.draw.circle(self.screen, GREEN, self.start_point, 10)
        pygame.draw.circle(self.screen, BLACK, self.start_point, 10, 1)
        
        # 绘制车辆
        if self.editing:
            pass  # 编辑模式不显示车辆
        elif self.show_all_cars:
            for car in self.ga.population:
                car.draw(self.screen)
        else:
            # 只显示最佳车辆
            if self.ga.best_car:
                self.ga.best_car.draw(self.screen)
            else:
                self.ga.population[0].draw(self.screen)
        
        # 绘制UI
        for button in self.buttons:
            pygame.draw.rect(self.screen, GRAY, button["rect"])
            pygame.draw.rect(self.screen, BLACK, button["rect"], 1)
            text = self.font.render(button["text"], True, BLACK)
            text_rect = text.get_rect(center=button["rect"].center)
            self.screen.blit(text, text_rect)
        
        # 绘制滑块
        for slider in self.sliders:
            # 滑块背景
            pygame.draw.rect(self.screen, GRAY, (slider["x"], slider["y"], slider["width"], 10))
            
            # 滑块位置
            ratio = (slider["value"] - slider["min"]) / (slider["max"] - slider["min"])
            handle_pos = slider["x"] + ratio * slider["width"]
            pygame.draw.circle(self.screen, BLACK, (int(handle_pos), slider["y"] + 5), 8)
            
            # 滑块文本
            text = self.font.render(f"{slider['text']}: {slider['value']:.2f}", True, BLACK)
            self.screen.blit(text, (slider["x"] + slider["width"] + 10, slider["y"] - 5))
        
        # 绘制状态信息
        if not self.editing:
            # 显示当前代数和最佳适应度
            gen_text = self.font.render(f"代数: {self.ga.generation}", True, BLACK)
            self.screen.blit(gen_text, (10, SCREEN_HEIGHT - 60))
            
            fitness_text = self.font.render(f"最佳适应度: {self.ga.best_fitness:.2f}", True, BLACK)
            self.screen.blit(fitness_text, (10, SCREEN_HEIGHT - 30))
        else:
            # 编辑模式提示
            edit_text = self.font.render(f"编辑模式: {self.edit_mode}", True, BLACK)
            self.screen.blit(edit_text, (10, SCREEN_HEIGHT - 30))
            
            # 操作提示
            help_text = self.font.render("按 E 退出编辑模式，开始模拟", True, BLACK)
            self.screen.blit(help_text, (200, SCREEN_HEIGHT - 30))
        
        pygame.display.flip()
    
    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.handle_events()
            self.update()
            self.draw()
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run() 