import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import random
import math
import time
import os
from collections import deque

print("正在加载模块...")

# 定义颜色
BLUE = (0, 120, 255)
DARK_BLUE = (0, 60, 120)
GREEN = (0, 255, 100)
RED = (255, 50, 50)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (150, 150, 150)
ORANGE = (255, 150, 0)
PURPLE = (180, 60, 210)
BROWN = (139, 69, 19)

# 屏幕尺寸
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 800

# 环境分区
WATER_HEIGHT = 600  # 水域高度
LAND_HEIGHT = SCREEN_HEIGHT - WATER_HEIGHT  # 陆地高度

# 模拟速度控制
BASE_SPEED = 60  # 基础FPS
MAX_SPEED_MULTIPLIER = 20  # 最大速度倍数

# 统计数据记录周期（帧）
STAT_RECORD_INTERVAL = 100

# Environment类和其他生物类的定义保持不变...

# 添加新的Simulator类
class Simulator:
    """模拟器主类"""
    def __init__(self):
        # 初始化pygame
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("生态系统进化模拟器")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 14)
        self.large_font = pygame.font.SysFont('Arial', 24)
        
        # 模拟状态
        self.running = True
        self.paused = False
        self.simulation_speed = 1
        self.current_generation = 0
        self.frame_count = 0
        
        # 环境
        self.environment = Environment()
        
        # 生物列表
        self.organisms = []
        self.new_organisms = []  # 用于存储新出生的生物
        
        # 各种群
        self.big_fish_population = []
        self.small_fish_population = []
        self.shrimp_population = []
        self.adaptive_fish_population = []
        self.humans = []
        
        # 统计数据
        self.stats = {
            "big_fish_count": [],
            "small_fish_count": [],
            "shrimp_count": [],
            "adaptive_fish_count": [],
            "human_count": [],
            "male_female_ratio": [],
            "adaptive_male_female_ratio": [],
            "gender_ratio_genes": [],
            "avg_gender_ratio_gene": [],
            "environmental_stress": [],
            "generation": []
        }
        
        # UI元素
        self.sidebar_width = 300
        self.chart_height = 200
        self.dashboard_height = 200
        
        # 初始化生态系统
        self.initialize_ecosystem()
        
        # 设置matplotlib样式
        plt.style.use('dark_background')
        
    def initialize_ecosystem(self):
        """初始化生态系统，创建初始生物群体"""
        # 清空现有生物
        self.organisms = []
        
        # 创建初始种群
        self.create_initial_population(BigFish, 20)
        self.create_initial_population(SmallFish, 40)
        self.create_initial_population(Shrimp, 80)
        self.create_initial_population(AdaptiveFish, 30)
        self.create_initial_population(Human, 5)
        
        # 更新种群列表
        self.update_population_lists()
    
    def create_initial_population(self, species_class, count):
        """创建特定物种的初始种群"""
        # 计算安全生成区域
        spawn_margin = 50
        water_spawn_width = SCREEN_WIDTH - 2 * spawn_margin
        water_spawn_height = WATER_HEIGHT - 2 * spawn_margin
        
        for _ in range(count):
            # 在水域内随机位置
            x = spawn_margin + random.random() * water_spawn_width
            y = spawn_margin + random.random() * water_spawn_height
            
            # 人类部分应该在陆地上
            if species_class == Human and random.random() < 0.7:
                y = WATER_HEIGHT + random.random() * (LAND_HEIGHT - spawn_margin)
                
            # 创建生物并添加到列表
            organism = species_class(x, y)
            self.organisms.append(organism)

    def update_population_lists(self):
        """按物种更新种群列表"""
        self.big_fish_population = [org for org in self.organisms if org.species == "big_fish" and org.alive]
        self.small_fish_population = [org for org in self.organisms if org.species == "small_fish" and org.alive]
        self.shrimp_population = [org for org in self.organisms if org.species == "shrimp" and org.alive]
        self.adaptive_fish_population = [org for org in self.organisms if org.species == "adaptive_fish" and org.alive]
        self.humans = [org for org in self.organisms if org.species == "human" and org.alive]

    def update(self):
        """更新模拟状态"""
        if self.paused:
            return
            
        # 应用模拟速度
        for _ in range(self.simulation_speed):
            self.frame_count += 1
            
            # 更新环境
            self.environment.update()
            
            # 处理繁殖和捕食
            self.new_organisms = []
            
            # 更新生物并检查交互
            for i, org in enumerate(self.organisms):
                if not org.alive:
                    continue
                    
                # 更新生物
                offspring = org.update(self.organisms, self.environment)
                if offspring:
                    self.new_organisms.append(offspring)
                    
                # 捕猎检查
                if org.state == "hunting" and org.target and org.target.alive:
                    distance = math.sqrt((org.x - org.target.x)**2 + (org.y - org.target.y)**2)
                    if distance < org.size + org.target.size:
                        org.try_eat(org.target)
                        
                # 交配检查
                if org.state == "mating" and org.target and org.target.alive:
                    distance = math.sqrt((org.x - org.target.x)**2 + (org.y - org.target.y)**2)
                    if distance < (org.size + org.target.size) * 0.8:
                        offspring = org.reproduce(org.target, self.environment)
                        if offspring:
                            self.new_organisms.append(offspring)
            
            # 添加新生物
            self.organisms.extend(self.new_organisms)
            
            # 定期移除死亡生物以提高性能
            if self.frame_count % 100 == 0:
                self.organisms = [org for org in self.organisms if org.alive]
            
            # 更新种群列表
            self.update_population_lists()
            
            # 记录统计数据
            if self.frame_count % STAT_RECORD_INTERVAL == 0:
                self.record_statistics()
                
            # 检查代数进展
            if self.frame_count % 1000 == 0:
                self.current_generation += 1
                
            # 种群控制：当数量过低时添加更多生物
            self.population_control()

    def population_control(self):
        """维持最小可存活种群"""
        # 最小种群阈值
        min_thresholds = {
            "big_fish": 3,
            "small_fish": 5,
            "shrimp": 8,
            "adaptive_fish": 5,
            "human": 2
        }
        
        # 检查每个物种种群
        if len(self.big_fish_population) < min_thresholds["big_fish"]:
            for _ in range(2):
                x = random.randint(50, SCREEN_WIDTH - 50)
                y = random.randint(50, WATER_HEIGHT - 50)
                self.organisms.append(BigFish(x, y))
                
        if len(self.small_fish_population) < min_thresholds["small_fish"]:
            for _ in range(3):
                x = random.randint(50, SCREEN_WIDTH - 50)
                y = random.randint(50, WATER_HEIGHT - 50)
                self.organisms.append(SmallFish(x, y))
                
        if len(self.shrimp_population) < min_thresholds["shrimp"]:
            for _ in range(5):
                x = random.randint(50, SCREEN_WIDTH - 50)
                y = random.randint(50, WATER_HEIGHT - 50)
                self.organisms.append(Shrimp(x, y))
                
        if len(self.adaptive_fish_population) < min_thresholds["adaptive_fish"]:
            for _ in range(3):
                x = random.randint(50, SCREEN_WIDTH - 50)
                y = random.randint(50, WATER_HEIGHT - 50)
                self.organisms.append(AdaptiveFish(x, y))
                
        if len(self.humans) < min_thresholds["human"]:
            for _ in range(1):
                x = random.randint(50, SCREEN_WIDTH - 50)
                y = random.randint(WATER_HEIGHT, SCREEN_HEIGHT - 50)
                self.organisms.append(Human(x, y))

    def record_statistics(self):
        """记录种群统计数据"""
        # 计数种群
        self.stats["big_fish_count"].append(len(self.big_fish_population))
        self.stats["small_fish_count"].append(len(self.small_fish_population))
        self.stats["shrimp_count"].append(len(self.shrimp_population))
        self.stats["adaptive_fish_count"].append(len(self.adaptive_fish_population))
        self.stats["human_count"].append(len(self.humans))
        self.stats["generation"].append(self.current_generation)
        self.stats["environmental_stress"].append(self.environment.stress_index)
        
        # 计算性别比例
        if self.adaptive_fish_population:
            male_count = sum(1 for fish in self.adaptive_fish_population if fish.gender == "male")
            female_count = len(self.adaptive_fish_population) - male_count
            ratio = male_count / len(self.adaptive_fish_population) if len(self.adaptive_fish_population) > 0 else 0.5
            self.stats["adaptive_male_female_ratio"].append(ratio)
            
            # 追踪性别比例基因
            gene_values = [fish.genotype["gender_ratio_gene"] for fish in self.adaptive_fish_population]
            self.stats["avg_gender_ratio_gene"].append(sum(gene_values) / len(gene_values) if gene_values else 0.5)
            self.stats["gender_ratio_genes"].extend(gene_values)
        else:
            self.stats["adaptive_male_female_ratio"].append(0.5)
            self.stats["avg_gender_ratio_gene"].append(0.5)
        
        # 限制数据点以防止内存问题
        max_data_points = 1000
        for key in self.stats:
            if isinstance(self.stats[key], list) and len(self.stats[key]) > max_data_points:
                self.stats[key] = self.stats[key][-max_data_points:]

    def create_population_chart(self):
        """创建种群趋势图表"""
        if not self.stats["generation"]:
            return None
            
        fig, ax = plt.subplots(figsize=(5, 3), dpi=80)
        generations = self.stats["generation"][-100:]
        
        # 绘制种群趋势
        if self.stats["big_fish_count"]:
            ax.plot(generations, self.stats["big_fish_count"][-100:], color='blue', label='大鱼')
        if self.stats["small_fish_count"]:
            ax.plot(generations, self.stats["small_fish_count"][-100:], color='green', label='小鱼')
        if self.stats["shrimp_count"]:
            ax.plot(generations, self.stats["shrimp_count"][-100:], color='orange', label='虾')
        if self.stats["adaptive_fish_count"]:
            ax.plot(generations, self.stats["adaptive_fish_count"][-100:], color='purple', label='适应性鱼')
        if self.stats["human_count"]:
            ax.plot(generations, self.stats["human_count"][-100:], color='red', label='人类')
        
        ax.set_xlabel('代数')
        ax.set_ylabel('数量')
        ax.set_title('种群趋势')
        ax.legend(loc='upper right', fontsize='x-small')
        ax.grid(True, alpha=0.3)
        
        # 转换图表为pygame表面
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        plt.close(fig)
        
        return surf

    def create_gender_ratio_chart(self):
        """创建性别比例趋势图表"""
        if not self.stats["generation"] or not self.stats["adaptive_male_female_ratio"]:
            return None
            
        fig, ax = plt.subplots(figsize=(5, 3), dpi=80)
        generations = self.stats["generation"][-100:]
        
        # 绘制适应性鱼的性别比例趋势
        if self.stats["adaptive_male_female_ratio"]:
            ax.plot(generations, self.stats["adaptive_male_female_ratio"][-100:], color='magenta', label='雄性比例')
            ax.plot(generations, self.stats["avg_gender_ratio_gene"][-100:], color='cyan', label='性别基因')
        
        # 绘制环境压力
        if self.stats["environmental_stress"]:
            ax.plot(generations, self.stats["environmental_stress"][-100:], color='yellow', label='环境压力')
        
        ax.set_xlabel('代数')
        ax.set_ylabel('比例')
        ax.set_title('性别比例与环境压力')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize='x-small')
        ax.grid(True, alpha=0.3)
        
        # 转换图表为pygame表面
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        plt.close(fig)
        
        return surf

    def draw_dashboard(self):
        """绘制模拟仪表板"""
        # 仪表板背景
        dashboard_rect = pygame.Rect(SCREEN_WIDTH - self.sidebar_width, 0, self.sidebar_width, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, (30, 30, 40), dashboard_rect)
        
        # 绘制图表
        population_chart = self.create_population_chart()
        gender_chart = self.create_gender_ratio_chart()
        
        if population_chart:
            self.screen.blit(population_chart, (SCREEN_WIDTH - self.sidebar_width + 10, 10))
        
        if gender_chart:
            self.screen.blit(gender_chart, (SCREEN_WIDTH - self.sidebar_width + 10, 220))
        
        # 显示种群数量
        y_pos = 440
        title_text = self.large_font.render("当前种群", True, WHITE)
        self.screen.blit(title_text, (SCREEN_WIDTH - self.sidebar_width + 10, y_pos))
        y_pos += 30
        
        population_texts = [
            f"大鱼: {len(self.big_fish_population)}",
            f"小鱼: {len(self.small_fish_population)}",
            f"虾: {len(self.shrimp_population)}",
            f"适应性鱼: {len(self.adaptive_fish_population)}",
            f"人类: {len(self.humans)}",
            f"总数: {sum(1 for org in self.organisms if org.alive)}"
        ]
        
        for text in population_texts:
            text_surf = self.font.render(text, True, WHITE)
            self.screen.blit(text_surf, (SCREEN_WIDTH - self.sidebar_width + 15, y_pos))
            y_pos += 20
        
        # 适应性鱼的性别统计
        y_pos += 20
        if self.adaptive_fish_population:
            male_count = sum(1 for fish in self.adaptive_fish_population if fish.gender == "male")
            female_count = len(self.adaptive_fish_population) - male_count
            ratio = male_count / len(self.adaptive_fish_population) if len(self.adaptive_fish_population) > 0 else 0
            
            title_text = self.font.render("适应性鱼性别统计:", True, PURPLE)
            self.screen.blit(title_text, (SCREEN_WIDTH - self.sidebar_width + 10, y_pos))
            y_pos += 20
            
            gender_texts = [
                f"雄性: {male_count} ({ratio*100:.1f}%)",
                f"雌性: {female_count} ({(1-ratio)*100:.1f}%)",
                f"平均性别基因: {sum(fish.genotype['gender_ratio_gene'] for fish in self.adaptive_fish_population) / len(self.adaptive_fish_population):.3f}"
            ]
            
            for text in gender_texts:
                text_surf = self.font.render(text, True, WHITE)
                self.screen.blit(text_surf, (SCREEN_WIDTH - self.sidebar_width + 15, y_pos))
                y_pos += 20
        
        # 模拟控制
        y_pos = SCREEN_HEIGHT - 150
        pygame.draw.line(self.screen, WHITE, 
                        (SCREEN_WIDTH - self.sidebar_width + 10, y_pos), 
                        (SCREEN_WIDTH - 10, y_pos), 1)
        y_pos += 10
        
        control_text = self.font.render(f"速度: {self.simulation_speed}x", True, WHITE)
        self.screen.blit(control_text, (SCREEN_WIDTH - self.sidebar_width + 15, y_pos))
        y_pos += 20
        
        pause_text = self.font.render("已暂停" if self.paused else "运行中", True, YELLOW if self.paused else GREEN)
        self.screen.blit(pause_text, (SCREEN_WIDTH - self.sidebar_width + 15, y_pos))
        y_pos += 20
        
        gen_text = self.font.render(f"代数: {self.current_generation}", True, WHITE)
        self.screen.blit(gen_text, (SCREEN_WIDTH - self.sidebar_width + 15, y_pos))
        y_pos += 20
        
        frame_text = self.font.render(f"帧数: {self.frame_count}", True, WHITE)
        self.screen.blit(frame_text, (SCREEN_WIDTH - self.sidebar_width + 15, y_pos))
        
        # 控制帮助
        y_pos = SCREEN_HEIGHT - 50
        controls_text = self.font.render("控制: P=暂停, +/-=速度, R=重置", True, WHITE)
        self.screen.blit(controls_text, (SCREEN_WIDTH - self.sidebar_width + 15, y_pos))

    def draw(self):
        """绘制模拟状态"""
        # 清屏
        self.screen.fill(BLACK)
        
        # 绘制环境
        self.environment.draw(self.screen, self.font)
        
        # 绘制生物
        for organism in self.organisms:
            organism.draw(self.screen)
        
        # 绘制仪表板
        self.draw_dashboard()
        
        # 显示FPS
        fps_text = self.font.render(f"FPS: {int(self.clock.get_fps())}", True, WHITE)
        self.screen.blit(fps_text, (10, SCREEN_HEIGHT - 30))
        
        # 更新显示
        pygame.display.flip()

    def handle_events(self):
        """处理用户输入事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.KEYDOWN:
                # 暂停/继续
                if event.key == pygame.K_p:
                    self.paused = not self.paused
                    
                # 速度控制
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.simulation_speed = min(MAX_SPEED_MULTIPLIER, self.simulation_speed + 1)
                    
                elif event.key == pygame.K_MINUS:
                    self.simulation_speed = max(1, self.simulation_speed - 1)
                    
                # 重置模拟
                elif event.key == pygame.K_r:
                    self.initialize_ecosystem()
                    self.current_generation = 0
                    self.frame_count = 0
                    self.stats = {key: [] for key in self.stats}
                    self.environment = Environment()

    def spatial_partitioning(self):
        """优化版空间分区算法"""
        try:
            # 减小细胞大小，提高性能
            cell_size = 150  # 增大网格大小减少复杂度
            grid_width = SCREEN_WIDTH // cell_size + 1
            grid_height = SCREEN_HEIGHT // cell_size + 1
            
            # 使用字典而非二维数组，提高性能
            grid = {}
            
            # 仅处理活体生物
            active_organisms = [org for org in self.organisms if org.alive]
            
            # 将生物分配到网格
            for org in active_organisms:
                # 计算网格单元
                cell_x = int(org.x / cell_size)
                cell_y = int(org.y / cell_size)
                
                # 使用元组作为字典键
                cell_key = (cell_x, cell_y)
                
                # 将生物添加到对应网格
                if cell_key not in grid:
                    grid[cell_key] = []
                grid[cell_key].append(org)
            
            return grid, cell_size
        except Exception as e:
            print(f"空间分区错误: {e}")
            # 失败时返回简单结构
            return {}, 150

    def optimized_update(self):
        """修复版优化更新方法"""
        if self.paused:
            return
        
        try:
            # 应用模拟速度
            for _ in range(min(self.simulation_speed, 5)):  # 限制最大每帧更新次数
                self.frame_count += 1
                
                # 更新环境
                self.environment.update()
                
                # 创建空间分区网格
                grid, cell_size = self.spatial_partitioning()
                
                # 处理繁殖和捕食
                self.new_organisms = []
                
                # 限制处理的生物数量，防止卡死
                active_organisms = [org for org in self.organisms if org.alive]
                max_process = min(len(active_organisms), 500)  # 限制每帧处理的生物数量
                
                # 更新生物并检查交互
                for org in active_organisms[:max_process]:
                    # 获取生物的网格单元
                    cell_x = int(org.x / cell_size)
                    cell_y = int(org.y / cell_size)
                    cell_key = (cell_x, cell_y)
                    
                    # 收集附近的生物(当前单元和相邻单元)
                    nearby_orgs = []
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nearby_key = (cell_x + dx, cell_y + dy)
                            if nearby_key in grid:
                                nearby_orgs.extend(grid[nearby_key])
                    
                    # 只使用附近生物更新生物
                    offspring = org.update(nearby_orgs[:50], self.environment)  # 限制检测的生物数量
                    if offspring:
                        self.new_organisms.append(offspring)
                        
                    # 与附近生物的交互检查
                    for other in nearby_orgs[:20]:  # 限制检测的交互数量
                        if not other.alive or other == org:
                            continue
                            
                        # 使用快速距离估算
                        dx = org.x - other.x
                        dy = org.y - other.y
                        distance_sq = dx*dx + dy*dy
                        interact_threshold_sq = (org.size + other.size) * (org.size + other.size)
                        
                        # 只有当足够接近时才计算精确距离
                        if distance_sq > interact_threshold_sq:
                            continue
                            
                        distance = math.sqrt(distance_sq)
                        
                        # 捕猎检查
                        if org.state == "hunting" and org.target == other and distance < org.size + other.size:
                            org.try_eat(other)
                            
                        # 交配检查
                        if org.state == "mating" and org.target == other and distance < (org.size + other.size) * 0.8:
                            offspring = org.reproduce(other, self.environment)
                            if offspring:
                                self.new_organisms.append(offspring)
                
                # 添加新生物
                if len(self.new_organisms) > 0:
                    self.organisms.extend(self.new_organisms[:100])  # 限制每帧添加的新生物数量
                
                # 定期移除死亡生物
                if self.frame_count % 100 == 0:
                    self.organisms = [org for org in self.organisms if org.alive]
                
                # 更新种群列表
                if self.frame_count % 10 == 0:  # 减少更新频率
                    self.update_population_lists()
                
                # 记录统计数据
                if self.frame_count % STAT_RECORD_INTERVAL == 0:
                    self.record_statistics()
                    
                # 检查代数进展
                if self.frame_count % 1000 == 0:
                    self.current_generation += 1
                    print(f"当前代数: {self.current_generation}")
                    
                # 种群控制
                if self.frame_count % 200 == 0:  # 减少调用频率
                    self.population_control()
        
        except Exception as e:
            print(f"更新错误: {e}")
            import traceback
            traceback.print_exc()

    def draw_lite(self):
        """轻量级绘制方法 - 当标准版性能不足时使用"""
        # 清屏
        self.screen.fill(BLACK)
        
        # 简化的环境绘制
        water_rect = pygame.Rect(0, 0, SCREEN_WIDTH, WATER_HEIGHT)
        pygame.draw.rect(self.screen, (0, 100, 200), water_rect)
        
        land_rect = pygame.Rect(0, WATER_HEIGHT, SCREEN_WIDTH, LAND_HEIGHT)
        pygame.draw.rect(self.screen, BROWN, land_rect)
        
        # 简化生物绘制 - 只绘制一部分生物
        active_organisms = [org for org in self.organisms if org.alive]
        max_draw = min(len(active_organisms), 300)  # 限制绘制的生物数量
        
        for organism in active_organisms[:max_draw]:
            # 简化为基本圆形
            pygame.draw.circle(self.screen, organism.color, (int(organism.x), int(organism.y)), int(organism.size))
        
        # 简化的状态显示
        text = self.font.render(f"生物: {len(active_organisms)} | FPS: {int(self.clock.get_fps())} | 速度: {self.simulation_speed}x", True, WHITE)
        self.screen.blit(text, (10, SCREEN_HEIGHT - 30))
        
        # 更新显示
        pygame.display.flip()

    def run_simulation(self):
        """修复版主模拟循环"""
        print("开始模拟主循环...")
        try:
            # 添加帧率计数
            frame_count = 0
            last_time = time.time()
            fps_update_time = last_time
            
            while self.running:
                # 处理事件
                self.handle_events()
                
                # 更新模拟状态
                self.optimized_update()
                
                # 绘制一切
                self.draw()
                
                # 控制帧率
                self.clock.tick(BASE_SPEED)
                
                # 性能监控
                frame_count += 1
                current_time = time.time()
                if current_time - fps_update_time > 5.0:  # 每5秒显示一次FPS
                    fps = frame_count / (current_time - fps_update_time)
                    print(f"当前FPS: {fps:.1f}, 活体生物: {sum(1 for org in self.organisms if org.alive)}")
                    frame_count = 0
                    fps_update_time = current_time
                    
        except Exception as e:
            print(f"模拟错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            pygame.quit()
            print("模拟结束")

# 主程序入口点，确保程序正确启动
def main():
    print("初始化模拟器...")
    simulator = Simulator()
    print("开始模拟...")
    simulator.run_simulation()

# 在文件末尾添加，确保能正确运行模拟器
if __name__ == "__main__":
    print("启动生态系统进化模拟器...")
    main()