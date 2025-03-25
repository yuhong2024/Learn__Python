import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from matplotlib.font_manager import FontProperties

# 设置中文字体
try:
    font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')
except:
    font = None

# 定义游戏环境
class PathFindingGame:
    """寻路游戏环境"""
    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height
        self.start_point = (10, 10)
        self.end_point = (90, 90)
        self.obstacles = []
        
        # 创建障碍物
        self.create_obstacles()
        
    def create_obstacles(self):
        """创建随机障碍物"""
        np.random.seed(42)  # 固定随机种子以便复现
        
        # 添加一些矩形障碍物
        for _ in range(5):
            x = np.random.randint(10, self.width - 30)
            y = np.random.randint(10, self.height - 30)
            w = np.random.randint(10, 30)
            h = np.random.randint(10, 30)
            self.obstacles.append(('rect', (x, y, w, h)))
        
        # 添加一些圆形障碍物
        for _ in range(5):
            x = np.random.randint(10, self.width - 20)
            y = np.random.randint(10, self.height - 20)
            r = np.random.randint(5, 15)
            self.obstacles.append(('circle', (x, y, r)))
            
    def check_collision(self, path):
        """检查路径是否与障碍物碰撞"""
        # 逐段检查路径
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i+1]
            
            # 每条线段上取多个点检查
            for t in np.linspace(0, 1, 10):
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
                
                # 检查点是否在任何障碍物内
                for obstacle_type, obstacle_params in self.obstacles:
                    if obstacle_type == 'rect':
                        x0, y0, w, h = obstacle_params
                        if x0 <= x <= x0 + w and y0 <= y <= y0 + h:
                            return True
                    elif obstacle_type == 'circle':
                        x0, y0, r = obstacle_params
                        if (x - x0)**2 + (y - y0)**2 <= r**2:
                            return True
        
        return False
    
    def calc_path_length(self, path):
        """计算路径总长度"""
        length = 0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            length += np.sqrt(dx**2 + dy**2)
        return length
    
    def evaluate_path(self, path):
        """评估路径的适应度"""
        # 如果路径与障碍物碰撞，适应度为0
        if self.check_collision(path):
            return 0
        
        # 否则，适应度与路径长度成反比
        path_length = self.calc_path_length(path)
        return 1000 / (path_length + 1)  # 加1避免除零
    
    def render(self, fig, ax, path=None, generation=0, fitness=0):
        """渲染游戏环境"""
        ax.clear()
        
        # 绘制边界
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        
        # 绘制障碍物
        for obstacle_type, obstacle_params in self.obstacles:
            if obstacle_type == 'rect':
                x, y, w, h = obstacle_params
                obstacle = Rectangle((x, y), w, h, fill=True, color='red', alpha=0.6)
                ax.add_patch(obstacle)
            elif obstacle_type == 'circle':
                x, y, r = obstacle_params
                obstacle = Circle((x, y), r, fill=True, color='red', alpha=0.6)
                ax.add_patch(obstacle)
        
        # 绘制起点和终点
        ax.plot(self.start_point[0], self.start_point[1], 'go', markersize=10)
        ax.plot(self.end_point[0], self.end_point[1], 'bo', markersize=10)
        
        # 绘制路径
        if path is not None:
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]
            ax.plot(x_coords, y_coords, 'k-', linewidth=2)
            
            # 绘制路径上的点
            ax.plot(x_coords, y_coords, 'ko', markersize=3)
        
        # 添加标题
        if generation > 0:
            title = f"第 {generation} 代 - 适应度: {fitness:.2f}"
            ax.set_title(title, fontproperties=font)
        else:
            ax.set_title("智能寻路游戏", fontproperties=font)
            
        ax.set_xlabel("X 坐标", fontproperties=font)
        ax.set_ylabel("Y 坐标", fontproperties=font)
        
        ax.grid(True)
        fig.canvas.draw()


# 遗传算法寻找最优路径
class PathFindingGA:
    """路径寻找的遗传算法"""
    def __init__(self, 
                 game_env,
                 population_size=100,
                 num_points=5,
                 crossover_rate=0.8,
                 mutation_rate=0.2,
                 mutation_scale=10.0,
                 elitism=2,
                 generations=100):
        
        self.game = game_env
        self.population_size = population_size
        self.num_points = num_points  # 路径中的控制点数量
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.elitism = elitism
        self.generations = generations
        
        # 开始点和结束点
        self.start_point = game_env.start_point
        self.end_point = game_env.end_point
        
        # 记录每一代的数据
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_path_history = []
    
    def initialize_population(self):
        """初始化种群 - 每个个体是一个路径"""
        population = []
        
        for _ in range(self.population_size):
            # 创建一条随机路径 (包括起点和终点)
            path = [self.start_point]
            
            # 添加中间点
            for _ in range(self.num_points):
                x = np.random.randint(0, self.game.width)
                y = np.random.randint(0, self.game.height)
                path.append((x, y))
            
            # 添加终点
            path.append(self.end_point)
            
            population.append(path)
        
        return population
    
    def calculate_fitness(self, population):
        """计算种群中每条路径的适应度"""
        fitness_values = np.zeros(self.population_size)
        
        for i, path in enumerate(population):
            fitness_values[i] = self.game.evaluate_path(path)
            
        return fitness_values
    
    def select_parents(self, population, fitness_values):
        """使用锦标赛选择法选择父代个体"""
        selected_parents = []
        tournament_size = 5
        
        for _ in range(self.population_size):
            # 随机选择tournament_size个个体
            competitors_idx = np.random.choice(self.population_size, size=tournament_size, replace=False)
            competitors_fitness = fitness_values[competitors_idx]
            
            # 找出这些个体中适应度最高的
            winner_idx = competitors_idx[np.argmax(competitors_fitness)]
            selected_parents.append(population[winner_idx])
            
        return selected_parents
    
    def crossover(self, parents):
        """交叉操作 - 对路径中的控制点进行交叉"""
        offspring = []
        
        # 确保父代数量是偶数
        if len(parents) % 2 == 1:
            parents.append(parents[0])
        
        # 将父代随机配对
        np.random.shuffle(parents)
        
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i+1]
            
            # 判断是否进行交叉
            if np.random.rand() < self.crossover_rate:
                # 选择一个随机交叉点
                crossover_point = np.random.randint(1, len(parent1) - 1)
                
                # 创建两个后代
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
                
                offspring.append(child1)
                offspring.append(child2)
            else:
                # 不交叉，直接复制父代
                offspring.append(parent1)
                offspring.append(parent2)
        
        return offspring
    
    def mutate(self, offspring):
        """变异操作 - 随机调整路径中的控制点"""
        for i in range(len(offspring)):
            path = offspring[i]
            
            # 对路径中的每个控制点进行变异判断
            for j in range(1, len(path) - 1):  # 跳过起点和终点
                if np.random.rand() < self.mutation_rate:
                    # 在当前位置附近随机偏移
                    x, y = path[j]
                    
                    # 添加一个随机偏移量
                    dx = np.random.normal(0, self.mutation_scale)
                    dy = np.random.normal(0, self.mutation_scale)
                    
                    # 限制在游戏边界内
                    new_x = min(max(0, x + dx), self.game.width)
                    new_y = min(max(0, y + dy), self.game.height)
                    
                    path[j] = (new_x, new_y)
        
        return offspring
    
    def elitism_selection(self, population, offspring, fitness_values):
        """精英选择 - 保留最优个体"""
        if self.elitism <= 0:
            return offspring
        
        # 找到当前种群中最优的几个个体
        elite_indices = np.argsort(fitness_values)[-self.elitism:]
        elite_individuals = [population[idx] for idx in elite_indices]
        
        # 随机替换后代中的个体
        replace_indices = np.random.choice(len(offspring), size=self.elitism, replace=False)
        for i, idx in enumerate(replace_indices):
            offspring[idx] = elite_individuals[i]
        
        return offspring
    
    def run(self, visualize=True):
        """运行遗传算法"""
        # 初始化种群
        population = self.initialize_population()
        
        # 如果需要可视化
        if visualize:
            fig, ax = plt.subplots(figsize=(10, 8))
            self.game.render(fig, ax)
            plt.pause(1)
        
        # 迭代进化
        for generation in range(self.generations):
            # 计算适应度
            fitness_values = self.calculate_fitness(population)
            
            # 记录当前代的最佳个体和适应度
            best_idx = np.argmax(fitness_values)
            best_fitness = fitness_values[best_idx]
            best_path = population[best_idx]
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitness_values))
            self.best_path_history.append(best_path)
            
            # 输出当前代的信息
            if generation % 10 == 0 or generation == self.generations - 1:
                print(f"代数: {generation}, 最佳适应度: {best_fitness:.4f}")
            
            # 可视化当前最佳路径
            if visualize and (generation % 10 == 0 or generation == self.generations - 1):
                self.game.render(fig, ax, best_path, generation, best_fitness)
                plt.pause(0.5)
            
            # 选择父代个体
            parents = self.select_parents(population, fitness_values)
            
            # 交叉
            offspring = self.crossover(parents)
            
            # 变异
            offspring = self.mutate(offspring)
            
            # 精英选择
            offspring = self.elitism_selection(population, offspring, fitness_values)
            
            # 更新种群
            population = offspring
        
        # 返回最终的最佳解
        best_solution = self.best_path_history[-1]
        best_fitness = self.best_fitness_history[-1]
        
        return best_solution, best_fitness
    
    def plot_results(self):
        """绘制进化过程"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.best_fitness_history, 'b-', label='最佳适应度')
        plt.plot(self.avg_fitness_history, 'r-', label='平均适应度')
        plt.xlabel('代数', fontproperties=font)
        plt.ylabel('适应度', fontproperties=font)
        plt.title('智能寻路进化过程', fontproperties=font)
        plt.legend(prop=font)
        plt.grid(True)
        plt.show()
    
    def create_animation(self):
        """创建进化过程的动画"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 每10代展示一次路径
        paths_to_show = [self.best_path_history[i] for i in range(0, len(self.best_path_history), 10)]
        if len(self.best_path_history) % 10 != 0:
            paths_to_show.append(self.best_path_history[-1])
        
        def update(frame):
            gen = frame * 10
            if frame == len(paths_to_show) - 1 and len(self.best_path_history) % 10 != 0:
                gen = len(self.best_path_history) - 1
                
            path = paths_to_show[frame]
            fitness = self.best_fitness_history[gen]
            self.game.render(fig, ax, path, gen, fitness)
            return ax,
        
        ani = animation.FuncAnimation(
            fig, update, frames=len(paths_to_show), interval=500, blit=False
        )
        
        plt.tight_layout()
        return ani


if __name__ == "__main__":
    # 创建游戏环境
    game = PathFindingGame(width=100, height=100)
    
    # 创建并运行遗传算法
    ga = PathFindingGA(
        game_env=game,
        population_size=100,
        num_points=5,
        crossover_rate=0.8,
        mutation_rate=0.2,
        mutation_scale=10.0,
        elitism=5,
        generations=100
    )
    
    print("【遗传算法寻路】开始优化...")
    best_path, best_fitness = ga.run(visualize=True)
    print(f"\n【遗传算法寻路】最终结果:")
    print(f"最佳适应度: {best_fitness:.6f}")
    print(f"最佳路径长度: {game.calc_path_length(best_path):.2f}")
    print(f"最佳路径点数: {len(best_path)}")
    
    # 绘制进化过程
    ga.plot_results()
    
    # 创建动画
    ani = ga.create_animation()
    plt.show()
    
    print("\n实验结束。按Ctrl+C退出程序。") 