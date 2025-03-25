import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
try:
    font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')
except:
    font = None

class GeneticAlgorithm:
    """遗传算法类"""
    def __init__(self, 
                 objective_function, 
                 bounds,
                 population_size=50, 
                 chromosome_length=32, 
                 crossover_rate=0.8, 
                 mutation_rate=0.1, 
                 elitism=2,
                 generations=100):
        """
        初始化遗传算法参数
        
        Args:
            objective_function: 目标函数 (适应度函数)
            bounds: 搜索空间边界 [min, max]
            population_size: 种群大小
            chromosome_length: 染色体长度 (二进制编码长度)
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            elitism: 精英数量 (直接进入下一代的最优个体数量)
            generations: 最大迭代次数
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.generations = generations
        
        # 记录每一代的最佳适应度
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual_history = []
        
    def initialize_population(self):
        """初始化种群 - 随机生成二进制编码的染色体"""
        return np.random.randint(2, size=(self.population_size, self.chromosome_length))
    
    def decode_chromosome(self, chromosome):
        """将二进制染色体解码为实数"""
        # 将二进制数组转为十进制值
        decimal_value = 0
        for i, gene in enumerate(reversed(chromosome)):
            decimal_value += gene * (2 ** i)
        
        # 映射到搜索空间
        min_bound, max_bound = self.bounds
        mapped_value = min_bound + (decimal_value / (2**self.chromosome_length - 1)) * (max_bound - min_bound)
        
        return mapped_value
    
    def calculate_fitness(self, population):
        """计算种群中每个个体的适应度"""
        fitness_values = np.zeros(self.population_size)
        
        for i, chromosome in enumerate(population):
            x = self.decode_chromosome(chromosome)
            fitness_values[i] = self.objective_function(x)
            
        return fitness_values
    
    def select_parents(self, population, fitness_values):
        """使用轮盘赌选择法选择父代个体"""
        # 根据适应度计算选择概率
        fitness_sum = np.sum(fitness_values)
        if fitness_sum == 0:
            selection_probs = np.ones(len(fitness_values)) / len(fitness_values)
        else:
            selection_probs = fitness_values / fitness_sum
        
        # 选择父代
        parent_indices = np.random.choice(
            self.population_size, 
            size=self.population_size, 
            p=selection_probs
        )
        
        return population[parent_indices]
    
    def crossover(self, parents):
        """单点交叉操作"""
        offspring = np.copy(parents)
        
        # 确定哪些个体将进行交叉
        crossover_mask = np.random.rand(self.population_size // 2) < self.crossover_rate
        crossover_pairs = np.where(crossover_mask)[0]
        
        # 对选中的配对进行交叉
        for idx in crossover_pairs:
            # 选择一个随机交叉点
            crossover_point = np.random.randint(1, self.chromosome_length)
            
            # 交换染色体片段
            parent1_idx = idx * 2
            parent2_idx = idx * 2 + 1
            
            temp = np.copy(offspring[parent1_idx, crossover_point:])
            offspring[parent1_idx, crossover_point:] = offspring[parent2_idx, crossover_point:]
            offspring[parent2_idx, crossover_point:] = temp
            
        return offspring
    
    def mutate(self, offspring):
        """变异操作 - 随机翻转基因位"""
        mutation_mask = np.random.rand(self.population_size, self.chromosome_length) < self.mutation_rate
        
        # 对被选中的基因进行翻转 (0变1，1变0)
        offspring[mutation_mask] = 1 - offspring[mutation_mask]
        
        return offspring
    
    def elitism_selection(self, population, offspring, fitness_values):
        """精英选择 - 保留最优个体"""
        if self.elitism <= 0:
            return offspring
        
        # 找到当前种群中最优的几个个体
        elite_indices = np.argsort(fitness_values)[-self.elitism:]
        elite_individuals = population[elite_indices]
        
        # 随机替换后代中的个体
        replace_indices = np.random.choice(self.population_size, size=self.elitism, replace=False)
        offspring[replace_indices] = elite_individuals
        
        return offspring
    
    def run(self):
        """运行遗传算法"""
        # 初始化种群
        population = self.initialize_population()
        
        # 迭代进化
        for generation in range(self.generations):
            # 计算适应度
            fitness_values = self.calculate_fitness(population)
            
            # 记录当前代的最佳个体和适应度
            best_idx = np.argmax(fitness_values)
            best_fitness = fitness_values[best_idx]
            best_individual = self.decode_chromosome(population[best_idx])
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitness_values))
            self.best_individual_history.append(best_individual)
            
            # 输出当前代的信息
            if generation % 10 == 0 or generation == self.generations - 1:
                print(f"代数: {generation}, 最佳适应度: {best_fitness:.4f}, 最佳个体值: {best_individual:.4f}")
            
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
        best_solution = self.best_individual_history[-1]
        best_fitness = self.best_fitness_history[-1]
        
        return best_solution, best_fitness
    
    def plot_results(self):
        """绘制进化过程图表"""
        plt.figure(figsize=(12, 8))
        
        # 绘制适应度曲线
        plt.subplot(2, 1, 1)
        plt.plot(self.best_fitness_history, label='最佳适应度')
        plt.plot(self.avg_fitness_history, label='平均适应度')
        plt.xlabel('代数', fontproperties=font)
        plt.ylabel('适应度', fontproperties=font)
        plt.title('遗传算法适应度进化曲线', fontproperties=font)
        plt.legend(prop=font)
        plt.grid(True)
        
        # 绘制最佳个体值曲线
        plt.subplot(2, 1, 2)
        plt.plot(self.best_individual_history)
        plt.xlabel('代数', fontproperties=font)
        plt.ylabel('最佳个体的解', fontproperties=font)
        plt.title('最佳个体值进化曲线', fontproperties=font)
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


# 示例：寻找函数 f(x) = x * sin(10π * x) + 2 在区间 [-1, 2] 的最大值
def objective_function(x):
    """目标函数：x * sin(10π * x) + 2"""
    return x * np.sin(10 * np.pi * x) + 2

# 创建并运行标准遗传算法
bounds = [-1, 2]  # 搜索空间
ga = GeneticAlgorithm(
    objective_function=objective_function,
    bounds=bounds,
    population_size=100,
    chromosome_length=22,
    crossover_rate=0.8,
    mutation_rate=0.1,
    elitism=2,
    generations=100
)

print("【标准遗传算法】开始优化...")
best_solution, best_fitness = ga.run()
print(f"\n【标准遗传算法】最终结果:")
print(f"最优解 x = {best_solution:.6f}")
print(f"函数最大值 f(x) = {best_fitness:.6f}")

# 绘制进化过程
ga.plot_results()

# 绘制目标函数和找到的最佳解
x = np.linspace(bounds[0], bounds[1], 1000)
y = np.array([objective_function(xi) for xi in x])

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='目标函数')
plt.plot(best_solution, best_fitness, 'ro', markersize=10, label='遗传算法找到的最优解')
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('目标函数与遗传算法找到的最优解', fontproperties=font)
plt.legend(prop=font)
plt.grid(True)
plt.show()

# ===== 改进的遗传算法 =====
class ImprovedGeneticAlgorithm(GeneticAlgorithm):
    """改进的遗传算法类"""
    def __init__(self, 
                 objective_function, 
                 bounds,
                 population_size=50, 
                 chromosome_length=32, 
                 crossover_rate=0.8, 
                 mutation_rate=0.1, 
                 elitism=2,
                 generations=100,
                 adaptive_mutation=True,  # 自适应变异
                 tournament_size=3):      # 锦标赛选择的参数
        
        super().__init__(
            objective_function, 
            bounds,
            population_size, 
            chromosome_length, 
            crossover_rate, 
            mutation_rate, 
            elitism,
            generations
        )
        
        self.adaptive_mutation = adaptive_mutation
        self.tournament_size = tournament_size
        
        # 初始化变异率范围
        self.min_mutation_rate = 0.01
        self.max_mutation_rate = 0.2
    
    def select_parents(self, population, fitness_values):
        """使用锦标赛选择法代替轮盘赌选择"""
        selected_parents = np.zeros((self.population_size, self.chromosome_length), dtype=int)
        
        for i in range(self.population_size):
            # 随机选择tournament_size个个体
            competitors_idx = np.random.choice(self.population_size, size=self.tournament_size, replace=False)
            competitors_fitness = fitness_values[competitors_idx]
            
            # 找出这些个体中适应度最高的
            winner_idx = competitors_idx[np.argmax(competitors_fitness)]
            selected_parents[i] = population[winner_idx]
            
        return selected_parents
    
    def crossover(self, parents):
        """使用两点交叉代替单点交叉"""
        offspring = np.copy(parents)
        
        # 确定哪些个体将进行交叉
        crossover_mask = np.random.rand(self.population_size // 2) < self.crossover_rate
        crossover_pairs = np.where(crossover_mask)[0]
        
        # 对选中的配对进行交叉
        for idx in crossover_pairs:
            # 选择两个随机交叉点
            point1, point2 = np.sort(np.random.choice(self.chromosome_length, size=2, replace=False))
            
            # 交换染色体片段
            parent1_idx = idx * 2
            parent2_idx = idx * 2 + 1
            
            temp = np.copy(offspring[parent1_idx, point1:point2])
            offspring[parent1_idx, point1:point2] = offspring[parent2_idx, point1:point2]
            offspring[parent2_idx, point1:point2] = temp
            
        return offspring
    
    def mutate(self, offspring, generation=0):
        """自适应变异 - 随着代数增加，变异率逐渐减小"""
        if self.adaptive_mutation:
            # 计算动态变异率 - 随着进化代数增加而减小
            progress = generation / self.generations
            current_mutation_rate = self.max_mutation_rate - progress * (self.max_mutation_rate - self.min_mutation_rate)
        else:
            current_mutation_rate = self.mutation_rate
            
        mutation_mask = np.random.rand(self.population_size, self.chromosome_length) < current_mutation_rate
        
        # 对被选中的基因进行翻转
        offspring[mutation_mask] = 1 - offspring[mutation_mask]
        
        return offspring
    
    def run(self):
        """运行改进的遗传算法"""
        # 初始化种群
        population = self.initialize_population()
        
        # 迭代进化
        for generation in range(self.generations):
            # 计算适应度
            fitness_values = self.calculate_fitness(population)
            
            # 记录当前代的最佳个体和适应度
            best_idx = np.argmax(fitness_values)
            best_fitness = fitness_values[best_idx]
            best_individual = self.decode_chromosome(population[best_idx])
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitness_values))
            self.best_individual_history.append(best_individual)
            
            # 输出当前代的信息
            if generation % 10 == 0 or generation == self.generations - 1:
                print(f"代数: {generation}, 最佳适应度: {best_fitness:.4f}, 最佳个体值: {best_individual:.4f}")
            
            # 选择父代个体 (使用锦标赛选择)
            parents = self.select_parents(population, fitness_values)
            
            # 交叉 (使用两点交叉)
            offspring = self.crossover(parents)
            
            # 变异 (自适应变异)
            offspring = self.mutate(offspring, generation)
            
            # 精英选择
            offspring = self.elitism_selection(population, offspring, fitness_values)
            
            # 更新种群
            population = offspring
        
        # 返回最终的最佳解
        best_solution = self.best_individual_history[-1]
        best_fitness = self.best_fitness_history[-1]
        
        return best_solution, best_fitness


# 创建并运行改进的遗传算法
improved_ga = ImprovedGeneticAlgorithm(
    objective_function=objective_function,
    bounds=bounds,
    population_size=100,
    chromosome_length=22,
    crossover_rate=0.8,
    mutation_rate=0.1,
    elitism=5,  # 增加精英数量
    generations=100,
    adaptive_mutation=True,
    tournament_size=5
)

print("\n【改进的遗传算法】开始优化...")
improved_best_solution, improved_best_fitness = improved_ga.run()
print(f"\n【改进的遗传算法】最终结果:")
print(f"最优解 x = {improved_best_solution:.6f}")
print(f"函数最大值 f(x) = {improved_best_fitness:.6f}")

# 绘制改进算法的进化过程
improved_ga.plot_results()

# 比较两种算法的性能
plt.figure(figsize=(12, 6))
plt.plot(ga.best_fitness_history, 'b-', label='标准遗传算法')
plt.plot(improved_ga.best_fitness_history, 'r-', label='改进的遗传算法')
plt.xlabel('代数', fontproperties=font)
plt.ylabel('最佳适应度', fontproperties=font)
plt.title('标准遗传算法与改进遗传算法的比较', fontproperties=font)
plt.legend(prop=font)
plt.grid(True)
plt.show()

# 绘制两种算法找到的最佳解
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='目标函数')
plt.plot(best_solution, best_fitness, 'ro', markersize=8, label='标准遗传算法')
plt.plot(improved_best_solution, improved_best_fitness, 'go', markersize=8, label='改进的遗传算法')
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('两种遗传算法找到的最优解比较', fontproperties=font)
plt.legend(prop=font)
plt.grid(True)
plt.show() 