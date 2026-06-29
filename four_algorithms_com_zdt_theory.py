import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from matplotlib import rcParams
import warnings
import os
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings('ignore')

# ====================== 1. 全局配置 ======================
# ZDT 出图：汉字标注 + 华文宋体 + Morandi palette
fm.fontManager.addfont('C:/Windows/Fonts/STSONG.TTF')
rcParams['font.sans-serif'] = ['STSong']  # 华文宋体
rcParams['font.family'] = 'sans-serif'
rcParams["mathtext.fontset"] = "stix"
rcParams["axes.unicode_minus"] = False

# Morandi-inspired palette (brighter, higher saturation for print/screen clarity)
MORANDI_COLORS = {
    "NSGA2": "#5B8C9A",      # teal blue
    "MOEAD": "#C97B63",      # terracotta
    "SPEA3": "#8FAA6D",      # olive green
    "RL-MOEA1": "#9B7BB8",   # soft purple
    "RL-MOEA2": "#4A6FA5",   # clear blue (proposed method)
}
MORANDI_TRUE_PF = "#2F2F2F"
MORANDI_GRID = "#D0D0D0"
MORANDI_EDGE = "#FFFFFF"
MORANDI_MARKERS = {
    "NSGA2": "s",
    "MOEAD": "^",
    "SPEA3": "x",
    "RL-MOEA1": "o",
    "RL-MOEA2": "*",
}

# ZDT figure typography (larger canvas + spacing to avoid crowding)
ZDT_FIG_SIZE = (16.5, 13.5)
ZDT_FONT = {
    "suptitle": 22,
    "subtitle": 19,
    "label": 18,
    "tick": 17,
    "legend": 16,
    "bar_value": 16,
    "note": 17,
}


def _style_zdt_axis(ax, title: str, xlabel: str = None, ylabel: str = None):
    """Apply consistent enlarged fonts to one subplot."""
    ax.set_title(title, fontsize=ZDT_FONT["subtitle"], pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=ZDT_FONT["label"], labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=ZDT_FONT["label"], labelpad=6)
    ax.tick_params(axis="both", which="major", labelsize=ZDT_FONT["tick"])


def _draw_bars_with_value_labels(ax, algo_names, values, colors, value_formatter=None):
    """
    Draw bar chart and place value labels inside the reserved top margin (no clip at y-max).
    """
    if not values:
        return
    bars = ax.bar(
        algo_names,
        values,
        color=[colors[name] for name in algo_names],
        alpha=0.92,
        edgecolor=MORANDI_EDGE,
        linewidth=0.9,
    )
    ymax = float(max(values))
    ymin = float(min(values))
    y0 = 0.0 if ymin >= 0 else ymin * 1.05
    # Headroom for labels above the tallest bar
    y_top = ymax * 1.16 if ymax > 0 else 1.0
    if y_top <= ymax:
        y_top = ymax + 1e-6
    ax.set_ylim(y0, y_top)
    label_lift = (y_top - ymax) * 0.42
    for bar, value in zip(bars, values):
        if value_formatter is None:
            label = f"{value}"
        else:
            label = value_formatter(value)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_lift,
            label,
            ha="center",
            va="bottom",
            fontsize=ZDT_FONT["bar_value"],
            clip_on=False,
        )
    return bars


np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# ====================== 2. ZDT问题定义（完整保留并优化） ======================
class ZDTProblem:
    """ZDT系列测试问题"""
    def __init__(self, problem_name="ZDT1", n_vars=30):
        """
        初始化ZDT问题
        :param problem_name: 问题名称 (ZDT1, ZDT2, ZDT3, ZDT4, ZDT6)
        :param n_vars: 决策变量维度
        """
        self.problem_name = problem_name
        self.n_vars = n_vars
        self.n_objectives = 2
        self.bounds = self._get_variable_bounds()
    
    def _get_variable_bounds(self):
        """获取决策变量边界"""
        if self.problem_name == "ZDT4":
            # ZDT4第一个变量[0,1]，其余[-5,5]
            return [(0, 1)] + [(-5, 5)] * (self.n_vars - 1)
        else:
            # 其他ZDT问题所有变量[0,1]
            return [(0, 1)] * self.n_vars
    
    def evaluate(self, x):
        """
        评估解的目标值
        :param x: 决策变量向量 (n_vars,)
        :return: 目标值向量 (2,)
        """
        x = np.array(x, dtype=np.float64)
        if len(x) != self.n_vars:
            raise ValueError(f"决策变量维度错误，期望{self.n_vars}，实际{len(x)}")
        
        if self.problem_name == "ZDT1":
            return self._zdt1(x)
        elif self.problem_name == "ZDT2":
            return self._zdt2(x)
        elif self.problem_name == "ZDT3":
            return self._zdt3(x)
        elif self.problem_name == "ZDT4":
            return self._zdt4(x)
        elif self.problem_name == "ZDT6":
            return self._zdt6(x)
        else:
            raise ValueError(f"未知ZDT问题: {self.problem_name}，支持ZDT1/ZDT2/ZDT3/ZDT4/ZDT6")
    
    def _zdt1(self, x):
        """ZDT1: 凸Pareto前沿"""
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (self.n_vars - 1)
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h
        return np.array([f1, f2])
    
    def _zdt2(self, x):
        """ZDT2: 非凸Pareto前沿"""
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (self.n_vars - 1)
        h = 1 - (f1 / g) ** 2
        f2 = g * h
        return np.array([f1, f2])
    
    def _zdt3(self, x):
        """ZDT3: 不连续Pareto前沿"""
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (self.n_vars - 1)
        h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
        f2 = g * h
        return np.array([f1, f2])
    
    def _zdt4(self, x):
        """ZDT4: 多模态（易陷入局部最优）"""
        f1 = x[0]
        g = 1 + 10 * (self.n_vars - 1) + np.sum(x[1:] ** 2 - 10 * np.cos(4 * np.pi * x[1:]))
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h
        return np.array([f1, f2])
    
    def _zdt6(self, x):
        """ZDT6: 非均匀分布Pareto前沿"""
        f1 = 1 - np.exp(-4 * x[0]) * (np.sin(6 * np.pi * x[0])) ** 6
        g = 1 + 9 * (np.sum(x[1:]) / (self.n_vars - 1)) ** 0.25
        h = 1 - (f1 / g) ** 2
        f2 = g * h
        return np.array([f1, f2])

    def init_individual(self):
        """随机初始化一个可行解（符合变量边界）"""
        ind = []
        for (low, high) in self.bounds:
            ind.append(np.random.uniform(low, high))
        return np.array(ind)

def load_zdt_true_front(zdt_name="ZDT1", data_dir=None):
    """
    加载ZDT真实Pareto前沿数据（适配你的路径：C:\01common_codes\todolist\gso\moo_mclp_vertiport\ZDT前沿数据\前沿数据）
    :param zdt_name: ZDT问题名称（ZDT1/ZDT2/ZDT3/ZDT4/ZDT6）
    :param data_dir: 数据目录路径，默认自动拼接你的指定路径
    :return: 真实Pareto前沿 (n_points, 2)
    """
    # 默认路径配置（你的指定路径）
    if data_dir is None:
        base_dir = r"E:\train\pycharm\moo_mclp_vertiport\ZDT前沿数据\前沿数据"
        data_dir = base_dir
    
    # 拼接文件路径
    file_name = f"{zdt_name}.txt"
    file_path = os.path.join(data_dir, file_name)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"ZDT真实前沿文件不存在：{file_path}\n"
            f"请确认文件是否在该目录下，文件名应为{zdt_name}.txt（如ZDT1.txt）"
        )
    
    # 读取数据
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            # 按空格分割（支持多空格分隔）
            parts = line.split()
            if len(parts) < 2:
                print(f"警告：第{line_num}行数据格式错误，跳过该行：{line}")
                continue
            try:
                f1 = float(parts[0])
                f2 = float(parts[1])
                data_list.append([f1, f2])
            except ValueError:
                print(f"警告：第{line_num}行数据无法转换为浮点数，跳过该行：{line}")
                continue
    
    if not data_list:
        raise ValueError(f"文件{file_path}中未读取到有效Pareto前沿数据")
    
    return np.array(data_list, dtype=np.float64)

# ====================== 3. 核心MOO算法实现（适配ZDT连续问题） ======================
# ---------------------- 3.1 NSGA2（适配ZDT连续变量） ----------------------
class StandardNSGA2_ZDT:
    def __init__(self, problem, pop_size=50, max_iter=200, crossover_prob=0.8, mutation_prob=0.1):
        self.problem = problem
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.pop = self._init_population()
        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.pop])

    def _init_population(self):
        """初始化种群（适配ZDT连续变量）"""
        pop = []
        for _ in range(self.pop_size):
            ind = self.problem.init_individual()
            pop.append(ind)
        return np.array(pop)

    def _fast_nondominated_sort(self, fitness):
        """快速非支配排序"""
        pop_size = len(fitness)
        domination_count = np.zeros(pop_size, dtype=int)
        dominated_solutions = [[] for _ in range(pop_size)]
        ranks = np.zeros(pop_size, dtype=int)
        fronts = [[]]

        for i in range(pop_size):
            for j in range(pop_size):
                if i == j:
                    continue
                # 判断i是否支配j
                if (fitness[i][0] <= fitness[j][0] and fitness[i][1] <= fitness[j][1]) and \
                        (fitness[i][0] < fitness[j][0] or fitness[i][1] < fitness[j][1]):
                    dominated_solutions[i].append(j)
                # 判断j是否支配i
                elif (fitness[j][0] <= fitness[i][0] and fitness[j][1] <= fitness[i][1]) and \
                        (fitness[j][0] < fitness[i][0] or fitness[j][1] < fitness[i][1]):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                ranks[i] = 0
                fronts[0].append(i)

        current_rank = 0
        while len(fronts[current_rank]) > 0:
            next_front = []
            for i in fronts[current_rank]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        ranks[j] = current_rank + 1
                        next_front.append(j)
            current_rank += 1
            fronts.append(next_front)

        return [f for f in fronts if len(f) > 0], ranks

    def _calculate_crowding_distance(self, fitness, front):
        """计算拥挤度距离"""
        n = len(front)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([np.inf])
        
        distance = np.zeros(n)
        fitness_front = fitness[front]

        for obj in range(2):
            sorted_idx = np.argsort(fitness_front[:, obj])
            distance[sorted_idx[0]] = np.inf
            distance[sorted_idx[-1]] = np.inf
            obj_min = fitness_front[sorted_idx[0], obj]
            obj_max = fitness_front[sorted_idx[-1], obj]

            if obj_max - obj_min == 0:
                continue

            for i in range(1, n - 1):
                distance[sorted_idx[i]] += (fitness_front[sorted_idx[i + 1], obj] - 
                                           fitness_front[sorted_idx[i - 1], obj]) / (obj_max - obj_min)
        return distance

    def _tournament_selection(self, ranks, crowding):
        """锦标赛选择"""
        selected = []
        for _ in range(self.pop_size):
            idx1, idx2 = random.sample(range(len(self.fitness)), 2)
            if ranks[idx1] < ranks[idx2]:
                selected.append(self.pop[idx1])
            elif ranks[idx1] > ranks[idx2]:
                selected.append(self.pop[idx2])
            else:
                if crowding[idx1] > crowding[idx2]:
                    selected.append(self.pop[idx1])
                else:
                    selected.append(self.pop[idx2])
        return np.array(selected)

    def _sbx_crossover(self, parent1, parent2):
        """模拟二进制交叉（SBX，适配连续变量）"""
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        n_vars = len(parent1)
        child1 = parent1.copy()
        child2 = parent2.copy()
        eta = 20  # 交叉分布指数，越大子代越接近父代

        for i in range(n_vars):
            if parent1[i] != parent2[i]:
                low = self.problem.bounds[i][0]
                high = self.problem.bounds[i][1]
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                
                c1 = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                c2 = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
                # 边界截断
                child1[i] = np.clip(c1, low, high)
                child2[i] = np.clip(c2, low, high)
        return child1, child2

    def _polynomial_mutation(self, individual):
        """多项式变异（适配连续变量）"""
        mutated = individual.copy()
        n_vars = len(mutated)
        eta = 20  # 变异分布指数
        for i in range(n_vars):
            if random.random() < self.mutation_prob:
                low = self.problem.bounds[i][0]
                high = self.problem.bounds[i][1]
                u = np.random.random()
                if u <= 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                mutated[i] += delta * (high - low)
                # 边界截断
                mutated[i] = np.clip(mutated[i], low, high)
        return mutated

    def _create_offspring(self, parents):
        """生成子代种群"""
        offspring = []
        for i in range(0, self.pop_size, 2):
            if i + 1 >= self.pop_size:
                break
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = self._sbx_crossover(parent1, parent2)
            offspring.append(self._polynomial_mutation(child1))
            offspring.append(self._polynomial_mutation(child2))
        offspring = offspring[:self.pop_size]
        return np.array(offspring)

    def run(self):
        """运行NSGA2算法"""
        for _ in range(self.max_iter):
            # 非支配排序与拥挤度计算
            fronts, ranks = self._fast_nondominated_sort(self.fitness)
            crowding = np.zeros(len(self.fitness))
            for front in fronts:
                crowding[front] = self._calculate_crowding_distance(self.fitness, front)
            
            # 选择父代
            parents = self._tournament_selection(ranks, crowding)
            # 生成子代
            offspring = self._create_offspring(parents)
            # 评估子代
            offspring_fitness = np.array([self.problem.evaluate(ind) for ind in offspring])
            
            # 合并父代与子代
            combined_pop = np.vstack((self.pop, offspring))
            combined_fitness = np.vstack((self.fitness, offspring_fitness))
            
            # 新一代选择
            combined_fronts, combined_ranks = self._fast_nondominated_sort(combined_fitness)
            combined_crowding = np.zeros(len(combined_fitness))
            for front in combined_fronts:
                combined_crowding[front] = self._calculate_crowding_distance(combined_fitness, front)
            
            new_pop = []
            new_fitness = []
            for front in combined_fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(combined_pop[front])
                    new_fitness.extend(combined_fitness[front])
                else:
                    sorted_front = sorted(front, key=lambda x: combined_crowding[x], reverse=True)
                    selected = sorted_front[:self.pop_size - len(new_pop)]
                    new_pop.extend(combined_pop[selected])
                    new_fitness.extend(combined_fitness[selected])
                    break
            
            self.pop = np.array(new_pop)
            self.fitness = np.array(new_fitness)
        
        # 提取最终Pareto前沿
        final_fronts, _ = self._fast_nondominated_sort(self.fitness)
        final_pareto = self.fitness[final_fronts[0]] if len(final_fronts) > 0 and len(final_fronts[0]) > 0 else np.array([])
        return final_pareto

# ---------------------- 3.2 MOEAD（适配ZDT连续变量） ----------------------
class StandardMOEAD_ZDT:
    def __init__(self, problem, pop_size=50, max_iter=200, T=20, crossover_prob=0.8, mutation_prob=0.1):
        self.problem = problem
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.T = min(T, pop_size // 2)
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.lambdas = self._generate_weight_vectors()
        self.pop = self._init_population()
        self.objectives = np.array([self.problem.evaluate(ind) for ind in self.pop])
        self.neighbors = self._compute_neighbors()
        self.ideal_point = np.min(self.objectives, axis=0)

    def _generate_weight_vectors(self):
        """生成均匀分布的权重向量"""
        lambdas = []
        for i in range(self.pop_size):
            lam1 = i / (self.pop_size - 1) if self.pop_size > 1 else 0.5
            lam2 = 1 - lam1
            lambdas.append(np.array([lam1, lam2]))
        return np.array(lambdas)

    def _compute_neighbors(self):
        """计算每个权重向量的邻近个体"""
        neighbors = []
        for i in range(self.pop_size):
            distances = np.linalg.norm(self.lambdas - self.lambdas[i], axis=1)
            idx_sorted = np.argsort(distances)
            neighbors.append(idx_sorted[1:self.T + 1])
        return np.array(neighbors)

    def _init_population(self):
        """初始化种群（适配ZDT连续变量）"""
        pop = []
        for _ in range(self.pop_size):
            ind = self.problem.init_individual()
            pop.append(ind)
        return np.array(pop)

    def _tchebycheff_decomposition(self, obj_values, lambda_vec):
        """切比雪夫分解"""
        normalized = lambda_vec * np.abs(obj_values - self.ideal_point)
        return np.max(normalized)

    def _sbx_crossover(self, parent1, parent2):
        """模拟二进制交叉（SBX，适配连续变量）"""
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        n_vars = len(parent1)
        child1 = parent1.copy()
        child2 = parent2.copy()
        eta = 20

        for i in range(n_vars):
            if parent1[i] != parent2[i]:
                low = self.problem.bounds[i][0]
                high = self.problem.bounds[i][1]
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                
                c1 = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                c2 = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
                child1[i] = np.clip(c1, low, high)
                child2[i] = np.clip(c2, low, high)
        return child1, child2

    def _polynomial_mutation(self, individual):
        """多项式变异（适配连续变量）"""
        mutated = individual.copy()
        n_vars = len(mutated)
        eta = 20
        for i in range(n_vars):
            if random.random() < self.mutation_prob:
                low = self.problem.bounds[i][0]
                high = self.problem.bounds[i][1]
                u = np.random.random()
                if u <= 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                mutated[i] += delta * (high - low)
                mutated[i] = np.clip(mutated[i], low, high)
        return mutated

    def _update_ideal_point(self, new_obj):
        """更新理想点"""
        self.ideal_point = np.minimum(self.ideal_point, new_obj)

    def _update_neighbors(self, idx, child, child_obj):
        """更新邻近个体"""
        for j in self.neighbors[idx]:
            old_tch = self._tchebycheff_decomposition(self.objectives[j], self.lambdas[j])
            new_tch = self._tchebycheff_decomposition(child_obj, self.lambdas[j])
            if new_tch < old_tch:
                self.pop[j] = child.copy()
                self.objectives[j] = child_obj.copy()
                self._update_ideal_point(child_obj)

    def _get_pareto_front(self):
        """提取Pareto前沿"""
        pareto_indices = []
        for i in range(self.pop_size):
            is_dominated = False
            for j in range(self.pop_size):
                if i == j:
                    continue
                if (self.objectives[j][0] <= self.objectives[i][0] and self.objectives[j][1] <= self.objectives[i][1]) and \
                        (self.objectives[j][0] < self.objectives[i][0] or self.objectives[j][1] < self.objectives[i][1]):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_indices.append(i)
        return self.objectives[pareto_indices]

    def run(self):
        """运行MOEAD算法"""
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                parent_indices = random.sample(list(self.neighbors[i]), 2)
                p1, p2 = self.pop[parent_indices[0]], self.pop[parent_indices[1]]
                child1, child2 = self._sbx_crossover(p1, p2)
                child = self._polynomial_mutation(child1)
                child_obj = self.problem.evaluate(child)
                self._update_ideal_point(child_obj)
                self._update_neighbors(i, child, child_obj)
        final_pareto = self._get_pareto_front()
        return final_pareto

# ---------------------- 3.3 SPEA3（适配ZDT连续变量） ----------------------
class StandardSPEA3_ZDT:
    def __init__(self, problem, pop_size=50, archive_size=50, max_iter=200, crossover_prob=0.8, mutation_prob=0.1):
        self.problem = problem
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.max_iter = max_iter
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.pop = self._init_population()
        self.archive = np.empty((0, self.problem.n_vars))
        self.pop_obj = np.array([self.problem.evaluate(ind) for ind in self.pop])

    def _init_population(self):
        """初始化种群（适配ZDT连续变量）"""
        pop = []
        for _ in range(self.pop_size):
            ind = self.problem.init_individual()
            pop.append(ind)
        return np.array(pop)

    def _is_dominated(self, obj1, obj2):
        """判断obj2是否支配obj1"""
        return (obj2[0] <= obj1[0] and obj2[1] <= obj1[1]) and (obj2[0] < obj1[0] or obj2[1] < obj1[1])

    def _get_non_dominated(self, objectives):
        """提取非支配解索引"""
        n = len(objectives)
        if n == 0:
            return np.array([])
        non_dominated_idx = []
        for i in range(n):
            is_dominated = False
            for j in range(n):
                if i == j:
                    continue
                if self._is_dominated(objectives[i], objectives[j]):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated_idx.append(i)
        return np.array(non_dominated_idx)

    def _compute_strength_raw_fitness(self, combined_obj):
        """计算强度值和原始适应度"""
        n = len(combined_obj)
        if n == 0:
            return np.array([]), np.array([])
        strength = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._is_dominated(combined_obj[j], combined_obj[i]):
                    strength[i] += 1
        raw_fitness = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._is_dominated(combined_obj[i], combined_obj[j]):
                    raw_fitness[i] += strength[j]
            raw_fitness[i] += 1
        return strength, raw_fitness

    def _hypervolume_contribution(self, pareto_obj, ref_point):
        """计算超体积贡献度"""
        n = len(pareto_obj)
        if n == 0:
            return np.array([])
        sorted_idx = np.argsort(pareto_obj[:, 0])
        sorted_obj = pareto_obj[sorted_idx]
        contribution = np.zeros(n)
        prev_x = ref_point[0]
        prev_y = ref_point[1]
        for i in range(n):
            x = sorted_obj[i, 0]
            y = sorted_obj[i, 1]
            width = prev_x - x
            height = prev_y - y
            contribution[sorted_idx[i]] = width * height
            prev_y = y
        if n > 1:
            contribution[sorted_idx[0]] *= 1.2
            contribution[sorted_idx[-1]] *= 1.2
        return contribution

    def _environmental_selection(self, combined_pop, combined_obj):
        """环境选择"""
        if len(combined_obj) == 0:
            return np.empty((0, self.problem.n_vars))
        non_dominated_idx = self._get_non_dominated(combined_obj)
        if len(non_dominated_idx) == 0:
            _, raw_fitness = self._compute_strength_raw_fitness(combined_obj)
            sorted_idx = np.argsort(raw_fitness)
            return combined_pop[sorted_idx[:self.archive_size]]
        non_dominated_pop = combined_pop[non_dominated_idx]
        non_dominated_obj = combined_obj[non_dominated_idx]
        if len(non_dominated_idx) <= self.archive_size:
            new_archive = non_dominated_pop.copy()
            dominated_idx = [i for i in range(len(combined_obj)) if i not in non_dominated_idx]
            if len(dominated_idx) > 0 and len(new_archive) < self.archive_size:
                _, raw_fitness = self._compute_strength_raw_fitness(combined_obj)
                dominated_sorted = sorted(dominated_idx, key=lambda x: raw_fitness[x])
                need = self.archive_size - len(new_archive)
                new_archive = np.vstack((new_archive, combined_pop[dominated_sorted[:need]]))
        else:
            ref_point = np.max(non_dominated_obj, axis=0) + 1
            contribution = self._hypervolume_contribution(non_dominated_obj, ref_point)
            sorted_idx = np.argsort(contribution)[::-1]
            new_archive = non_dominated_pop[sorted_idx[:self.archive_size]]
        return new_archive

    def _binary_tournament_selection(self, archive):
        """二进制锦标赛选择"""
        if len(archive) == 0:
            return np.empty((0, self.problem.n_vars))
        selected = []
        archive_obj = np.array([self.problem.evaluate(ind) for ind in archive])
        _, raw_fitness = self._compute_strength_raw_fitness(archive_obj)
        for _ in range(self.pop_size):
            idx1, idx2 = random.sample(range(len(archive)), 2)
            if raw_fitness[idx1] < raw_fitness[idx2]:
                selected.append(archive[idx1])
            else:
                selected.append(archive[idx2])
        return np.array(selected)

    def _sbx_crossover(self, parent1, parent2):
        """模拟二进制交叉（SBX，适配连续变量）"""
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        n_vars = len(parent1)
        child1 = parent1.copy()
        child2 = parent2.copy()
        eta = 20

        for i in range(n_vars):
            if parent1[i] != parent2[i]:
                low = self.problem.bounds[i][0]
                high = self.problem.bounds[i][1]
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                
                c1 = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                c2 = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
                child1[i] = np.clip(c1, low, high)
                child2[i] = np.clip(c2, low, high)
        return child1, child2

    def _polynomial_mutation(self, individual):
        """多项式变异（适配连续变量）"""
        mutated = individual.copy()
        n_vars = len(mutated)
        eta = 20
        for i in range(n_vars):
            if random.random() < self.mutation_prob:
                low = self.problem.bounds[i][0]
                high = self.problem.bounds[i][1]
                u = np.random.random()
                if u <= 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                mutated[i] += delta * (high - low)
                mutated[i] = np.clip(mutated[i], low, high)
        return mutated

    def _create_offspring(self, parents):
        """生成子代"""
        if len(parents) < 2:
            return np.empty((0, self.problem.n_vars))
        offspring = []
        for i in range(0, self.pop_size, 2):
            if i + 1 >= self.pop_size:
                break
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = self._sbx_crossover(parent1, parent2)
            offspring.append(self._polynomial_mutation(child1))
            offspring.append(self._polynomial_mutation(child2))
        offspring = offspring[:self.pop_size]
        return np.array(offspring)

    def run(self):
        """运行SPEA3算法"""
        for _ in range(self.max_iter):
            if _ == 0:
                combined_pop = self.pop
                combined_obj = self.pop_obj
            else:
                if len(self.archive) == 0:
                    combined_pop = self.pop
                    combined_obj = self.pop_obj
                else:
                    combined_pop = np.vstack((self.pop, self.archive))
                    combined_obj = np.vstack(
                        (self.pop_obj, np.array([self.problem.evaluate(ind) for ind in self.archive])))
            self.archive = self._environmental_selection(combined_pop, combined_obj)
            if len(self.archive) == 0:
                break
            parents = self._binary_tournament_selection(self.archive)
            self.pop = self._create_offspring(parents)
            if len(self.pop) > 0:
                self.pop_obj = np.array([self.problem.evaluate(ind) for ind in self.pop])
            else:
                self.pop_obj = np.array([])
        final_archive_obj = np.array([self.problem.evaluate(ind) for ind in self.archive]) if len(self.archive) > 0 else np.array([])
        final_pareto_idx = self._get_non_dominated(final_archive_obj)
        final_pareto = final_archive_obj[final_pareto_idx] if len(final_pareto_idx) > 0 else np.array([])
        return final_pareto

# ---------------------- 3.4 RL-MOEA2（适配ZDT连续变量） ----------------------
class DQN_ZDT(nn.Module):
    """DQN网络（适配ZDT问题状态）"""
    def __init__(self, state_dim, action_dim):
        super(DQN_ZDT, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class RLMOEA2_ZDT:
    def __init__(self, problem, pop_size=50, max_iter=200):
        self.problem = problem
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.mutation_prob = 0.1
        
        # DQN配置
        self.state_dim = 4
        self.action_dim = 3
        self.dqn = DQN_ZDT(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        
        # 种群初始化
        self.pop = self._init_population()
        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.pop])
        self.current_iter = 0
        self.last_action = 0
        self.last_reward = 0.0
        
        # 归一化参考值
        self.max_fitness = np.max(self.fitness, axis=0) if len(self.fitness) > 0 else np.array([1.0, 1.0])
        self.min_fitness = np.min(self.fitness, axis=0) if len(self.fitness) > 0 else np.array([0.0, 0.0])
    
    def _init_population(self):
        """初始化种群（适配ZDT连续变量）"""
        pop = []
        for _ in range(self.pop_size):
            ind = self.problem.init_individual()
            pop.append(ind)
        return np.array(pop)
    
    def _get_state(self):
        """构造状态向量"""
        if len(self.fitness) == 0:
            return torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        
        # 归一化适应度
        fitness_range = self.max_fitness - self.min_fitness
        fitness_range = np.where(fitness_range == 0, 1, fitness_range)
        fitness_norm = (self.fitness - self.min_fitness) / fitness_range
        avg_fitness = np.mean(fitness_norm, axis=0)
        
        # 多样性
        diversity = np.std(self.fitness, axis=0).mean()
        diversity_norm = diversity / (np.max(self.fitness) - np.min(self.fitness) + 1e-6)
        
        # 迭代进度
        iter_progress = self.current_iter / self.max_iter
        
        return torch.tensor([avg_fitness[0], avg_fitness[1], diversity_norm, iter_progress], dtype=torch.float32)
    
    def _select_operator(self):
        """ε-贪心选择交叉算子"""
        if random.random() < self.epsilon:
            action = random.choice(range(self.action_dim))
        else:
            state = self._get_state()
            q_values = self.dqn(state)
            action = torch.argmax(q_values).item()
        self.last_action = action
        return action
    
    def _sbx_crossover(self, parent1, parent2):
        """模拟二进制交叉（SBX）"""
        n_vars = len(parent1)
        child1 = parent1.copy()
        child2 = parent2.copy()
        eta = 20

        for i in range(n_vars):
            if parent1[i] != parent2[i]:
                low = self.problem.bounds[i][0]
                high = self.problem.bounds[i][1]
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                
                c1 = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                c2 = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
                child1[i] = np.clip(c1, low, high)
                child2[i] = np.clip(c2, low, high)
        return child1, child2
    
    def _blend_crossover(self, parent1, parent2):
        """混合交叉（BLX-α，α=0.5）"""
        n_vars = len(parent1)
        child1 = parent1.copy()
        child2 = parent2.copy()
        alpha = 0.5

        for i in range(n_vars):
            low = self.problem.bounds[i][0]
            high = self.problem.bounds[i][1]
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            delta = alpha * (max_val - min_val)
            # 随机生成子代
            child1[i] = np.random.uniform(min_val - delta, max_val + delta)
            child2[i] = np.random.uniform(min_val - delta, max_val + delta)
            # 边界截断
            child1[i] = np.clip(child1[i], low, high)
            child2[i] = np.clip(child2[i], low, high)
        return child1, child2
    
    def _uniform_crossover(self, parent1, parent2):
        """均匀交叉（适配连续变量）"""
        n_vars = len(parent1)
        child1 = parent1.copy()
        child2 = parent2.copy()
        for i in range(n_vars):
            if random.random() < 0.5:
                child1[i], child2[i] = parent2[i], parent1[i]
        # 边界截断
        for i in range(n_vars):
            low = self.problem.bounds[i][0]
            high = self.problem.bounds[i][1]
            child1[i] = np.clip(child1[i], low, high)
            child2[i] = np.clip(child2[i], low, high)
        return child1, child2
    
    def _polynomial_mutation(self, individual):
        """多项式变异"""
        mutated = individual.copy()
        n_vars = len(mutated)
        eta = 20
        for i in range(n_vars):
            if random.random() < self.mutation_prob:
                low = self.problem.bounds[i][0]
                high = self.problem.bounds[i][1]
                u = np.random.random()
                if u <= 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                mutated[i] += delta * (high - low)
                mutated[i] = np.clip(mutated[i], low, high)
        return mutated
    
    def _update_dqn(self, reward):
        """更新DQN网络"""
        if self.current_iter == 0:
            return
        
        state = self._get_state()
        target = reward + 0.9 * torch.max(self.dqn(state))
        q_value = self.dqn(state)[self.last_action]
        loss = nn.MSELoss()(q_value, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def run(self):
        """运行RL-MOEA2算法"""
        # 内部非支配排序和拥挤度计算
        def _fast_nondominated_sort(fitness):
            pop_size = len(fitness)
            if pop_size == 0:
                return []
            domination_count = np.zeros(pop_size, dtype=int)
            dominated_solutions = [[] for _ in range(pop_size)]
            ranks = np.zeros(pop_size, dtype=int)
            fronts = [[]]
            for i in range(pop_size):
                for j in range(pop_size):
                    if i == j:
                        continue
                    if (fitness[i][0] <= fitness[j][0] and fitness[i][1] <= fitness[j][1]) and \
                            (fitness[i][0] < fitness[j][0] or fitness[i][1] < fitness[j][1]):
                        dominated_solutions[i].append(j)
                    elif (fitness[j][0] <= fitness[i][0] and fitness[j][1] <= fitness[i][1]) and \
                            (fitness[j][0] < fitness[i][0] or fitness[j][1] < fitness[i][1]):
                        domination_count[i] += 1
                if domination_count[i] == 0:
                    ranks[i] = 0
                    fronts[0].append(i)
            current_rank = 0
            while len(fronts[current_rank]) > 0:
                next_front = []
                for i in fronts[current_rank]:
                    for j in dominated_solutions[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            ranks[j] = current_rank + 1
                            next_front.append(j)
                current_rank += 1
                fronts.append(next_front)
            return [f for f in fronts if len(f) > 0]
        
        def _calculate_crowding_distance(fitness, front):
            n = len(front)
            if n <= 2:
                return np.array([np.inf] * n)
            distance = np.zeros(n)
            fitness_front = fitness[front]
            for obj in range(2):
                sorted_idx = np.argsort(fitness_front[:, obj])
                distance[sorted_idx[0]] = np.inf
                distance[sorted_idx[-1]] = np.inf
                obj_min = fitness_front[sorted_idx[0], obj]
                obj_max = fitness_front[sorted_idx[-1], obj]
                if obj_max - obj_min == 0:
                    continue
                for i in range(1, n - 1):
                    distance[sorted_idx[i]] += (fitness_front[sorted_idx[i + 1], obj] - 
                                               fitness_front[sorted_idx[i - 1], obj]) / (obj_max - obj_min)
            return distance
        
        old_pareto_size = 0
        
        for iter in range(self.max_iter):
            self.current_iter = iter
            if len(self.fitness) == 0:
                break
            
            # 更新参考值
            self.max_fitness = np.maximum(self.max_fitness, np.max(self.fitness, axis=0))
            self.min_fitness = np.minimum(self.min_fitness, np.min(self.fitness, axis=0))
            
            # 选择交叉算子
            action = self._select_operator()
            
            # 锦标赛选择父代
            fronts = _fast_nondominated_sort(self.fitness)
            ranks = np.zeros(len(self.fitness))
            crowding = np.zeros(len(self.fitness))
            for rank, front in enumerate(fronts):
                ranks[front] = rank
                crowding[front] = _calculate_crowding_distance(self.fitness, front)
            
            parents = []
            for _ in range(self.pop_size):
                idx1, idx2 = random.sample(range(len(self.pop)), 2)
                if ranks[idx1] < ranks[idx2]:
                    parents.append(self.pop[idx1])
                elif ranks[idx1] > ranks[idx2]:
                    parents.append(self.pop[idx2])
                else:
                    if crowding[idx1] > crowding[idx2]:
                        parents.append(self.pop[idx1])
                    else:
                        parents.append(self.pop[idx2])
            parents = np.array(parents)
            
            # 执行交叉
            offspring = []
            for i in range(0, self.pop_size, 2):
                if i + 1 >= self.pop_size:
                    break
                parent1, parent2 = parents[i], parents[i + 1]
                
                if action == 0:
                    child1, child2 = self._sbx_crossover(parent1, parent2)
                elif action == 1:
                    child1, child2 = self._blend_crossover(parent1, parent2)
                else:
                    child1, child2 = self._uniform_crossover(parent1, parent2)
                
                offspring.append(self._polynomial_mutation(child1))
                offspring.append(self._polynomial_mutation(child2))
            offspring = np.array(offspring[:self.pop_size])
            
            # 评估子代
            offspring_fitness = np.array([self.problem.evaluate(ind) for ind in offspring])
            
            # 合并选择
            combined_pop = np.vstack((self.pop, offspring))
            combined_fitness = np.vstack((self.fitness, offspring_fitness))
            
            fronts = _fast_nondominated_sort(combined_fitness)
            crowding = np.zeros(len(combined_fitness))
            for front in fronts:
                crowding[front] = _calculate_crowding_distance(combined_fitness, front)
            
            new_pop = []
            new_fitness = []
            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(combined_pop[front])
                    new_fitness.extend(combined_fitness[front])
                else:
                    front_sorted = sorted(front, key=lambda x: crowding[x], reverse=True)
                    selected = front_sorted[:self.pop_size - len(new_pop)]
                    new_pop.extend(combined_pop[selected])
                    new_fitness.extend(combined_fitness[selected])
                    break
            
            if len(new_pop) == 0:
                new_pop = combined_pop[:self.pop_size].tolist()
                new_fitness = combined_fitness[:self.pop_size].tolist()
            
            self.pop = np.array(new_pop[:self.pop_size])
            self.fitness = np.array(new_fitness[:self.pop_size])
            
            # 计算奖励
            current_fronts = _fast_nondominated_sort(self.fitness)
            current_pareto_size = len(current_fronts[0]) if len(current_fronts) > 0 else 0
            reward = (current_pareto_size - old_pareto_size) * 0.1
            old_pareto_size = current_pareto_size
            
            # 更新DQN
            self._update_dqn(reward)
        
        # 提取Pareto前沿
        if len(self.fitness) == 0:
            return np.array([])
        fronts = _fast_nondominated_sort(self.fitness)
        final_pareto = self.fitness[fronts[0]] if len(fronts) > 0 and len(fronts[0]) > 0 else np.array([])
        return final_pareto

# ---------------------- 3.5 RL-MOEA1（适配ZDT连续变量） ----------------------
class RLMOEA1_ZDT:
    def __init__(self, problem, pop_size=50, max_iter=200, crossover_prob=0.8, mutation_prob=0.1):
        self.problem = problem
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.pop = self._init_population()
        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.pop])

    def _init_population(self):
        """初始化种群（适配ZDT连续变量）"""
        pop = []
        for _ in range(self.pop_size):
            ind = self.problem.init_individual()
            pop.append(ind)
        return np.array(pop)

    def _polynomial_mutation(self, individual):
        """多项式变异"""
        mutated = individual.copy()
        n_vars = len(mutated)
        eta = 20
        for i in range(n_vars):
            if random.random() < self.mutation_prob:
                low = self.problem.bounds[i][0]
                high = self.problem.bounds[i][1]
                u = np.random.random()
                if u <= 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                mutated[i] += delta * (high - low)
                mutated[i] = np.clip(mutated[i], low, high)
        return mutated

    def run(self):
        """运行简化版RL-MOEA"""
        def _fast_nondominated_sort(fitness):
            pop_size = len(fitness)
            if pop_size == 0:
                return []
            domination_count = np.zeros(pop_size, dtype=int)
            dominated_solutions = [[] for _ in range(pop_size)]
            ranks = np.zeros(pop_size, dtype=int)
            fronts = [[]]
            for i in range(pop_size):
                for j in range(pop_size):
                    if i == j:
                        continue
                    if (fitness[i][0] <= fitness[j][0] and fitness[i][1] <= fitness[j][1]) and \
                            (fitness[i][0] < fitness[j][0] or fitness[i][1] < fitness[j][1]):
                        dominated_solutions[i].append(j)
                    elif (fitness[j][0] <= fitness[i][0] and fitness[j][1] <= fitness[i][1]) and \
                            (fitness[j][0] < fitness[i][0] or fitness[j][1] < fitness[i][1]):
                        domination_count[i] += 1
                if domination_count[i] == 0:
                    ranks[i] = 0
                    fronts[0].append(i)
            current_rank = 0
            while len(fronts[current_rank]) > 0:
                next_front = []
                for i in fronts[current_rank]:
                    for j in dominated_solutions[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            ranks[j] = current_rank + 1
                            next_front.append(j)
                current_rank += 1
                fronts.append(next_front)
            return [f for f in fronts if len(f) > 0]
        
        for _ in range(self.max_iter):
            if len(self.fitness) == 0:
                break
            
            # 自适应权重选择父代
            fitness_max = np.max(self.fitness, axis=0)
            fitness_max = np.where(fitness_max == 0, 1, fitness_max)
            fitness_norm = self.fitness / fitness_max
            weights = 1 - (fitness_norm[:, 0] + fitness_norm[:, 1]) / 2
            weights = np.maximum(weights, 0)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(weights)) / len(weights)
            parent_indices = np.random.choice(len(self.pop), size=self.pop_size, p=weights)
            parents = self.pop[parent_indices]

            # SBX交叉
            offspring = []
            for i in range(0, self.pop_size, 2):
                if i + 1 >= self.pop_size:
                    break
                parent1, parent2 = parents[i], parents[i + 1]
                # SBX交叉
                if random.random() < self.crossover_prob:
                    n_vars = len(parent1)
                    child1 = parent1.copy()
                    child2 = parent2.copy()
                    eta = 20
                    for j in range(n_vars):
                        if parent1[j] != parent2[j]:
                            low = self.problem.bounds[j][0]
                            high = self.problem.bounds[j][1]
                            u = np.random.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (eta + 1))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                            c1 = 0.5 * ((1 + beta) * parent1[j] + (1 - beta) * parent2[j])
                            c2 = 0.5 * ((1 - beta) * parent1[j] + (1 + beta) * parent2[j])
                            child1[j] = np.clip(c1, low, high)
                            child2[j] = np.clip(c2, low, high)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                offspring.append(self._polynomial_mutation(child1))
                offspring.append(self._polynomial_mutation(child2))
            offspring = np.array(offspring[:self.pop_size])

            # 评估子代
            offspring_fitness = np.array([self.problem.evaluate(ind) for ind in offspring])

            # 合并选择
            combined_pop = np.vstack((self.pop, offspring))
            combined_fitness = np.vstack((self.fitness, offspring_fitness))
            fronts = _fast_nondominated_sort(combined_fitness)

            new_pop = []
            new_fitness = []
            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(combined_pop[front])
                    new_fitness.extend(combined_fitness[front])
                else:
                    break
            if len(new_pop) == 0:
                new_pop = combined_pop[:self.pop_size].tolist()
                new_fitness = combined_fitness[:self.pop_size].tolist()
            self.pop = np.array(new_pop[:self.pop_size])
            self.fitness = np.array(new_fitness[:self.pop_size])

        # 提取Pareto前沿
        if len(self.fitness) == 0:
            return np.array([])
        fronts = _fast_nondominated_sort(self.fitness)
        final_pareto = self.fitness[fronts[0]] if len(fronts) > 0 and len(fronts[0]) > 0 else np.array([])
        return final_pareto

# ====================== 4. 评价指标计算（IGD、HV） ======================
class MOEAMetrics_ZDT:
    @staticmethod
    def igd(pareto_front, true_front):
        """计算IGD指标（越小越好）"""
        if len(pareto_front) == 0 or len(true_front) == 0:
            return np.inf
        igd_sum = 0.0
        for tf in true_front:
            distances = np.linalg.norm(pareto_front - tf, axis=1)
            igd_sum += np.min(distances)
        return igd_sum / len(true_front)

    @staticmethod
    def hypervolume(pareto_front, ref_point):
        """计算HV指标（越大越好）"""
        if len(pareto_front) == 0:
            return 0.0
        sorted_pareto = pareto_front[np.argsort(pareto_front[:, 0])]
        hv = 0.0
        prev_y = ref_point[1]
        for point in sorted_pareto:
            x, y = point[0], point[1]
            if y > prev_y:
                continue
            width = ref_point[0] - x
            height = prev_y - y
            hv += width * height
            prev_y = y
        return hv

# ====================== 5. ZDT问题算法对比主逻辑 ======================
def run_zdt_algorithm_comparison(zdt_name="ZDT1", n_vars=30, pop_size=50, max_iter=200):
    """
    运行ZDT问题的MOO算法对比
    :param zdt_name: ZDT问题名称（ZDT1/ZDT2/ZDT3/ZDT4/ZDT6）
    :param n_vars: 决策变量维度
    :param pop_size: 种群大小
    :param max_iter: 最大迭代次数
    :return: 指标数据框、算法结果
    """
    # 1. 初始化ZDT问题
    print(f"============= 开始 {zdt_name} 问题算法对比 =============")
    problem = ZDTProblem(problem_name=zdt_name, n_vars=n_vars)
    print(f"问题维度：{n_vars}，种群大小：{pop_size}，最大迭代：{max_iter}")

    # 2. 加载真实Pareto前沿
    try:
        true_pareto_front = load_zdt_true_front(zdt_name=zdt_name)
        print(f"成功加载 {zdt_name} 真实Pareto前沿，共{len(true_pareto_front)}个点")
    except Exception as e:
        print(f"加载真实前沿失败：{e}")
        true_pareto_front = None

    # 3. 初始化算法
    algorithms = {
        "NSGA2": StandardNSGA2_ZDT(problem, pop_size=pop_size, max_iter=max_iter),
        "MOEAD": StandardMOEAD_ZDT(problem, pop_size=pop_size, max_iter=max_iter),
        "SPEA3": StandardSPEA3_ZDT(problem, pop_size=pop_size, max_iter=max_iter),
        "RL-MOEA1": RLMOEA1_ZDT(problem, pop_size=pop_size, max_iter=max_iter),
        "RL-MOEA2": RLMOEA2_ZDT(problem, pop_size=pop_size, max_iter=max_iter)
    }

    # 4. 运行所有算法
    results = {}
    print("\n开始运行算法...")
    for algo_name, algo in algorithms.items():
        print(f"正在运行 {algo_name}...")
        pareto_front = algo.run()
        results[algo_name] = pareto_front
        print(f"{algo_name} 完成，得到{len(pareto_front)}个Pareto解")
    print("所有算法运行完成！")

    # 5. 计算评价指标
    metrics = {}
    if true_pareto_front is not None:
        # 生成参考点（HV计算用，真实前沿最大值+1）
        ref_point = np.max(true_pareto_front, axis=0) + 1
        print(f"\nHV计算参考点：{ref_point}")

        for algo_name, pf in results.items():
            # 计算IGD
            igd_value = MOEAMetrics_ZDT.igd(pf, true_pareto_front)
            # 计算HV
            hv_value = MOEAMetrics_ZDT.hypervolume(pf, ref_point)
            metrics[algo_name] = {
                "IGD": round(igd_value, 6),
                "HV": round(hv_value, 6),
                "Pareto解数量": len(pf)
            }
    else:
        # 无真实前沿时，仅统计Pareto解数量
        for algo_name, pf in results.items():
            metrics[algo_name] = {
                "IGD": "无真实前沿",
                "HV": "无真实前沿",
                "Pareto解数量": len(pf)
            }

    # 6. 生成指标对比表格
    metrics_df = pd.DataFrame(metrics).T
    print("\n===================== 算法性能指标对比 =====================")
    print(metrics_df)

    # 7. 可视化结果（汉字标注 + 宋体 + Morandi palette)
    colors = MORANDI_COLORS
    markers = MORANDI_MARKERS
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=ZDT_FIG_SIZE)
    fig.suptitle(
        f"{zdt_name}: 多目标算法对比",
        fontsize=ZDT_FONT["suptitle"],
        fontweight="bold",
        y=0.995,
    )

    # 7.1 Pareto front comparison (with true front)
    _style_zdt_axis(
        ax1,
        "近似Pareto前沿",
        xlabel="目标1 ($f_1$)",
        ylabel="目标2 ($f_2$)",
    )
    ax1.grid(True, alpha=0.45, linestyle="--", color=MORANDI_GRID)

    if true_pareto_front is not None:
        ax1.plot(
            true_pareto_front[:, 0],
            true_pareto_front[:, 1],
            color=MORANDI_TRUE_PF,
            linewidth=2.2,
            label="真实Pareto前沿",
            alpha=1.0,
        )
    for algo_name, pf in results.items():
        if len(pf) > 0:
            ax1.scatter(
                pf[:, 0],
                pf[:, 1],
                c=colors[algo_name],
                marker=markers[algo_name],
                label=algo_name,
                alpha=0.92,
                s=64,
                edgecolors=MORANDI_EDGE,
                linewidths=0.6,
            )
    ax1.legend(loc="best", framealpha=0.92, fontsize=ZDT_FONT["legend"])

    # 7.2 IGD comparison
    _style_zdt_axis(ax2, "IGD（越小越好）", ylabel="IGD")
    algo_names = list(metrics.keys())
    if true_pareto_front is not None:
        igd_values = [metrics[name]["IGD"] for name in algo_names]
        _draw_bars_with_value_labels(ax2, algo_names, igd_values, colors)
    else:
        ax2.text(
            0.5,
            0.5,
            "真实前沿不可用；\nIGD未计算",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=ZDT_FONT["note"],
        )
    ax2.grid(True, alpha=0.45, axis="y", linestyle="--", color=MORANDI_GRID)

    # 7.3 HV comparison
    _style_zdt_axis(ax3, "超体积（越大越好）", ylabel="超体积")
    if true_pareto_front is not None:
        hv_values = [metrics[name]["HV"] for name in algo_names]
        _draw_bars_with_value_labels(ax3, algo_names, hv_values, colors)
    else:
        ax3.text(
            0.5,
            0.5,
            "真实前沿不可用；\nHV未计算",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=ZDT_FONT["note"],
        )
    ax3.grid(True, alpha=0.45, axis="y", linestyle="--", color=MORANDI_GRID)

    # 7.4 Number of non-dominated solutions
    _style_zdt_axis(ax4, "Pareto解集规模", ylabel="解的数量")
    count_values = [metrics[name]["Pareto解数量"] for name in algo_names]
    _draw_bars_with_value_labels(
        ax4, algo_names, count_values, colors, value_formatter=lambda v: f"{int(v)}"
    )
    ax4.grid(True, alpha=0.45, axis="y", linestyle="--", color=MORANDI_GRID)

    fig.tight_layout(rect=[0, 0.02, 1, 0.990], pad=1.0, h_pad=1.0, w_pad=2.0)
    save_img_name = f"{zdt_name}_comparison_results_中文版.png"
    plt.savefig(save_img_name, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.show()
    print(f"\n对比图片已保存：{save_img_name}")

    # 8. 保存指标表格
    save_csv_name = f"{zdt_name}_算法性能指标_中文版.csv"
    metrics_df.to_csv(save_csv_name, encoding="utf-8-sig")
    print(f"指标表格已保存：{save_csv_name}")

    return metrics_df, results

# ====================== 6. 主函数 ======================
if __name__ == "__main__":
    # Change zdt_name to run a specific benchmark: ZDT1 / ZDT2 / ZDT3 / ZDT4 / ZDT6
    zdt_name = "ZDT4"  # e.g. set to "ZDT2" for ZDT2 experiments
    n_vars = 30  # ZDT问题默认30维
    pop_size = 50
    max_iter = 500

    # 运行对比
    metrics_df, results = run_zdt_algorithm_comparison(
        zdt_name=zdt_name,
        n_vars=n_vars,
        pop_size=pop_size,
        max_iter=max_iter
    )