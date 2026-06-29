import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from matplotlib import rcParams
import warnings
import os
import geopandas as gpd
import fiona
from math import radians, cos, sin, asin, sqrt

warnings.filterwarnings('ignore')

# ====================== 1. 全局配置 ======================
# Publication-quality global style (Times New Roman, English labels)
plt.style.use("default")

np.random.seed(42)
random.seed(42)

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9,
    "axes.unicode_minus": False,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.linewidth": 1.0,

    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    "legend.fontsize": 8,
    "legend.frameon": False,

    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.format": "png",
})


# ====================== 2. 问题定义：最大覆盖选址问题 ======================
class MaxCoverLocationProblem:
    def __init__(self, candidate_sites, demand_points, coverage_radius, build_costs, demand_weights): # 输入:候选选址点坐标、需求点坐标、覆盖半径、建设成本、需求权重
        self.candidate_sites = np.array(candidate_sites) # 候选选址点坐标
        self.demand_points = np.array(demand_points) # 需求点坐标
        self.coverage_radius = coverage_radius # 覆盖半径
        self.build_costs = np.array(build_costs) # 建设成本
        self.demand_weights = np.array(demand_weights) # 需求权重
        self.n_candidates = len(candidate_sites) # 候选选址点数量
        self.n_demands = len(demand_points) # 需求点数量
        self.coverage_matrix = self._compute_coverage_matrix() # 覆盖矩阵

    def _compute_coverage_matrix(self): # 计算覆盖矩阵
        """
        计算覆盖矩阵，使用地理距离（大圆距离）计算经纬度之间的距离
        coverage_radius单位：公里
        """

        def haversine_distance(lon1, lat1, lon2, lat2):
            """
            计算两点之间的地理距离（大圆距离）
            参数：经度1, 纬度1, 经度2, 纬度2（单位：度）
            返回：距离（单位：公里）
            """
            # 转换为弧度
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            # Haversine公式
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            # 地球半径（公里）
            r = 6371.0
            return c * r

        coverage = np.zeros((self.n_candidates, self.n_demands)) # 为覆盖关系分配存储空间，初始化为全零，后续根据实际距离填充 0/1 值，是整个最大覆盖选址问题的核心数据结构
        for i in range(self.n_candidates):
            for j in range(self.n_demands):
                # 使用地理距离计算
                dist = haversine_distance(
                    self.candidate_sites[i][0], self.candidate_sites[i][1],
                    self.demand_points[j][0], self.demand_points[j][1]
                )
                if dist <= self.coverage_radius:
                    coverage[i, j] = 1 # 如果距离小于等于覆盖半径，则认为该候选选址点覆盖了该需求点，覆盖矩阵中对应位置的值为1
        return coverage

    def evaluate(self, solution): # 评估解决方案的适应度，solution为二进制数组，1表示选中该候选点，0表示未选中
        build_cost = np.sum(solution * self.build_costs) # 计算建设成本
        covered_demands = np.any(self.coverage_matrix[solution == 1], axis=0) # 检查哪些需求点被覆盖了，axis=0表示按列检查，即检查哪些需求点被覆盖了
        uncovered_demand = np.sum(self.demand_weights * (1 - covered_demands)) # 计算未覆盖需求，self.demand_weights 的取值来源于 上海市社区人口数据 Shapefile 中的 average_po 字段，存储的是每个需求点（社区）的 平均人口数（或人口权重）
        return np.array([uncovered_demand, build_cost])

    def save(self, filepath):
        """
        保存problem对象到npz文件
        """
        np.savez(
            filepath,
            candidate_sites=self.candidate_sites,
            demand_points=self.demand_points,
            coverage_radius=self.coverage_radius,
            build_costs=self.build_costs,
            demand_weights=self.demand_weights,
            coverage_matrix=self.coverage_matrix,
            n_candidates=self.n_candidates,
            n_demands=self.n_demands
        )
        print(f"problem对象已保存到: {filepath}")

    @classmethod # cls 是 Python 类方法（@classmethod）的第一个参数，代表 类本身（而不是类的实例）。类似于实例方法的 self 代表当前实例，cls 代表当前类。调用类方法时，Python 会自动将类作为第一个参数传入，无需手动提供。
    def load(cls, filepath):
        """
        从npz文件加载problem对象
        从 .npz 文件中读取所有属性，并绕过 __init__ 正常流程（因为 __init__ 会重新计算覆盖矩阵）。
        通过 cls.__new__(cls) 创建一个空实例，然后手动填充属性。
        这样加载后，实例可以直接用于 evaluate 和算法优化。
        """
        data = np.load(filepath, allow_pickle=False)
        # 创建problem对象，但跳过coverage_matrix的计算
        problem = cls.__new__(cls)
        problem.candidate_sites = data['candidate_sites']
        problem.demand_points = data['demand_points']
        problem.coverage_radius = float(data['coverage_radius'])
        problem.build_costs = data['build_costs']
        problem.demand_weights = data['demand_weights']
        problem.coverage_matrix = data['coverage_matrix']
        problem.n_candidates = int(data['n_candidates'])
        problem.n_demands = int(data['n_demands'])
        print(f"problem对象已从文件加载: {filepath}")
        return problem

# ====================== 3. 算法实现 ======================
# 示例：添加DQN策略网络实现自适应算子选择
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module): # 这是一个多层感知机（MLP），属于深度学习中最基础的神经网络结构,没有强化学习成分，但是定义为DQN类是为了与RL-MOEA2算法中的DQN类保持一致，方便后续的集成。
    def __init__(self, state_dim, action_dim): #定义了 DQN 网络的三个全连接层，分别是第一隐藏层（输入层）、第二隐藏层和输出层
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64) # 在调用时输入维度为 4 来源于算法设计者选择的 4 个状态特征（平均未覆盖需求、平均建设成本、种群多样性、迭代进度）
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim) # 在调用时输出维度为 3，对应 3 种交叉算子（单点交叉、两点交叉、均匀交叉）

    def forward(self, x): # 前两行对前两个全连接层的输出应用 ReLU 激活函数，引入非线性（本质分段线性函数）。最后一行是输出层，直接输出 Q 值，Q 值可以是负数
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 在RL-MOEA中集成DQN：状态=种群分布+收敛进度，动作=选择交叉/变异算子
class RLMOEA2:
    def __init__(self, problem, pop_size=80, max_iter=300): # 敏感性分析最优参数: P=80, G=300
        self.problem = problem
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.mutation_prob = 0.05 # 变异概率（敏感性分析最优值：0.05→HV=66.41G vs 0.10→61.85G，+7.4%）

        # 初始化DQN
        self.state_dim = 4  # 状态：适应度均值、成本、多样性、迭代进度
        self.action_dim = 3  # 动作：单点交叉/两点交叉/均匀交叉
        self.dqn = DQN(self.state_dim, self.action_dim)
        # ===== DQNMOEA 改进1：目标网络（Target Network）=====
        self.target_dqn = DQN(self.state_dim, self.action_dim)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.0005) # Adam学习率（敏感性分析最优值：5e-4→HV=62.39G）
        self.criterion = nn.MSELoss()
        self.epsilon = 0.85  # 探索率（敏感性分析最优值：0.85→HV=63.21G vs 0.90→62.24G）
        self.epsilon_decay = 0.995 # 探索率衰减率，0.995表示每次迭代后探索率衰减0.005，即每次迭代后探索率变为原来的0.995倍。
        self.epsilon_min = 0.1 # 探索率最小值，0.1表示探索率不能低于0.1，即每次迭代后探索率不能小于0.1。

        # ===== DQNMOEA 改进2：经验回放缓冲区（Experience Replay）=====
        self.replay_buffer = []
        self.max_replay_size = 500
        self.batch_size = 16
        self.gamma = 0.80  # 折扣因子（敏感性分析最优值：0.80→HV=62.85G vs 0.90→61.95G）
        self.target_update_freq = 10
        self.update_counter = 0
        self.state_before_action = None

        # ===== DQNMOEA 改进3：超体积参考点（Hypervolume Reference Point）=====
        total_demand = np.sum(self.problem.demand_weights)
        total_cost = np.sum(self.problem.build_costs)
        self.hv_ref_point = np.array([total_demand * 1.1, total_cost * 1.1])

        # 初始化种群
        self.pop = self._init_population()
        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.pop]) # 评估初始种群的适应度

        # 用于状态计算的变量
        self.current_iter = 0 # 用于存储当前迭代次数，用于计算迭代进度。
        self.last_action = 0 # 用于存储上一次选择的动作，用于计算奖励更新。
        self.last_reward = 0.0 # 用于存储上一次选择的动作的奖励值，用于计算奖励更新。

        # 用于归一化的参考值
        self.max_fitness = np.max(self.fitness, axis=0) if len(self.fitness) > 0 else np.array([1.0, 1.0]) # 用于归一化适应度上界，将适应度归一化到0-1之间。
        self.min_fitness = np.min(self.fitness, axis=0) if len(self.fitness) > 0 else np.array([0.0, 0.0]) # 用于归一化适应度下界

    def _init_population(self):
        pop = []
        greedy_sol = self._greedy_max_cover()
        pop.append(greedy_sol) # 添加贪心解作为初始种群的第一个解。
        while len(pop) < self.pop_size:
            sol = np.random.randint(0, 2, self.problem.n_candidates) # 随机生成一个二进制数组，表示一个解。
            n_selected = np.sum(sol) # 计算解中选中的候选点数量。
            if 1 <= n_selected <= self.problem.n_candidates // 3:
                pop.append(sol) # 如果解中选中的候选点数量在1到候选点总数1/3之间，则添加到种群中。
        return np.array(pop)

    def _greedy_max_cover(self): # 贪心最大覆盖初始化，在不超出设施数量上限的前提下，尽可能多地覆盖需求点 
        sol = np.zeros(self.problem.n_candidates, dtype=int)
        uncovered = np.ones(self.problem.n_demands, dtype=bool) # 布尔类型符合目标函数的指示
        while np.any(uncovered):
            cover_counts = np.sum(self.problem.coverage_matrix[:, uncovered], axis=1)
            if np.max(cover_counts) == 0:
                break
            best_idx = np.argmax(cover_counts)
            sol[best_idx] = 1
            uncovered = uncovered & ~(self.problem.coverage_matrix[best_idx] == 1)
            if np.sum(sol) >= self.problem.n_candidates // 3:
                break
        return sol

    def _compute_hypervolume(self, pareto_fitness):
        """计算2目标最小化问题的超体积（Hypervolume Indicator）
        超体积同时衡量Pareto前沿的收敛性和分布性，是多目标优化中最权威的综合指标。"""
        if len(pareto_fitness) == 0:
            return 0.0
        mask = (pareto_fitness[:, 0] <= self.hv_ref_point[0]) & \
               (pareto_fitness[:, 1] <= self.hv_ref_point[1])
        points = pareto_fitness[mask]
        if len(points) == 0:
            return 0.0
        sorted_idx = np.argsort(points[:, 0])
        sorted_points = points[sorted_idx]

        hv = 0.0
        best_f2 = self.hv_ref_point[1]
        for p in sorted_points:
            if p[1] < best_f2:
                hv += (self.hv_ref_point[0] - p[0]) * (best_f2 - p[1])
                best_f2 = p[1]
        return hv

    def _get_state(self):
        # 构造状态向量：归一化的未覆盖需求、建设成本、解多样性、迭代进度
        if len(self.fitness) == 0:
            return torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

        # 适应度归一化
        fitness_range = self.max_fitness - self.min_fitness
        fitness_range = np.where(fitness_range == 0, 1, fitness_range) # 防止后续除法出现除零错误
        fitness_norm = (self.fitness - self.min_fitness) / fitness_range
        avg_fitness = np.mean(fitness_norm, axis=0)

        # 多样性（标准差）
        diversity = np.std(self.fitness, axis=0).mean()
        diversity_norm = diversity / (np.max(self.fitness) - np.min(self.fitness) + 1e-6)

        # 迭代进度
        iter_progress = self.current_iter / self.max_iter

        return torch.tensor([avg_fitness[0], avg_fitness[1], diversity_norm, iter_progress], dtype=torch.float32)

    def _select_operator(self):
        # ε-贪心选择动作（算子）
        # ===== DQNMOEA 改进4：先记录state_before_action，形成正确的(s,a,r,s')经验元组 =====
        state = self._get_state()
        self.state_before_action = state
        if random.random() < self.epsilon:
            action = random.choice(range(self.action_dim))
        else:
            q_values = self.dqn(state)
            action = torch.argmax(q_values).item()
        self.last_action = action
        return action # 返回选择的动作（是一个整数，取值范围为 0、1 或 2），用于后续的交叉算子选择。

    def _crossover_single_point(self, parent1, parent2): # 选择这三种交叉算子是基于 编码类型（二进制）、搜索行为多样性、DQN 学习可行性、实现简单性 以及 经典有效性 的综合考虑
        """单点交叉"""
        cross_point = random.randint(1, self.problem.n_candidates - 1)
        child1 = np.hstack((parent1[:cross_point], parent2[cross_point:]))
        child2 = np.hstack((parent2[:cross_point], parent1[cross_point:]))
        return child1, child2

    def _crossover_two_point(self, parent1, parent2):
        """两点交叉"""
        point1 = random.randint(1, self.problem.n_candidates - 2)
        point2 = random.randint(point1 + 1, self.problem.n_candidates - 1)
        child1 = np.hstack((parent1[:point1], parent2[point1:point2], parent1[point2:]))
        child2 = np.hstack((parent2[:point1], parent1[point1:point2], parent2[point2:]))
        return child1, child2

    def _crossover_uniform(self, parent1, parent2):
        """均匀交叉"""
        mask = np.random.randint(0, 2, self.problem.n_candidates)
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    def _bit_flip_mutation(self, individual):
        """位翻转变异"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_prob: # 以 self.mutation_prob（0.1）的概率翻转每一位
                mutated[i] = 1 - mutated[i]
        n_selected = np.sum(mutated)
        if n_selected == 0:
            mutated[random.randint(0, self.problem.n_candidates - 1)] = 1 # 如果解中选中的候选点数量为0，则随机选择一个候选点选中。
        elif n_selected > self.problem.n_candidates // 3:
            selected_idx = np.where(mutated == 1)[0]
            delete_idx = random.sample(list(selected_idx), n_selected - (self.problem.n_candidates // 3)) # 如果解中选中的候选点数量大于候选点总数1/3，则随机选择一些候选点取消选中。
            mutated[delete_idx] = 0
        return mutated # 返回变异后的解。

    def _update_dqn(self):
        """DQNMOEA 改进5：使用经验回放和目标网络的标准化DQN更新
        - 从经验池随机采样mini-batch，打破时序相关性
        - 使用目标网络计算TD目标，稳定Q值更新
        - 周期性同步目标网络权重
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从经验池随机采样
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)

        # 当前Q值: Q(s, a)
        q_values = self.dqn(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 目标Q值: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_dqn(next_states)
            max_next_q = next_q_values.max(1)[0]
            target = rewards + self.gamma * max_next_q

        loss = self.criterion(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 定期同步目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

    def run(self):
        """运行RL-MOEA算法"""

        # 非支配排序函数
        def _fast_nondominated_sort(fitness): 
            pop_size = len(fitness)
            if pop_size == 0:
                return []
            domination_count = np.zeros(pop_size, dtype=int)
            dominated_solutions = [[] for _ in range(pop_size)]
            ranks = np.zeros(pop_size, dtype=int)
            fronts = [[]]
            for i in range(pop_size): # 两层循环：O(MN²)，其中 M 为目标数（这里固定 2），N 为种群大小。对于小规模种群（如 N=50~100）可以接受，大规模种群需改用更高效算法（如基于树的方法）。
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

        # 拥挤度计算
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

        for iter in range(self.max_iter):
            self.current_iter = iter

            # 检查fitness是否为空
            if len(self.fitness) == 0:
                break

            # 更新参考值
            self.max_fitness = np.maximum(self.max_fitness, np.max(self.fitness, axis=0))
            self.min_fitness = np.minimum(self.min_fitness, np.min(self.fitness, axis=0))

            # ===== DQNMOEA 改进3：基于超体积的奖励计算 =====
            # 计算动作前的Pareto前沿超体积作为基准
            old_fronts = _fast_nondominated_sort(self.fitness)
            if len(old_fronts) > 0 and len(old_fronts[0]) > 0:
                old_hv = self._compute_hypervolume(self.fitness[old_fronts[0]])
            else:
                old_hv = 0.0

            # DQN选择交叉算子（内部记录state_before_action）
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

            # 根据选择的动作执行交叉
            offspring = []
            for i in range(0, self.pop_size, 2):
                if i + 1 >= self.pop_size:
                    break
                parent1, parent2 = parents[i], parents[i + 1]

                if action == 0:
                    child1, child2 = self._crossover_single_point(parent1, parent2)
                elif action == 1:
                    child1, child2 = self._crossover_two_point(parent1, parent2)
                else:
                    child1, child2 = self._crossover_uniform(parent1, parent2)

                offspring.append(self._bit_flip_mutation(child1))
                offspring.append(self._bit_flip_mutation(child2))
            offspring = np.array(offspring[:self.pop_size])

            # 评估子代
            offspring_fitness = np.array([self.problem.evaluate(ind) for ind in offspring])

            # 合并父代和子代
            combined_pop = np.vstack((self.pop, offspring))
            combined_fitness = np.vstack((self.fitness, offspring_fitness))

            # 环境选择：非支配排序+拥挤度选择，全局排序，逐层填充，用来筛选下一代
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

            # 确保至少有一些个体
            if len(new_pop) == 0:
                new_pop = combined_pop[:self.pop_size].tolist()
                new_fitness = combined_fitness[:self.pop_size].tolist()

            self.pop = np.array(new_pop[:self.pop_size])
            self.fitness = np.array(new_fitness[:self.pop_size])

            # ===== DQNMOEA 改进3：基于超体积改进的奖励 =====
            # 超体积同时衡量Pareto前沿的收敛性和分布性
            new_fronts = _fast_nondominated_sort(self.fitness)
            if len(new_fronts) > 0 and len(new_fronts[0]) > 0:
                new_hv = self._compute_hypervolume(self.fitness[new_fronts[0]])
            else:
                new_hv = 0.0
            # 超体积相对改进率作为奖励（正值=改进，负值=退化）
            reward = (new_hv - old_hv) / (old_hv + 1e-6)

            # ===== DQNMOEA 改进4：存储经验元组 (s, a, r, s') =====
            state_after = self._get_state()
            if self.state_before_action is not None:
                self.replay_buffer.append(
                    (self.state_before_action, self.last_action, reward, state_after)
                )
                if len(self.replay_buffer) > self.max_replay_size:
                    self.replay_buffer.pop(0)

            # ===== DQNMOEA 改进5：标准化DQN更新 =====
            self._update_dqn()

        # 提取帕累托前沿
        if len(self.fitness) == 0:
            return np.array([])
        fronts = _fast_nondominated_sort(self.fitness)
        if len(fronts) > 0 and len(fronts[0]) > 0:
            final_pareto = self.fitness[fronts[0]]
        else:
            final_pareto = np.array([])
        return final_pareto


# ====================== 4. 数据加载与主函数 ======================
def load_shanghai_data():
    """
    加载上海的服务设施供需数据
    返回：候选点坐标、需求点坐标、覆盖半径、建设成本、需求权重
    """
    # 数据文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_file = os.path.join(current_dir, "shanghai_data_processer", "垂直起降场候选选址结果.xlsx")
    demand_file = os.path.join(current_dir, "shanghai_data_processer", "community_population_20220301_having_pops.shp")

    print("=" * 60)
    print("开始加载上海数据...")
    print("=" * 60)

    # 1. 读取候选点数据（Excel）
    print(f"\n步骤1: 读取候选点数据...")
    print(f"文件路径: {candidate_file}")
    if not os.path.exists(candidate_file):
        raise FileNotFoundError(f"候选点文件不存在: {candidate_file}")

    candidate_df = pd.read_excel(candidate_file, engine='openpyxl')
    print(f"候选点数据形状: {candidate_df.shape}")
    print(f"候选点列名: {list(candidate_df.columns)}")

    # 检查并提取所需字段
    if 'FID' not in candidate_df.columns:
        raise ValueError(f"候选点文件中未找到FID列，可用列名: {list(candidate_df.columns)}")
    if 'gpsx' not in candidate_df.columns:
        raise ValueError(f"候选点文件中未找到gpsx列，可用列名: {list(candidate_df.columns)}")
    if 'gpsy' not in candidate_df.columns:
        raise ValueError(f"候选点文件中未找到gpsy列，可用列名: {list(candidate_df.columns)}")

    # 提取候选点坐标（经度，纬度）
    candidate_sites = candidate_df[['gpsx', 'gpsy']].values
    n_candidates = len(candidate_sites)
    print(f"候选点数量: {n_candidates}")

    # 2. 读取需求点数据（Shapefile）
    print(f"\n步骤2: 读取需求点数据...")
    print(f"文件路径: {demand_file}")
    if not os.path.exists(demand_file):
        raise FileNotFoundError(f"需求点文件不存在: {demand_file}")

    # 尝试不同的编码方式读取shp文件
    encodings_to_try = ['utf-8', 'gbk', 'gb18030', 'latin-1', 'cp936']
    demand_gdf = None

    for encoding in encodings_to_try:
        try:
            with fiona.open(demand_file, encoding=encoding) as src:
                demand_gdf = gpd.GeoDataFrame.from_features(src, crs=src.crs)
            print(f"成功使用 {encoding} 编码读取文件")
            break
        except (UnicodeDecodeError, Exception):
            if encoding == encodings_to_try[-1]:
                try:
                    demand_gdf = gpd.read_file(demand_file)
                    break
                except Exception:
                    raise ValueError("无法读取需求点shp文件")
            continue

    if demand_gdf is None:
        raise ValueError("无法读取需求点shp文件")

    print(f"需求点数据形状: {demand_gdf.shape}")
    print(f"需求点列名: {list(demand_gdf.columns)}")

    # 检查并提取所需字段
    if 'objectid_1' not in demand_gdf.columns:
        raise ValueError(f"需求点文件中未找到objectid_1列，可用列名: {list(demand_gdf.columns)}")
    if 'gpsx' not in demand_gdf.columns:
        raise ValueError(f"需求点文件中未找到gpsx列，可用列名: {list(demand_gdf.columns)}")
    if 'gpsy' not in demand_gdf.columns:
        raise ValueError(f"需求点文件中未找到gpsy列，可用列名: {list(demand_gdf.columns)}")
    if 'average_po' not in demand_gdf.columns:
        raise ValueError(f"需求点文件中未找到average_po列，可用列名: {list(demand_gdf.columns)}")

    # 提取需求点坐标（经度，纬度）
    demand_points = demand_gdf[['gpsx', 'gpsy']].values
    n_demands = len(demand_points)
    print(f"需求点数量: {n_demands}")

    # 提取需求权重（平均人数）
    demand_weights = demand_gdf['average_po'].values
    print(f"需求权重范围: [{np.min(demand_weights):.2f}, {np.max(demand_weights):.2f}]")

    # 3. 设置参数
    coverage_radius = 5  # 5公里
    build_costs = np.ones(n_candidates)  # 全部为1

    print(f"\n参数设置:")
    print(f"  覆盖半径: {coverage_radius} 公里")
    print(f"  建设成本: 全部为 {build_costs[0]}")
    print(f"  候选点数量: {n_candidates}")
    print(f"  需求点数量: {n_demands}")

    return candidate_sites, demand_points, coverage_radius, build_costs, demand_weights


def run_algorithm_RLMOEA2():
    # 定义problem保存文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    problem_file = os.path.join(current_dir, "shanghai_problem.npz")

    # 1. 尝试加载已保存的problem对象，如果不存在则创建并保存
    if os.path.exists(problem_file):
        print("=" * 60)
        print("发现已保存的problem文件，正在加载...")
        print("=" * 60)
        problem = MaxCoverLocationProblem.load(problem_file)
        # 从problem对象中获取candidate_sites，用于后续可视化
        candidate_sites = problem.candidate_sites
    else:
        print("=" * 60)
        print("未找到已保存的problem文件，正在从原始数据生成...")
        print("=" * 60)
        # 加载上海数据
        candidate_sites, demand_points, coverage_radius, build_costs, demand_weights = load_shanghai_data()

        # 初始化问题
        problem = MaxCoverLocationProblem(
            candidate_sites, demand_points, coverage_radius, build_costs, demand_weights
        )

        # 保存problem对象以便下次使用
        problem.save(problem_file)
        print("提示：下次运行将自动加载已保存的problem文件，无需重新计算覆盖矩阵")

    # 3. 初始化并运行RL-MOEA2算法
    algo = RLMOEA2(problem, pop_size=80, max_iter=300)  # 敏感性分析最优值: P=80, G=300
    algo_name = "RL-MOEA2"

    print("开始运行算法...")
    pareto_front = algo.run()
    print("算法运行完成！")

    # 从前沿面上平均选取其中4个方案，在候选服务设施点数据的基础上空间制图可视化出来，从而对比出4个方案的选中的候选服务设施的分布情况
    # 将前沿面可视化出来,并着重显示出被选中的四个方案
    print("\n" + "=" * 60)
    print("开始生成帕累托前沿可视化...")
    print("=" * 60)

    # 初始化变量，用于后续空间可视化
    selected_solutions = None
    selected_fitness = None
    n_show = 0

    # 1. 首先可视化帕累托前沿，并标注选中的4个方案
    if len(pareto_front) > 0:
        # 获取最终种群和解
        final_pop = algo.pop  # 最终种群（解）
        final_fitness = algo.fitness  # 最终适应度

        # 找到帕累托前沿对应的解
        pareto_solutions = []
        pareto_fitness_list = []

        for pf_point in pareto_front:
            distances = np.linalg.norm(final_fitness - pf_point, axis=1)
            min_idx = np.argmin(distances)
            if distances[min_idx] < 1e-6:
                pareto_solutions.append(final_pop[min_idx])
                pareto_fitness_list.append(final_fitness[min_idx])

        if len(pareto_solutions) > 0:
            pareto_solutions = np.array(pareto_solutions)
            pareto_fitness_list = np.array(pareto_fitness_list)

            # 按第一目标（未覆盖需求）排序
            sorted_idx = np.argsort(pareto_fitness_list[:, 0])
            pareto_solutions = pareto_solutions[sorted_idx]
            pareto_fitness_list = pareto_fitness_list[sorted_idx]

            # 平均选取4个方案
            n_solutions = len(pareto_solutions)
            n_show = min(4, n_solutions)
            if n_solutions >= 4:
                selected_indices = np.linspace(0, n_solutions - 1, 4, dtype=int)
                selected_solutions = pareto_solutions[selected_indices]
                selected_fitness = pareto_fitness_list[selected_indices]
            else:
                selected_solutions = pareto_solutions
                selected_fitness = pareto_fitness_list

            # 按第一目标排序帕累托前沿用于可视化
            sorted_pareto_idx = np.argsort(pareto_front[:, 0])
            sorted_pareto_front = pareto_front[sorted_pareto_idx]

            # # 创建帕累托前沿可视化图
            # fig_pareto, ax_pareto = plt.subplots(1, 1, figsize=(10, 8))
            #
            # # 绘制所有帕累托前沿点
            # ax_pareto.scatter(sorted_pareto_front[:, 0], sorted_pareto_front[:, 1],
            #                  c='lightblue', s=50, alpha=0.6, label='帕累托前沿解',
            #                  marker='o', edgecolors='blue', linewidths=0.5)
            #
            # # 连接帕累托前沿点形成前沿曲线
            # ax_pareto.plot(sorted_pareto_front[:, 0], sorted_pareto_front[:, 1],
            #               'b-', alpha=0.3, linewidth=1, label='帕累托前沿')
            #
            # # 标注选中的4个方案（用不同颜色和标记突出显示）
            # colors_selected = ['red', 'orange', 'green', 'purple']
            # markers_selected = ['*', 's', '^', 'D']
            # for idx, fitness in enumerate(selected_fitness):
            #     ax_pareto.scatter(fitness[0], fitness[1],
            #                     c=colors_selected[idx], s=300, alpha=0.9,
            #                     marker=markers_selected[idx], edgecolors='darkred',
            #                     linewidths=2, zorder=10,
            #                     label=f'选中方案 {idx+1}')
            #     # 添加文本标注
            #     ax_pareto.annotate(f'方案{idx+1}\n未覆盖:{fitness[0]:.1f}\n成本:{fitness[1]:.0f}',
            #                       xy=(fitness[0], fitness[1]),
            #                       xytext=(10, 10), textcoords='offset points',
            #                       fontsize=9, fontweight='bold',
            #                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
            #                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            #
            # # 设置图表属性
            # ax_pareto.set_xlabel('未覆盖需求（最小化）', fontsize=12, fontweight='bold')
            # ax_pareto.set_ylabel('建设成本（最小化）', fontsize=12, fontweight='bold')
            # ax_pareto.set_title(f'{algo_name} - 帕累托前沿及选中方案', fontsize=14, fontweight='bold')
            # ax_pareto.grid(True, alpha=0.3, linestyle='--')
            # ax_pareto.legend(loc='best', fontsize=9)
            #
            # plt.tight_layout()
            # pareto_plot_file = f"{algo_name}_帕累托前沿可视化.png"
            # plt.savefig(pareto_plot_file, dpi=300, bbox_inches="tight")
            # print(f"帕累托前沿可视化已保存: {pareto_plot_file}")
            # plt.show()
            # ====================== Nature/Science 风格 Pareto Front 绘图 ======================

            # 排版更紧凑（期刊推荐尺寸）
            fig_pareto, ax = plt.subplots(figsize=(4.8, 4.2))

            # 所有Pareto点（灰色小点）
            ax.scatter(sorted_pareto_front[:, 0],
                       sorted_pareto_front[:, 1],
                       s=18,
                       color="gray",
                       alpha=0.7,
                       edgecolor="none")

            # ===== 新增：给每个Pareto点标 solution =====
            #for i, (x, y) in enumerate(sorted_pareto_front):
            #    ax.text(x, y,
            #            f"S{i + 1}",
            #            fontsize=6,
            #            ha='right',
            #            va='bottom',
            #            color='black',
            #            alpha=0.7)

            # 前沿线（黑色细线）
            ax.plot(sorted_pareto_front[:, 0],
                    sorted_pareto_front[:, 1],
                    color="black",
                    linewidth=1.0)

            # 突出显示4个方案（统一强调色）
            # ====================== Nature/Science 风格 Pareto Front 绘图（带图例） ======================

            fig_pareto, ax = plt.subplots(figsize=(4.8, 4.2))

            # 所有Pareto点（灰色）
            ax.scatter(sorted_pareto_front[:, 0],
                       sorted_pareto_front[:, 1],
                       s=18,
                       color="#8DA0CB",
                       alpha=0.7,
                       label="Pareto Solutions",
                       edgecolor="none")

            # Pareto front line
            ax.plot(sorted_pareto_front[:, 0],
                    sorted_pareto_front[:, 1],
                    color="#1B3A6F",
                    linewidth=1.0,
                    label="Pareto Front")

            # Highlight 4 selected schemes
            highlight_color = "#0072B2"

            for i, fit in enumerate(selected_fitness):
                ax.scatter(fit[0], fit[1],
                           s=90,
                           color="#0072B2",
                           edgecolor="black",
                           linewidth=0.8,
                           zorder=5,
                           label="Selected Scheme" if i == 0 else None)

                ax.annotate(
                    f"S{i + 1}",
                    xy=(fit[0], fit[1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color="#1B3A6F",
                    zorder=10
                )

            # Axis labels
            ax.set_xlabel("Uncovered Demand")
            ax.set_ylabel("Construction Cost")

            # Nature风格边框
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # ✅添加图例（右上角）
            ax.legend(loc="upper right", frameon=False)

            plt.tight_layout()

            pareto_plot_file = f"{algo_name}_Pareto_Front.png"
            plt.savefig(pareto_plot_file, dpi=600, bbox_inches="tight")
            print(f"Pareto front figure saved: {pareto_plot_file}")

            plt.show()

    print("\n" + "=" * 60)
    print("开始生成空间可视化...")
    print("=" * 60)

    if len(pareto_front) > 0 and selected_solutions is not None:
        # 创建空间可视化（根据实际方案数量调整布局）
        if n_show == 1:
            fig, axes = plt.subplots(1, 1, figsize=(10, 8))
            axes = [axes]
        elif n_show == 2:
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            axes = axes.flatten()
        elif n_show == 3:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            axes = axes.flatten()
            # 隐藏第4个子图
            axes[3].axis('off')
        else:  # n_show == 4
            fig, axes = plt.subplots(
                2, 2,
                figsize=(10, 8),
                constrained_layout=True  # ✅自动防挤压（推荐）
            )
            axes = axes.flatten()

        # 读取候选点数据用于可视化
        current_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_file = os.path.join(current_dir, "shanghai_data_processer", "垂直起降场候选选址结果.xlsx")
        candidate_df = pd.read_excel(candidate_file, engine='openpyxl')

        # 读取需求点数据用于可视化
        demand_file = os.path.join(current_dir, "shanghai_data_processer",
                                   "community_population_20220301_having_pops.shp")
        encodings_to_try = ['utf-8', 'gbk', 'gb18030', 'latin-1', 'cp936']
        demand_gdf = None
        for encoding in encodings_to_try:
            try:
                with fiona.open(demand_file, encoding=encoding) as src:
                    demand_gdf = gpd.GeoDataFrame.from_features(src, crs=src.crs)
                break
            except (UnicodeDecodeError, Exception):
                if encoding == encodings_to_try[-1]:
                    demand_gdf = gpd.read_file(demand_file)
                continue

        # # 为每个方案绘制地图
        # for idx in range(n_show):
        #     solution = selected_solutions[idx]
        #     fitness = selected_fitness[idx]
        #     ax = axes[idx]
        #
        #     # 获取选中的候选点索引
        #     selected_indices = np.where(solution == 1)[0]
        #     unselected_indices = np.where(solution == 0)[0]
        #
        #     # 绘制所有候选点（未选中的用灰色）
        #     if len(unselected_indices) > 0:
        #         ax.scatter(candidate_df.iloc[unselected_indices]['gpsx'].values,
        #                     candidate_df.iloc[unselected_indices]['gpsy'].values,
        #                     c='lightgray', s=30, alpha=0.5, label='未选中候选点', marker='o')
        #
        #     # 绘制选中的候选点（用红色）
        #     if len(selected_indices) > 0:
        #         ax.scatter(candidate_df.iloc[selected_indices]['gpsx'].values,
        #                     candidate_df.iloc[selected_indices]['gpsy'].values,
        #                     c='red', s=100, alpha=0.8, label='选中候选点', marker='*', edgecolors='darkred', linewidths=1.5)
        #
        #     # 绘制需求点（用浅蓝色）
        #     if demand_gdf is not None:
        #         ax.scatter(demand_gdf['gpsx'], demand_gdf['gpsy'],
        #                     c='lightblue', s=5, alpha=0.3, label='需求点', marker='.')
        #
        #     # 设置标题和标签
        #     ax.set_title(f'方案 {idx+1}\n未覆盖需求: {fitness[0]:.2f}, 建设成本: {fitness[1]:.0f}\n选中候选点数: {len(selected_indices)}',
        #                 fontsize=12, fontweight='bold')
        #     ax.set_xlabel('经度', fontsize=10)
        #     ax.set_ylabel('纬度', fontsize=10)
        #     ax.grid(True, alpha=0.3)
        #     ax.legend(loc='upper right', fontsize=8)
        #
        #     # 设置坐标轴范围
        #     if len(candidate_sites) > 0:
        #         ax.set_xlim([candidate_sites[:, 0].min() - 0.01, candidate_sites[:, 0].max() + 0.01])
        #         ax.set_ylim([candidate_sites[:, 1].min() - 0.01, candidate_sites[:, 1].max() + 0.01])
        # ====================== Nature/Science 风格空间分布绘图（带统一图例） ======================

        highlight_color = "#D55E00"

        for idx in range(n_show):
            ax = axes[idx]
            solution = selected_solutions[idx]

            selected_idx = np.where(solution == 1)[0]

            # Demand points
            ax.scatter(demand_gdf["gpsx"], demand_gdf["gpsy"],
                       s=2,
                       color="lightgray",
                       alpha=0.4,
                       label="Demand Points" if idx == 0 else None)

            # Candidate sites
            ax.scatter(candidate_df["gpsx"], candidate_df["gpsy"],
                       s=6,
                       color="gray",
                       alpha=0.6,
                       label="Candidate Sites" if idx == 0 else None)

            # Selected facilities
            ax.scatter(candidate_df.iloc[selected_idx]["gpsx"],
                       candidate_df.iloc[selected_idx]["gpsy"],
                       s=28,
                       color=highlight_color,
                       edgecolor="black",
                       linewidth=0.4,
                       zorder=5,
                       label="Selected Facilities" if idx == 0 else None)

            ax.set_title(f"Scheme {idx + 1}", fontsize=16)

            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.set_xlabel("Longitude (deg. E)", fontsize=14)
            ax.set_ylabel("Latitude (deg. N)", fontsize=14)
            ax.tick_params(labelsize=12)

        # ✅统一图例放在整张图下方
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels,
                   loc="lower center",
                   ncol=3,
                   frameon=False,
                   fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))

        # ✅保存输出
        output_file = f"{algo_name}_Spatial_Solutions.png"
        plt.savefig(output_file, dpi=600, bbox_inches="tight")
        print(f"Spatial solutions saved: {output_file}")

        plt.show()

        # ====================== 新增：S1 vs S4 局部放大 ======================
        print("\nGenerating zoomed detail maps (S1 vs S4)...")

        if n_show >= 4:
            fig_zoom, axes_zoom = plt.subplots(
                2, 1,
                figsize=(5, 8),
                constrained_layout=True
            )

            zoom_solutions = [selected_solutions[0], selected_solutions[3]]
            titles = ["Scheme 1 (Zoomed)", "Scheme 4 (Zoomed)"]

            for i, sol in enumerate(zoom_solutions):
                ax = axes_zoom[i]

                selected_idx = np.where(sol == 1)[0]
                selected_points = candidate_df.iloc[selected_idx]

                # ===== 中心缩放（更论文风格）=====
                center_x = selected_points["gpsx"].mean()
                center_y = selected_points["gpsy"].mean()

                range_x = (selected_points["gpsx"].max() - selected_points["gpsx"].min())
                range_y = (selected_points["gpsy"].max() - selected_points["gpsy"].min())

                scale = 0.6  # ✅ 缩小范围（0.5~0.7最佳）

                ax.set_xlim(center_x - range_x * scale, center_x + range_x * scale)
                ax.set_ylim(center_y - range_y * scale, center_y + range_y * scale)

                # ===== 绘图 =====
                ax.scatter(demand_gdf["gpsx"], demand_gdf["gpsy"],
                           s=2, color="lightgray", alpha=0.3)

                ax.scatter(candidate_df["gpsx"], candidate_df["gpsy"],
                           s=6, color="gray", alpha=0.4)

                ax.scatter(selected_points["gpsx"], selected_points["gpsy"],
                           s=35, color="#D55E00", edgecolor= "black", linewidth=0.5)

                # ===== 设置放大范围 =====
                ax.set_xlim(center_x - range_x * scale,
                            center_x + range_x * scale)

                ax.set_ylim(center_y - range_y * scale,
                            center_y + range_y * scale)

                ax.set_title(titles[i])
                ax.set_xlabel("Longitude (deg. E)")
                ax.set_ylabel("Latitude (deg. N)")

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            #plt.tight_layout()

            zoom_file = f"{algo_name}_S1_S4_Zoom.png"
            plt.savefig(zoom_file, dpi=600, bbox_inches="tight")
            print(f"Zoom detail saved: {zoom_file}")

            plt.show()

        # 保存选中的4个方案的详细信息到Excel
        solution_data = []
        for idx, (solution, fitness) in enumerate(zip(selected_solutions, selected_fitness)):
            selected_indices = np.where(solution == 1)[0]
            solution_data.append({
                'Scheme': idx + 1,
                'Uncovered_Demand': fitness[0],
                'Construction_Cost': fitness[1],
                'N_Selected': len(selected_indices),
                'Selected_Indices': ','.join(map(str, selected_indices.tolist()))
            })

        solution_df = pd.DataFrame(solution_data)
        solution_df.to_excel(f"{algo_name}_4_Schemes_Detail.xlsx", index=False, engine='openpyxl')
        print(f"Scheme details saved: {algo_name}_4_Schemes_Detail.xlsx")
    elif len(pareto_front) == 0:
        print("警告：帕累托前沿为空，无法进行可视化")
    else:
        print("警告：未能找到帕累托前沿对应的解")

    return pareto_front, algo


# ====================== 5. 主函数 ======================
if __name__ == "__main__":
    pareto_front, algo = run_algorithm_RLMOEA2()