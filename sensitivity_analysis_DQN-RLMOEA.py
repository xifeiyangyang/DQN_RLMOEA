"""
RL-MOEA2 参数敏感性分析 — SCI 论文实验章节
==============================================
对 6 个关键参数进行敏感性分析，每个参数取 5 个水平，
每个水平重复运行 3 次，以 HV（超体积）为主要评价指标，
生成带误差棒的出版级图表和 LaTeX 表格。
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
import warnings
import os
import time
from itertools import product
import json

import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings('ignore')

# ====================== 1. Publication Style (Times New Roman) ======================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 8
rcParams['axes.unicode_minus'] = False
rcParams['axes.linewidth'] = 0.5
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5
rcParams['xtick.major.size'] = 3
rcParams['ytick.major.size'] = 3
rcParams['xtick.labelsize'] = 7
rcParams['ytick.labelsize'] = 7
rcParams['lines.linewidth'] = 1.0
rcParams['lines.markersize'] = 3
rcParams['legend.fontsize'] = 7
rcParams['legend.frameon'] = False
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 600
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.02

COLD_PALETTE = ['#2B7BBA', '#3DA882', '#8B5EC7', '#00A19C', '#5C8ADB']

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# ====================== 2. 问题类（从原代码导入） ======================
from RLMOEA2_shanghai_MOSMCLP import MaxCoverLocationProblem, load_shanghai_data


# ====================== 3. 参数化 RL-MOEA2 ======================
class DQN_Sens(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN_Sens, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class RLMOEA2_Sensitivity:
    """可参数化的 RL-MOEA2，所有关键超参数均可外部设置"""

    def __init__(self, problem,
                 pop_size=50, max_iter=200,
                 mutation_prob=0.1,
                 epsilon=0.9, epsilon_decay=0.995, epsilon_min=0.1,
                 learning_rate=0.001, gamma=0.9,
                 batch_size=16, replay_size=500, target_update_freq=10):

        self.problem = problem
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.mutation_prob = mutation_prob

        self.state_dim = 4
        self.action_dim = 3
        self.dqn = DQN_Sens(self.state_dim, self.action_dim)
        self.target_dqn = DQN_Sens(self.state_dim, self.action_dim)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.replay_buffer = []
        self.max_replay_size = replay_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        self.state_before_action = None

        total_demand = np.sum(self.problem.demand_weights)
        total_cost = np.sum(self.problem.build_costs)
        self.hv_ref_point = np.array([total_demand * 1.1, total_cost * 1.1])

        self.pop = self._init_population()
        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.pop])
        self.current_iter = 0
        self.last_action = 0
        self.last_reward = 0.0
        self.max_fitness = np.max(self.fitness, axis=0) if len(self.fitness) > 0 else np.array([1.0, 1.0])
        self.min_fitness = np.min(self.fitness, axis=0) if len(self.fitness) > 0 else np.array([0.0, 0.0])

    def _init_population(self):
        pop = []
        greedy_sol = self._greedy_max_cover()
        pop.append(greedy_sol)
        while len(pop) < self.pop_size:
            sol = np.random.randint(0, 2, self.problem.n_candidates)
            n_selected = np.sum(sol)
            if 1 <= n_selected <= self.problem.n_candidates // 3:
                pop.append(sol)
        return np.array(pop)

    def _greedy_max_cover(self):
        sol = np.zeros(self.problem.n_candidates, dtype=int)
        uncovered = np.ones(self.problem.n_demands, dtype=bool)
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
        if len(self.fitness) == 0:
            return torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        fitness_range = self.max_fitness - self.min_fitness
        fitness_range = np.where(fitness_range == 0, 1, fitness_range)
        fitness_norm = (self.fitness - self.min_fitness) / fitness_range
        avg_fitness = np.mean(fitness_norm, axis=0)
        diversity = np.std(self.fitness, axis=0).mean()
        diversity_norm = diversity / (np.max(self.fitness) - np.min(self.fitness) + 1e-6)
        iter_progress = self.current_iter / self.max_iter
        return torch.tensor([avg_fitness[0], avg_fitness[1], diversity_norm, iter_progress],
                            dtype=torch.float32)

    def _select_operator(self):
        state = self._get_state()
        self.state_before_action = state
        if random.random() < self.epsilon:
            action = random.choice(range(self.action_dim))
        else:
            q_values = self.dqn(state)
            action = torch.argmax(q_values).item()
        self.last_action = action
        return action

    def _crossover_single_point(self, parent1, parent2):
        cross_point = random.randint(1, self.problem.n_candidates - 1)
        child1 = np.hstack((parent1[:cross_point], parent2[cross_point:]))
        child2 = np.hstack((parent2[:cross_point], parent1[cross_point:]))
        return child1, child2

    def _crossover_two_point(self, parent1, parent2):
        point1 = random.randint(1, self.problem.n_candidates - 2)
        point2 = random.randint(point1 + 1, self.problem.n_candidates - 1)
        child1 = np.hstack((parent1[:point1], parent2[point1:point2], parent1[point2:]))
        child2 = np.hstack((parent2[:point1], parent1[point1:point2], parent2[point2:]))
        return child1, child2

    def _crossover_uniform(self, parent1, parent2):
        mask = np.random.randint(0, 2, self.problem.n_candidates)
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    def _bit_flip_mutation(self, individual):
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_prob:
                mutated[i] = 1 - mutated[i]
        n_selected = np.sum(mutated)
        if n_selected == 0:
            mutated[random.randint(0, self.problem.n_candidates - 1)] = 1
        elif n_selected > self.problem.n_candidates // 3:
            selected_idx = np.where(mutated == 1)[0]
            delete_idx = random.sample(list(selected_idx), n_selected - (self.problem.n_candidates // 3))
            mutated[delete_idx] = 0
        return mutated

    def _update_dqn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        q_values = self.dqn(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_dqn(next_states)
            max_next_q = next_q_values.max(1)[0]
            target = rewards + self.gamma * max_next_q
        loss = self.criterion(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

    def run(self):
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

        for iter in range(self.max_iter):
            self.current_iter = iter
            if len(self.fitness) == 0:
                break
            self.max_fitness = np.maximum(self.max_fitness, np.max(self.fitness, axis=0))
            self.min_fitness = np.minimum(self.min_fitness, np.min(self.fitness, axis=0))
            old_fronts = _fast_nondominated_sort(self.fitness)
            old_hv = self._compute_hypervolume(self.fitness[old_fronts[0]]) \
                if (len(old_fronts) > 0 and len(old_fronts[0]) > 0) else 0.0
            action = self._select_operator()
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
                    parents.append(self.pop[idx1] if crowding[idx1] > crowding[idx2] else self.pop[idx2])
            parents = np.array(parents)
            offspring = []
            for i in range(0, self.pop_size, 2):
                if i + 1 >= self.pop_size:
                    break
                p1, p2 = parents[i], parents[i + 1]
                if action == 0:
                    c1, c2 = self._crossover_single_point(p1, p2)
                elif action == 1:
                    c1, c2 = self._crossover_two_point(p1, p2)
                else:
                    c1, c2 = self._crossover_uniform(p1, p2)
                offspring.append(self._bit_flip_mutation(c1))
                offspring.append(self._bit_flip_mutation(c2))
            offspring = np.array(offspring[:self.pop_size])
            offspring_fitness = np.array([self.problem.evaluate(ind) for ind in offspring])
            combined_pop = np.vstack((self.pop, offspring))
            combined_fitness = np.vstack((self.fitness, offspring_fitness))
            fronts = _fast_nondominated_sort(combined_fitness)
            crowding = np.zeros(len(combined_fitness))
            for front in fronts:
                crowding[front] = _calculate_crowding_distance(combined_fitness, front)
            new_pop, new_fitness = [], []
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
            new_fronts = _fast_nondominated_sort(self.fitness)
            new_hv = self._compute_hypervolume(self.fitness[new_fronts[0]]) \
                if (len(new_fronts) > 0 and len(new_fronts[0]) > 0) else 0.0
            reward = (new_hv - old_hv) / (old_hv + 1e-6)
            state_after = self._get_state()
            if self.state_before_action is not None:
                self.replay_buffer.append((self.state_before_action, self.last_action, reward, state_after))
                if len(self.replay_buffer) > self.max_replay_size:
                    self.replay_buffer.pop(0)
            self._update_dqn()

        if len(self.fitness) == 0:
            return np.array([])
        fronts = _fast_nondominated_sort(self.fitness)
        return self.fitness[fronts[0]] if (len(fronts) > 0 and len(fronts[0]) > 0) else np.array([])


# ====================== 4. 实验运行框架 ======================
def single_run(problem, run_seed, **kwargs):
    """单次运行 RL-MOEA2，返回 (HV, n_solutions, runtime_sec)"""
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)

    t0 = time.time()
    algo = RLMOEA2_Sensitivity(problem, **kwargs)
    pf = algo.run()
    elapsed = time.time() - t0

    hv = algo._compute_hypervolume(pf) if len(pf) > 0 else 0.0
    n_sol = len(pf)
    return hv, n_sol, elapsed


def run_sensitivity(problem, param_name, param_values, base_kwargs, n_repeats=3):
    """
    对单个参数进行敏感性分析
    :param problem: 问题实例
    :param param_name: 参数名称
    :param param_values: 参数取值列表
    :param base_kwargs: 基准参数配置
    :param n_repeats: 每个水平的重复次数
    :return: dict {value: [(hv, nsol, time), ...]}
    """
    results = {}
    total = len(param_values) * n_repeats
    count = 0

    for val in param_values:
        runs = []
        for rep in range(n_repeats):
            count += 1
            kwargs = base_kwargs.copy()
            kwargs[param_name] = val
            run_seed = hash(f"{param_name}_{val}_{rep}_sensitivity") % (2**31)

            print(f"  [{count}/{total}] {param_name}={val}, rep {rep+1}/{n_repeats}", end=" ... ", flush=True)
            hv, nsol, elapsed = single_run(problem, run_seed, **kwargs)
            runs.append((hv, nsol, elapsed))
            print(f"HV={hv:.4f}, nsol={nsol}, time={elapsed:.1f}s", flush=True)

        results[val] = runs

    return results


# ====================== 5. 敏感性参数定义 ======================
SENSITIVITY_CONFIG = {
    "pop_size": {
        "name_cn": "种群规模 $P$",
        "name_en": "Population size",
        "values": [20, 30, 50, 80, 100],
        "default": 50,
    },
    "max_iter": {
        "name_cn": "最大迭代次数 $G$",
        "name_en": "Max iterations",
        "values": [50, 100, 150, 200, 300],
        "default": 200,
    },
    "mutation_prob": {
        "name_cn": "变异概率 $p_m$",
        "name_en": "Mutation probability",
        "values": [0.05, 0.10, 0.15, 0.20, 0.25],
        "default": 0.10,
    },
    "epsilon": {
        "name_cn": "初始探索率 $\\varepsilon_0$",
        "name_en": "Initial exploration rate",
        "values": [0.70, 0.80, 0.85, 0.90, 0.95],
        "default": 0.90,
    },
    "learning_rate": {
        "name_cn": "学习率 $\\eta$",
        "name_en": "Learning rate",
        "values": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "default": 1e-3,
    },
    "gamma": {
        "name_cn": "折扣因子 $\\gamma$",
        "name_en": "Discount factor",
        "values": [0.80, 0.85, 0.90, 0.95, 0.99],
        "default": 0.90,
    },
}

# 基准参数
BASE_KWARGS = {
    "pop_size": 50,
    "max_iter": 200,
    "mutation_prob": 0.10,
    "epsilon": 0.90,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.10,
    "learning_rate": 0.001,
    "gamma": 0.90,
    "batch_size": 16,
    "replay_size": 500,
    "target_update_freq": 10,
}

N_REPEATS = 3

# ====================== 6. 缓存与加载 ======================
RESULTS_CACHE_FILE = "sensitivity_results.json"

def cache_key(param_name, value, rep):
    return f"{param_name}_{value}_{rep}"

def save_results(all_results, filepath=RESULTS_CACHE_FILE):
    """保存结果为 JSON（可序列化格式）"""
    serializable = {}
    for param_name, param_data in all_results.items():
        serializable[param_name] = {}
        for val, runs in param_data.items():
            serializable[param_name][str(val)] = [
                {"hv": float(hv), "nsol": int(nsol), "time": float(t)}
                for hv, nsol, t in runs
            ]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n结果已缓存至: {filepath}")

def load_results(filepath=RESULTS_CACHE_FILE):
    """加载缓存结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    results = {}
    for param_name, param_data in raw.items():
        results[param_name] = {}
        for val_str, runs in param_data.items():
            val = float(val_str) if '.' in val_str else int(val_str)
            results[param_name][val] = [
                (r["hv"], r["nsol"], r["time"]) for r in runs
            ]
    return results

# ====================== 7. 主函数 ======================
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    problem_file = os.path.join(current_dir, "shanghai_problem.npz")

    # 加载问题
    if os.path.exists(problem_file):
        print("=" * 60)
        print("加载已保存的 problem 对象...")
        problem = MaxCoverLocationProblem.load(problem_file)
        print("=" * 60)
    else:
        print("未找到 problem.npz，从原始数据生成...")
        candidate_sites, demand_points, coverage_radius, build_costs, demand_weights = load_shanghai_data()
        problem = MaxCoverLocationProblem(
            candidate_sites, demand_points, coverage_radius, build_costs, demand_weights
        )
        problem.save(problem_file)

    # 加载或运行实验
    if os.path.exists(RESULTS_CACHE_FILE):
        print(f"\n发现缓存结果文件 {RESULTS_CACHE_FILE}，直接加载...")
        all_results = load_results()
    else:
        print("\n" + "=" * 60)
        print("开始参数敏感性分析实验")
        print(f"参数数量: {len(SENSITIVITY_CONFIG)}")
        print(f"每个参数的取值水平: 5")
        print(f"每个水平重复次数: {N_REPEATS}")
        print(f"总运行次数: {len(SENSITIVITY_CONFIG) * 5 * N_REPEATS}")
        print("=" * 60)

        all_results = {}
        for param_name, config in SENSITIVITY_CONFIG.items():
            print(f"\n{'=' * 60}")
            print(f"Parameter: {config['name_en']} ({param_name})")
            print(f"Values: {config['values']}")
            print(f"{'=' * 60}")
            import sys; sys.stdout.flush()

            results = run_sensitivity(
                problem, param_name, config["values"],
                BASE_KWARGS, n_repeats=N_REPEATS
            )
            all_results[param_name] = results
            # Incremental save after each parameter
            save_results(all_results)
            print(f"  [Saved] {param_name} results cached")
            sys.stdout.flush()

    # ==================== 8. 可视化 ====================
    print("\n" + "=" * 60)
    print("生成敏感性分析图表...")
    print("=" * 60)

    # 汇总统计
    summary_data = {}
    for param_name, config in SENSITIVITY_CONFIG.items():
        param_results = all_results[param_name]
        means, stds, means_nsol, means_time = [], [], [], []
        for val in config["values"]:
            runs = param_results[val]
            hvs = [r[0] for r in runs]
            nsols = [r[1] for r in runs]
            times = [r[2] for r in runs]
            means.append(np.mean(hvs))
            stds.append(np.std(hvs))
            means_nsol.append(np.mean(nsols))
            means_time.append(np.mean(times))
        summary_data[param_name] = {
            "values": config["values"],
            "hv_mean": means,
            "hv_std": stds,
            "nsol_mean": means_nsol,
            "time_mean": means_time,
        }

    # 创建 2x3 子图 (Times New Roman, 冷色系)
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.5))
    axes = axes.flatten()

    param_keys = list(SENSITIVITY_CONFIG.keys())
    panel_labels = ['a', 'b', 'c', 'd', 'e', 'f']

    for idx, param_name in enumerate(param_keys[:5]):  # panels (a)-(e) only
        ax = axes[idx]
        config = SENSITIVITY_CONFIG[param_name]
        data = summary_data[param_name]
        x_vals = data["values"]
        y_mean = data["hv_mean"]
        y_std = data["hv_std"]

        # 主色
        color = COLD_PALETTE[idx % len(COLD_PALETTE)]

        # 折线 + 误差带
        ax.plot(x_vals, y_mean, 'o-', color=color, markersize=4,
                linewidth=1.0, markerfacecolor='white', markeredgewidth=0.8, markeredgecolor=color)
        ax.fill_between(x_vals,
                        [m - s for m, s in zip(y_mean, y_std)],
                        [m + s for m, s in zip(y_mean, y_std)],
                        color=color, alpha=0.12)

        # 标注基准值
        default_val = config["default"]
        if default_val in x_vals:
            default_idx = x_vals.index(default_val)
            ax.axvline(x=default_val, color='#888888', linestyle='--',
                       linewidth=0.5, alpha=0.6)

        # 面板标签
        ax.set_title(f"{panel_labels[idx]}  {config['name_en']}", fontsize=8,
                     loc='left', fontweight='normal')
        ax.set_xlabel(config["name_cn"], fontsize=7)
        ax.set_ylabel("HV", fontsize=7)

    # Panel (f): Dual metric — HV Range (bars, left) + mean CV (diamonds, right)
    ax = axes[5]
    ax2 = ax.twinx()

    hv_ranges = []
    mean_cvs = []
    for param_name in param_keys:
        data = summary_data[param_name]
        hv_ranges.append(max(data["hv_mean"]) - min(data["hv_mean"]))
        cvs_per_level = [s / (m + 1e-8) for m, s in zip(data["hv_mean"], data["hv_std"])]
        mean_cvs.append(np.mean(cvs_per_level))

    bars = ax.bar(range(6), hv_ranges, color=COLD_PALETTE, alpha=0.70, edgecolor='white', linewidth=0.5,
                  label='HV Range', zorder=2)
    max_idx = np.argmax(hv_ranges)
    bars[max_idx].set_edgecolor('black')
    bars[max_idx].set_linewidth(1.2)
    for i, (bar, val) in enumerate(zip(bars, hv_ranges)):
        y_off = 0.3 if i == max_idx else 0.15
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_off,
                f'{val:.2f}', ha='center', fontsize=6, fontweight='bold' if i == max_idx else 'normal',
                color='#333333')

    ax2.scatter(range(6), mean_cvs, marker='D', s=50, c='#E31A1C', edgecolor='black', linewidth=0.4,
                zorder=4)
    for i, cv in enumerate(mean_cvs):
        ax2.annotate(f'{cv:.3f}', (i, cv), textcoords='offset points',
                     xytext=(0, -14), fontsize=5.5, ha='center', color='#E31A1C')

    ax.set_xticks(range(6))
    ax.set_xticklabels([SENSITIVITY_CONFIG[p]["name_en"] for p in param_keys], fontsize=5.5, rotation=25)
    ax.set_ylabel('HV Range [x10^9]', fontsize=7)
    ax2.set_ylabel('Mean CV', fontsize=7, color='#E31A1C')
    ax2.set_ylim(0, 0.025)
    ax2.tick_params(axis='y', labelcolor='#E31A1C', labelsize=6)
    ax.set_title('(f) Sensitivity: Impact (bar) vs Stability (diamond)', fontsize=8, loc='left')

    # 全局调整
    plt.tight_layout(pad=1.2, h_pad=1.5, w_pad=1.5)
    fig.suptitle("Parameter Sensitivity Analysis — RL-MOEA2 on Shanghai MOSMCLP",
                 fontsize=9, fontweight='normal', y=1.02)

    os.makedirs("figures", exist_ok=True)
    save_name = "figures/Fig_Sensitivity_Analysis.png"
    plt.savefig(save_name, dpi=600)
    plt.close()
    print(f"Figure saved: {save_name}")

    # ==================== 9. 统计表格 ====================
    print("\n" + "=" * 60)
    print("生成统计表格...")
    print("=" * 60)

    table_rows = []
    for param_name in param_keys:
        config = SENSITIVITY_CONFIG[param_name]
        data = summary_data[param_name]
        for v, m, s, ns, t in zip(data["values"], data["hv_mean"],
                                   data["hv_std"], data["nsol_mean"],
                                   data["time_mean"]):
            is_default = "Y" if v == config["default"] else ""
            table_rows.append({
                "Parameter": config["name_en"],
                "Value": f"{v:.4g}",
                "HV (mean)": f"{m:.4f}",
                "HV (std)": f"{s:.4f}",
                "CV": f"{s / (m + 1e-8):.4f}",
                "# Solutions": f"{ns:.1f}",
                "Time (s)": f"{t:.1f}",
                "Default": is_default,
            })

    df_table = pd.DataFrame(table_rows)
    csv_name = "figures/Sensitivity_Analysis_Table.csv"
    df_table.to_csv(csv_name, index=False, encoding='utf-8-sig')
    print(f"Table saved: {csv_name}")
    print("\n" + df_table.to_string(index=False))

    print("\n" + "=" * 60)
    print("参数敏感性分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
