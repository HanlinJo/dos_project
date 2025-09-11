import numpy as np
import torch

class EAManager:
    def __init__(self, population_size, elite_count, crossover_prob, mutation_prob, mutation_strength, device):
        self.population_size = population_size
        self.elite_count = elite_count
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_strength = mutation_strength
        self.device = device
        self.population = []
        self.fitness = np.zeros(population_size)

    def initialize_population(self, initial_actor):
        """
        使用初始化的 actor 来创建种群
        """
        for _ in range(self.population_size):
            # 创建 actor 的深拷贝以确保每个个体都是独立的
            actor_copy = type(initial_actor)(initial_actor.args, initial_actor.obs_space, initial_actor.act_space, self.device)
            actor_copy.load_state_dict(initial_actor.state_dict())
            self.population.append(actor_copy)

    def evaluate_population(self, env, runner):
        """
        评估种群中每个个体的适应度
        """
        for i, actor in enumerate(self.population):
            # 使用 runner 来评估 actor 的表现
            # 这里我们简化为直接获取 runner 中的评估回报
            # 你可以根据实际情况进行修改
            self.fitness[i] = runner.evaluate(actor)

    def evolve(self):
        """
        进化种群，包括选择、交叉和变异
        """
        # 1. 选择
        sorted_indices = np.argsort(self.fitness)[::-1]
        new_population = [self.population[i] for i in sorted_indices[:self.elite_count]]

        # 2. 交叉和变异
        while len(new_population) < self.population_size:
            parent1_idx, parent2_idx = np.random.choice(sorted_indices, 2, replace=False)
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]

            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population

    def crossover(self, parent1, parent2):
        """
        对两个父代进行交叉操作
        """
        child = type(parent1)(parent1.args, parent1.obs_space, parent1.act_space, self.device)
        child.load_state_dict(parent1.state_dict())

        if np.random.rand() < self.crossover_prob:
            for child_param, parent2_param in zip(child.parameters(), parent2.parameters()):
                # 以一定的概率选择 parent2 的参数
                mask = torch.rand(child_param.data.size()) < 0.5
                mask = mask.to(self.device)
                child_param.data[mask] = parent2_param.data[mask]

        return child

    def mutate(self, actor):
        """
        对 actor 进行变异操作
        """
        if np.random.rand() < self.mutation_prob:
            for param in actor.parameters():
                noise = torch.randn(param.data.size()) * self.mutation_strength
                noise = noise.to(self.device)
                param.data += noise

        return actor

    def get_elite_agent(self):
        """
        获取种群中适应度最高的个体
        """
        elite_index = np.argmax(self.fitness)
        return self.population[elite_index]