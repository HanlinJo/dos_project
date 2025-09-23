import random
import numpy as np
import fastrand, math
from .mod_utils import is_lnorm_key
import torch
import torch.nn as nn


class SSNE:
    def __init__(self, args):
        self.current_gen = 0
        self.current_gen = 0
        self.args = args

        # 使用 .get("key", default_value) 的方式从字典中安全地获取值
        # 如果字典中没有对应的 "key"，则会使用提供的 default_value
        self.prob_reset_and_sup = args.get("prob_reset_and_sup", 0.05)
        self.frac = args.get("frac", 1.0)
    
        # pop_size 应该是必需参数，但使用 .get 依然是好习惯
        self.population_size = args.get("pop_size")
        if self.population_size is None:
            raise ValueError("参数 'pop_size' 未在配置中定义！")  # 如果是必需参数，最好抛出错误
        elite_fraction = args.get("elite_fraction", 0.2)
        self.num_elitists = int(elite_fraction * self.population_size)
        if self.num_elitists < 1: self.num_elitists = 1
 
        self.rl_policy = None
        self.selection_stats = {'elite': 0, 'selected': 0, 'discarded':0, 'total':0.0000001}

    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[fastrand.pcg32bounded(len(offsprings))])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def regularize_weight(self, weight, mag):
        if weight > mag: weight = mag
        if weight < -mag: weight = -mag
        return weight

    # def crossover_inplace(self, gene1, gene2, agent_index:int):
    #     # Evaluate the parents


    #     b_1 = None 
    #     b_2 = None 
    #     for param1, param2 in zip(gene1.agent_W[agent_index].parameters(), gene2.agent_W[agent_index].parameters()):
    #         # References to the variable tensors
    #         W1 = param1.data
    #         W2 = param2.data 
    #         if len(W1.shape) == 1:
    #             b_1 = W1
    #             b_2 = W2

    #     for param1, param2 in zip(gene1.agent_W[agent_index].parameters(), gene2.agent_W[agent_index].parameters()):
    #         # References to the variable tensors
    #         W1 = param1.data
    #         W2 = param2.data

    #         if len(W1.shape) == 2: #Weights no bias
    #             num_variables = W1.shape[0]
    #             # Crossover opertation [Indexed by row]
    #             num_cross_overs = fastrand.pcg32bounded(num_variables * 2)  # Lower bounded on full swaps
    #             for i in range(num_cross_overs):
    #                 receiver_choice = random.random()  # Choose which gene to receive the perturbation
    #                 if receiver_choice < 0.5:
    #                     ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
    #                     W1[ind_cr,:] = W2[ind_cr,:]
    #                     b_1[ind_cr] = b_2[ind_cr]
    #                 else:
    #                     ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
    #                     W2[ind_cr,:] = W1[ind_cr,:]
    #                     b_2[ind_cr] = b_1[ind_cr]

        # Evaluate the children
    def crossover_inplace(self, gene1, gene2, agent_index:int):
        # 将网络参数转换为迭代器，以便成对处理（权重+偏置）
        params1 = iter(gene1.agent_W[agent_index].parameters())
        params2 = iter(gene2.agent_W[agent_index].parameters())

        # 循环直到所有参数都被处理完毕
        while True:
            try:
                # 获取当前层的权重矩阵
                w1_param = next(params1)
                w2_param = next(params2)

                # 获取当前层对应的偏置向量
                b1_param = next(params1)
                b2_param = next(params2)

                # 引用参数的数据张量
                W1, b_1 = w1_param.data, b1_param.data
                W2, b_2 = w2_param.data, b2_param.data

                # 确保我们正在处理一个权重矩阵和一个偏置向量
                # 这是为了跳过像 LayerNorm 这样可能不符合 (权重, 偏置) 成对结构的层
                if len(W1.shape) != 2 or len(b_1.shape) != 1:
                    continue

                # 确保权重和偏置的维度匹配
                if W1.shape[0] != b_1.shape[0]:
                    continue

                num_variables = W1.shape[0]
                # 交叉操作 [按行索引]
                num_cross_overs = fastrand.pcg32bounded(num_variables * 2)  # 与原始逻辑保持一致
                for i in range(num_cross_overs):
                    receiver_choice = random.random()  # 选择哪个基因接收扰动
                    ind_cr = fastrand.pcg32bounded(num_variables)  # 在当前层的维度内生成安全索引

                    if receiver_choice < 0.5:
                        # 将 gene2 的神经元（权重行 + 偏置）复制到 gene1
                        W1[ind_cr, :] = W2[ind_cr, :].clone() # 使用 .clone() 避免张量共享问题
                        b_1[ind_cr] = b_2[ind_cr].clone()
                    else:
                        # 将 gene1 的神经元复制到 gene2
                        W2[ind_cr, :] = W1[ind_cr, :].clone()
                        b_2[ind_cr] = b_1[ind_cr].clone()

            except StopIteration:
                # 当 next() 无法获取更多参数时，表示所有层都已处理完毕
                break

    def mutate_inplace(self, gene, agent_index, agent_level= False):
        trials = 5

        mut_strength = 0.1
        num_mutation_frac = 0.1
        super_mut_strength = 10
        super_mut_prob = self.prob_reset_and_sup
        reset_prob = super_mut_prob + self.prob_reset_and_sup

        num_params = len(list(gene.agent_W[agent_index].parameters()))
        ssne_probabilities = np.random.uniform(0, 1, num_params) * 2
        model_params = gene.agent_W[agent_index].state_dict()

        for i, key in enumerate(model_params): #Mutate each param

            if is_lnorm_key(key):
                continue

            # References to the variable keys
            W = model_params[key]
            if len(W.shape) == 2: #Weights, no bias
                if agent_level:
                    ssne_prob = 1.0 #ssne_probabilities[i]
                    action_prob = 1.0
                else :
                    ssne_prob = 1.0
                    action_prob = ssne_probabilities[i]

                if random.random() < ssne_prob:
                    num_variables = W.shape[0]
                    # Crossover opertation [Indexed by row]
                    for index in range(num_variables):
                        random_num_num = random.random()
                        if random_num_num <= action_prob :
                            #print(W)
                            index_list = random.sample(range(W.shape[1]), int(W.shape[1] * self.frac))
                            random_num = random.random()
                            if random_num < super_mut_prob:  # Super Mutation probability
                                for ind in index_list:
                                    W[index, ind] += random.gauss(0, super_mut_strength * W[index, ind])
                            elif random_num < reset_prob:  # Reset probability
                                for ind in index_list:
                                    W[index, ind] = random.gauss(0, 1)
                            else:  # mutation even normal
                                for ind in index_list:
                                    W[index, ind] += random.gauss(0, mut_strength * W[index, ind])

                            # 【关键修改】使用 torch.clamp 替代 np.clip
                            # Regularization hard limit
                            W[index, :] = torch.clamp(W[index, :], min=-1000000, max=1000000)


    def clone(self, master, replacee,agent_index:int):  # Replace the replacee individual with master


        for target_param, source_param in zip(replacee.agent_W[agent_index].parameters(), master.agent_W[agent_index].parameters()):
            target_param.data.copy_(source_param.data)
        #replacee.buffer.reset()
        #replacee.buffer.add_content_of(master.buffer)

    def reset_genome(self, gene):
        for param in (gene.agent_W.parameters()):
            param.data.copy_(param.data)


    def epoch(self, pop, fitness_evals, agent_index, agent_level=False):
        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        index_rank = np.argsort(fitness_evals)[::-1]
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)

        # Figure out unselected candidates
        unselects = []; new_elitists = []
        for i in range(self.population_size):
            if i not in offsprings and i not in elitist_index:
                unselects.append(i)
        random.shuffle(unselects)

        # COMPUTE RL_SELECTION RATE
        if self.rl_policy is not None: # RL Transfer happened
            self.selection_stats['total'] += 1.0

            if self.rl_policy in elitist_index: self.selection_stats['elite'] += 1.0
            elif self.rl_policy in offsprings: self.selection_stats['selected'] += 1.0
            elif self.rl_policy in unselects: self.selection_stats['discarded'] += 1.0
            self.rl_policy = None

        # Elitism step, assigning elite candidates to some unselects
        for i in elitist_index:
            try: replacee = unselects.pop(0)
            except: replacee = offsprings.pop(0)
            new_elitists.append(replacee)
            self.clone(master=pop[i], replacee=pop[replacee], agent_index=agent_index)

        # Crossover between elite and offsprings for the unselected genes with 100 percent probability

        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            unselects.append(unselects[fastrand.pcg32bounded(len(unselects))])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(new_elitists)
            off_j = random.choice(offsprings)
            self.clone(master=pop[off_i], replacee=pop[i],agent_index= agent_index)
            self.clone(master=pop[off_j], replacee=pop[j],agent_index = agent_index)

            if agent_level:
                if random.random() < 0.5:
                    self.clone(master=pop[i], replacee=pop[j], agent_index=agent_index)
                else :
                    self.clone(master=pop[j], replacee=pop[i], agent_index=agent_index)
            else :
                self.crossover_inplace(pop[i], pop[j],agent_index)

        # # Crossover for selected offsprings
        # for i in offsprings:
        #     if random.random() < self.args.crossover_prob:
        #         others = offsprings.copy()
        #         others.remove(i)
        #         off_j = random.choice(others)
        #         self.clone(self.distilation_crossover(pop[i], pop[off_j]), pop[i],agent_index)

        # Mutate all genes in the population except the new elitists
        for i in range(self.population_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.args["mutation_prob"]:
                    self.mutate_inplace(pop[i],agent_index=agent_index,agent_level=agent_level)

        return new_elitists[0]


def unsqueeze(array, axis=1):
    if axis == 0: return np.reshape(array, (1, len(array)))
    elif axis == 1: return np.reshape(array, (len(array), 1))



