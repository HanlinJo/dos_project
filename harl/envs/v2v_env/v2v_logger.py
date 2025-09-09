# # # harl/envs/v2v/v2v_logger.py

# # import time
# # import numpy as np
# # from harl.common.base_logger import BaseLogger

# # class V2VLogger(BaseLogger):
# #     def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
# #         """
# #         初始化V2V环境的日志记录器。
# #         - 继承自BaseLogger，处理基础的日志功能。
# #         - 增加了对V2V环境特定指标的跟踪。
# #         """
# #         super(V2VLogger, self).__init__(
# #             args, algo_args, env_args, num_agents, writter, run_dir
# #         )
# #         # 初始化用于存储每个周期内各指标的列表
# #         self.episode_rewards = []
# #         self.episode_collisions = []
# #         self.episode_prr = []
# #         self.episode_attack_success_rate = []

# #     def get_task_name(self):
# #         """
# #         从环境参数中获取任务的名称，用于日志和结果文件夹的命名。
# #         """
# #         # 您可以自定义任务名称的格式
# #         return f'{self.env_args["num_vehicles"]}v{self.env_args["num_attackers"]}'

# #     def init(self,episodes):
# #         """
# #         在训练开始时被调用，用于初始化或重置统计数据。
# #         """
# #         self.start = time.time()
# #         self.episode_rewards = []
# #         self.episode_collisions = []
# #         self.episode_prr = []
# #         self.epidoes = episodes
# #         self.episode_attack_success_rate = []

# #     def per_step(self, data):
# #         """
# #         每个环境步骤后被调用。
# #         主要目的是为了在episode结束时能获取到最新的info信息。
# #         """
# #         # 在on-policy runner中，infos会在episode_done之前被设置为最新的infos
# #         super().per_step(data)

# #     def episode_done(self):
# #         """
# #         每个episode结束后被调用。
# #         在这里，我们从infos中提取并累积整个episode的统计数据。
# #         """
# #         super().episode_done()
# #         # self.infos 是一个列表，每个线程对应一个字典
# #         # 我们在这里假设主要关注第一个线程的最终信息
# #         final_info = self.infos[0]
        
# #         # 累积各项指标
# #         # final_info中应该包含环境在done=True时返回的统计信息
# #         if "step_reward" in final_info: # 奖励是累积的，这里只记录最后一步的（或者您可以在runner中累积）
# #             # 注意: on_policy_base_runner.py 会自动累积奖励到 self.episode_rewards 中
# #             # 所以我们主要关注环境特有的指标
# #             pass

# #         if "total_collisions" in final_info:
# #             self.episode_collisions.append(final_info["total_collisions"])
        
# #         if "prr" in final_info:
# #             self.episode_prr.append(final_info["prr"])

# #         if "attack_success_rate" in final_info:
# #             self.episode_attack_success_rate.append(final_info["attack_success_rate"])


# #     def log_train(self, train_infos, total_num_steps):
# #         """
# #         定期被调用以记录训练信息。
# #         计算周期内的平均指标，并打印到控制台和写入TensorBoard。
# #         """
# #         super().log_train(train_infos, total_num_steps)
        
# #         # 计算平均值，如果列表为空则默认为0
# #         avg_rewards = np.mean(self.episode_rewards) if len(self.episode_rewards) > 0 else 0.0
# #         avg_collisions = np.mean(self.episode_collisions) if len(self.episode_collisions) > 0 else 0.0
# #         avg_prr = np.mean(self.episode_prr) if len(self.episode_prr) > 0 else 0.0
# #         avg_attack_success = np.mean(self.episode_attack_success_rate) if len(self.episode_attack_success_rate) > 0 else 0.0

# #         # 打印到控制台
# #         print(
# #             f"episodes: {self.episodes} | "
# #             f"Steps: {total_num_steps}/{self.algo_args['train']['num_env_steps']} | "
# #             f"FPS: {int(total_num_steps / (time.time() - self.start))} | "
# #             f"Reward: {avg_rewards:.3f} | "
# #             f"Collisions: {avg_collisions:.2f} | "
# #             f"PRR: {avg_prr:.3f} | "
# #             f"Attack Success: {avg_attack_success:.3f}"
# #         )

# #         # 准备要写入TensorBoard的数据
# #         env_infos = {
# #             "average_episode_rewards": avg_rewards,
# #             "average_collisions": avg_collisions,
# #             "average_prr": avg_prr,
# #             "average_attack_success_rate": avg_attack_success
# #         }
# #         self.log_env(env_infos, total_num_steps)

# #         # 清空列表，为下一个记录周期做准备
# #         self.episode_rewards = []
# #         self.episode_collisions = []
# #         self.episode_prr = []
# #         self.episode_attack_success_rate = []

# #     def eval_init(self):
# #         """初始化评估阶段的统计数据。"""
# #         super().eval_init()
# #         self.eval_episode_collisions = []
# #         self.eval_episode_prr = []
# #         self.eval_episode_attack_success_rate = []

# #     def eval_thread_done(self, tid):
# #         """每个评估线程结束后，累积其统计数据。"""
# #         super().eval_thread_done(tid)
# #         info = self.eval_infos[tid][0]
# #         if "total_collisions" in info:
# #             self.eval_episode_collisions.append(info["total_collisions"])
# #         if "prr" in info:
# #             self.eval_episode_prr.append(info["prr"])
# #         if "attack_success_rate" in info:
# #             self.eval_episode_attack_success_rate.append(info["attack_success_rate"])

# #     def eval_log(self, eval_episodes, total_num_steps):
# #         """在所有评估结束后，记录评估结果。"""
# #         # 从runner获取评估奖励
# #         self.eval_episode_rewards = np.concatenate(
# #             [rewards for rewards in self.eval_episode_rewards if rewards]
# #         )
        
# #         # 计算评估指标的平均值
# #         eval_avg_rew = np.mean(self.eval_episode_rewards)
# #         eval_avg_collisions = np.mean(self.eval_episode_collisions)
# #         eval_avg_prr = np.mean(self.eval_episode_prr)
# #         eval_avg_attack_success = np.mean(self.eval_episode_attack_success_rate)

# #         # 准备写入TensorBoard的数据
# #         eval_env_infos = {
# #             "eval_average_episode_rewards": eval_avg_rew,
# #             "eval_average_collisions": eval_avg_collisions,
# #             "eval_average_prr": eval_avg_prr,
# #             "eval_average_attack_success_rate": eval_avg_attack_success
# #         }
# #         self.log_env(eval_env_infos, total_num_steps)
        
# #         # 打印到控制台
# #         print(
# #             f"---------------------------------------------------\n"
# #             f"Evaluation after {total_num_steps} steps: \n"
# #             f"Avg Reward: {eval_avg_rew:.3f}, "
# #             f"Avg Collisions: {eval_avg_collisions:.2f}, "
# #             f"Avg PRR: {eval_avg_prr:.3f}, "
# #             f"Avg Attack Success: {eval_avg_attack_success:.3f}\n"
# #             f"---------------------------------------------------"
# #         )
# # harl/envs/v2v/v2v_logger.py

# import time
# import numpy as np
# from harl.common.base_logger import BaseLogger

# class V2VLogger(BaseLogger):
#     def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
#         """
#         初始化V2V环境的日志记录器。
#         - 继承自BaseLogger，处理基础的日志功能。
#         - 增加了对V2V环境特定指标的跟踪。
#         """
#         # FIX: 调用父类的 __init__ 是正确的，无需修改
#         super(V2VLogger, self).__init__(
#             args, algo_args, env_args, num_agents, writter, run_dir
#         )
#         # 初始化用于存储每个周期内各指标的列表
#         # 注意：父类 BaseLogger 会处理奖励的累积，这里主要关注自定义指标
#         self.episode_collisions = []
#         self.episode_prr = []
#         self.episode_attack_success_rate = []

#     def get_task_name(self):
#         """
#         从环境参数中获取任务的名称，用于日志和结果文件夹的命名。
#         """
#         return f'{self.env_args["num_vehicles"]}v{self.env_args["num_attackers"]}'

#     def init(self, episodes):
#         """
#         在训练开始时被调用，用于初始化或重置统计数据。
#         """
#         # FIX: 必须调用父类的 init 方法来初始化其属性，如 train_episode_rewards
#         super().init(episodes)
        
#         # FIX: 修正拼写错误 epidoes -> episodes
#         self.episodes = episodes
        
#         # 重置V2V特定指标的列表
#         self.episode_collisions = []
#         self.episode_prr = []
#         self.episode_attack_success_rate = []

#     def per_step(self, data):
#         """
#         每个环境步骤后被调用。
#         在这里处理V2V环境的特定指标。
#         """
#         # FIX: 调用父类的 per_step 来处理奖励累积等通用逻辑
#         super().per_step(data)
#         # print(data)
#         # 从 data 元组中解包 infos
#         infos = data[4]
#         # print(infos)
#         # 检查是否是episode的最后一步
#         dones = data[3]
#         # print(dones)
#         # exit(0)
#         dones_env = np.all(dones, axis=1)

#         for t in range(self.algo_args["train"]["n_rollout_threads"]):
#             if dones_env[t]:
#                 # 当一个episode结束时，从该线程的info中提取最终统计数据
#                 # BaseLogger只关心奖励，子类在这里处理特定指标
#                 final_info = infos[t]
#                 if "total_collisions" in final_info:
#                     self.episode_collisions.append(final_info["total_collisions"])
#                 if "prr" in final_info:
#                     self.episode_prr.append(final_info["prr"])
#                 if "attack_success_rate" in final_info:
#                     self.episode_attack_success_rate.append(final_info["attack_success_rate"])

#     # NOTE: episode_done 方法被移除，其逻辑已合并到 per_step 中，以更好地匹配 BaseLogger 的流程

#     # FIX: 方法名和签名必须与父类保持一致，才能正确覆盖
#     def episode_log(self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer):
#         """
#         定期被调用以记录训练信息。
#         计算周期内的平均指标，并打印到控制台和写入TensorBoard。
#         """
#         # FIX: 首先调用父类的 episode_log 来处理标准日志输出
#         super().episode_log(actor_train_infos, critic_train_info, actor_buffer, critic_buffer)
        
#         # 计算自定义指标的平均值，如果列表为空则默认为0
#         avg_collisions = np.mean(self.episode_collisions) if self.episode_collisions else 0.0
#         avg_prr = np.mean(self.episode_prr) if self.episode_prr else 0.0
#         avg_attack_success = np.mean(self.episode_attack_success_rate) if self.episode_attack_success_rate else 0.0

#         # 打印自定义指标到控制台
#         print(
#             f"Average total collisions: {avg_collisions:.2f} | "
#             f"Average PRR: {avg_prr:.3f} | "
#             f"Average Attack Success Rate: {avg_attack_success:.3f}"
#         )

#         # 准备要写入TensorBoard的自定义环境数据
#         env_infos = {
#             "average_collisions": avg_collisions,
#             "average_prr": avg_prr,
#             "average_attack_success_rate": avg_attack_success
#         }
#         self.log_env(env_infos)

#         # 清空列表，为下一个记录周期做准备
#         self.episode_collisions = []
#         self.episode_prr = []
#         self.episode_attack_success_rate = []

#     def eval_init(self):
#         """初始化评估阶段的统计数据。"""
#         # FIX: 调用父类方法以确保其属性被初始化
#         super().eval_init()
#         self.eval_episode_collisions = []
#         self.eval_episode_prr = []
#         self.eval_episode_attack_success_rate = []

#     def eval_thread_done(self, tid):
#         """每个评估线程结束后，累积其统计数据。"""
#         # FIX: 调用父类方法来累积奖励
#         super().eval_thread_done(tid)
        
#         # 假设eval_infos的结构是每个线程一个info列表，我们取第一个info
#         if self.eval_infos and len(self.eval_infos) > tid and self.eval_infos[tid]:
#             info = self.eval_infos[tid][0] # 通常在评估时，每个线程只跑一个episode
#             if "total_collisions" in info:
#                 self.eval_episode_collisions.append(info["total_collisions"])
#             if "prr" in info:
#                 self.eval_episode_prr.append(info["prr"])
#             if "attack_success_rate" in info:
#                 self.eval_episode_attack_success_rate.append(info["attack_success_rate"])

#     # FIX: 方法签名与父类保持一致
#     def eval_log(self, eval_episode):
#         """在所有评估结束后，记录评估结果。"""
#         # 计算评估指标的平均值
#         eval_avg_collisions = np.mean(self.eval_episode_collisions) if self.eval_episode_collisions else 0.0
#         eval_avg_prr = np.mean(self.eval_episode_prr) if self.eval_episode_prr else 0.0
#         eval_avg_attack_success = np.mean(self.eval_episode_attack_success_rate) if self.eval_episode_attack_success_rate else 0.0

#         # 准备写入TensorBoard的数据
#         # eval_env_infos = {
#         #     "eval_average_collisions": eval_avg_collisions,
#         #     "eval_average_prr": eval_avg_prr,
#         #     "eval_average_attack_success_rate": eval_avg_attack_success
#         # }
#         env_infos = {
#             "average_collisions": self.episode_collisions,
#             "average_prr": self.episode_prr,
#             "average_attack_success_rate": self.episode_attack_success_rate
#         }
#         # 使用父类方法将自定义评估指标写入TensorBoard
#         self.log_env(env_infos)
        
#         # 打印自定义评估结果到控制台
#         print(f"Evaluation: Average Collisions: {self.episode_collisions:.2f}")
#         print(f"Evaluation: Average PRR: {self.episode_prr:.3f}")
#         print(f"Evaluation: Average Attack Success Rate: {self.episode_attack_success_rate:.3f}")

#         # FIX: 最后调用父类的 eval_log 来处理奖励的日志记录和文件写入
#         super().eval_log(eval_episode)

# harl/envs/v2v/v2v_logger.py

import time
import numpy as np
from harl.common.base_logger import BaseLogger

class V2VLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        """
        初始化V2V环境的日志记录器，参照SMACLogger的模式。
        """
        super(V2VLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )

    def get_task_name(self):
        """
        从环境参数中获取任务的名称。
        """
        return f'{self.env_args["num_vehicles"]}v{self.env_args["num_attackers"]}'

    def init(self, episodes):
        """
        在训练开始时被调用，初始化所有统计数据。
        PATTERN: 参照smac_logger，为每个并行的rollout_thread初始化状态。
        """
        super().init(episodes) # 调用父类init来初始化奖励相关的统计
        self.episodes = episodes

        # 初始化用于累积在当前日志周期内完成的episode的统计数据
        self.episode_prr = []
        self.episode_attack_success_rate = []
        self.episode_collision_rate = []

        # 初始化用于跟踪每个并行环境中累积指标的“上一个”状态
        # 这对于计算增量指标至关重要
        num_threads = self.algo_args["train"]["n_rollout_threads"]
        self.last_total_collisions = np.zeros(num_threads, dtype=np.float32)
        self.last_message_failures = np.zeros(num_threads, dtype=np.float32)


    def per_step(self, data):
        """
        每个环境步骤后被调用。
        主要负责在episode结束时，从info中提取最终指标并累积。
        """
        super().per_step(data) # 父类方法会累积奖励
        
        infos = data[4]
        # print(infos)
        self.infos = infos # <--- ADD THIS LINE
        dones = data[3]
        # print(dones)
        dones_env = np.all(dones, axis=1)

        for t in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[t]:
                # 当一个episode结束时，从该线程的info中提取最终统计数据
                # info的结构是 [{...}], 我们取第一个元素
                final_info = infos[t][0] 
                
                if "prr" in final_info:
                    self.episode_prr.append(final_info["prr"])
                if "attack_success_rate" in final_info:
                    self.episode_attack_success_rate.append(final_info["attack_success_rate"])
                if "collision_rate" in final_info:
                    self.episode_collision_rate.append(final_info["collision_rate"])

    def episode_log(self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer):
        """
        定期被调用以记录训练信息。
        PATTERN: 参照smac_logger和football_logger，正确格式化数据后调用父类方法。
        """
        # 首先调用父类的episode_log来处理标准的日志输出和奖励记录
        super().episode_log(actor_train_infos, critic_train_info, actor_buffer, critic_buffer)
        
        # 1. 处理增量指标 (从最新的infos中计算)
        incre_collisions = []
        incre_message_failures = []
        
        latest_total_collisions = []
        latest_message_failures = []
    
        infos = self.infos
    
        for i in range(self.algo_args["train"]["n_rollout_threads"]):
            info = infos[i][0] # 每个线程的info
            if "total_collisions" in info:
                incre_collisions.append(info["total_collisions"] - self.last_total_collisions[i])
                latest_total_collisions.append(info["total_collisions"])
            if "message_failures" in info:
                incre_message_failures.append(info["message_failures"] - self.last_message_failures[i])
                latest_message_failures.append(info["message_failures"])
    
        # 更新 "上一个" 状态
        if latest_total_collisions:
            self.last_total_collisions = np.array(latest_total_collisions)
        if latest_message_failures:
            self.last_message_failures = np.array(latest_message_failures)
            
        # 2. 准备写入TensorBoard的自定义环境数据
        # FIX: 确保所有传递给 log_env 的值都是列表
        env_infos_log = {
            # 对于已经计算出的单一汇总值，将其包装在列表中
            "increment_collisions": [np.sum(incre_collisions)],
            "increment_message_failures": [np.sum(incre_message_failures)],
            
            # 对于在周期内收集的episodes的最终指标，直接传递原始列表
            # base_logger 会自动计算均值
            "average_prr": self.episode_prr,
            "average_attack_success_rate": self.episode_attack_success_rate,
            "average_collision_rate": self.episode_collision_rate
        }
        self.log_env(env_infos_log)
    
        # 3. 打印自定义指标到控制台
        # 注意: 我们在这里自己计算均值用于打印，但传递给log_env的是原始列表
        avg_prr = np.mean(self.episode_prr) if self.episode_prr else 0.0
        avg_attack_success = np.mean(self.episode_attack_success_rate) if self.episode_attack_success_rate else 0.0
        avg_collision_rate = np.mean(self.episode_collision_rate) if self.episode_collision_rate else 0.0
    
        print(f"On these episodes: "
              f"Avg PRR={avg_prr:.3f}, "
              f"Avg Attack Success Rate={avg_attack_success:.3f}, "
              f"Avg Collision Rate={avg_collision_rate:.3f}.")
        print(f"In this log interval: "
              f"Total new collisions={np.sum(incre_collisions):.0f}, "
              f"Total new message failures={np.sum(incre_message_failures):.0f}.\n")
    
        # 4. 清空列表，为下一个记录周期做准备
        self.episode_prr = []
        self.episode_attack_success_rate = []
        self.episode_collision_rate = []

    def eval_init(self):
        """初始化评估阶段的统计数据。"""
        super().eval_init()
        self.eval_prr = []
        self.eval_attack_success_rate = []
        self.eval_collision_rate = []
        self.eval_total_collisions = []
        self.eval_message_failures = []

    def eval_thread_done(self, tid):
        """每个评估线程结束后，累积其统计数据。"""
        super().eval_thread_done(tid)
        info = self.eval_infos[tid][0]
        if "prr" in info:
            self.eval_prr.append(info["prr"])
        if "attack_success_rate" in info:
            self.eval_attack_success_rate.append(info["attack_success_rate"])
        if "collision_rate" in info:
            self.eval_collision_rate.append(info["collision_rate"])
        if "total_collisions" in info:
            self.eval_total_collisions.append(info["total_collisions"])
        if "message_failures" in info:
            self.eval_message_failures.append(info["message_failures"])

    def eval_log(self, eval_episode):
        """在所有评估结束后，记录评估结果。"""
        # 计算评估指标的平均值
        eval_avg_prr = np.mean(self.eval_prr)
        eval_avg_attack_success = np.mean(self.eval_attack_success_rate)
        eval_avg_collision_rate = np.mean(self.eval_collision_rate)
        eval_avg_total_collisions = np.mean(self.eval_total_collisions)
        eval_avg_message_failures = np.mean(self.eval_message_failures)
        
        # 准备写入TensorBoard的数据
        eval_env_infos = {
            "eval_average_prr": [eval_avg_prr],
            "eval_average_attack_success_rate": [eval_avg_attack_success],
            "eval_average_collision_rate": [eval_avg_collision_rate],
            "eval_average_total_collisions": [eval_avg_total_collisions],
            "eval_average_message_failures": [eval_avg_message_failures]
        }
        self.log_env(eval_env_infos)
        
        # 打印自定义评估结果到控制台
        print(
            f"Evaluation Summary: \n"
            f"Avg PRR: {eval_avg_prr:.3f}, "
            f"Avg Attack Success: {eval_avg_attack_success:.3f}, "
            f"Avg Collision Rate: {eval_avg_collision_rate:.3f}\n"
            f"Avg Total Collisions: {eval_avg_total_collisions:.2f}, "
            f"Avg Message Failures: {eval_avg_message_failures:.2f}"
        )

        # 最后调用父类的 eval_log 来处理奖励的日志记录和文件写入
        super().eval_log(eval_episode)