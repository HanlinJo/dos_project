# -*- coding: utf-8 -*-s
from matplotlib.patches import Patch
import numpy as np
import random
import time
from collections import defaultdict, deque
import logging
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import math

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('V2X-RL-Environment-Enhanced-SINR')
np.random.seed(42)
# 全局攻击模式切换
TARGETED_ATTACK_MODE = True

class Packet:
    """表示用于传输的V2X数据包"""

    def __init__(self, sender_id, timestamp, position, packet_id, size=190, is_attack=False):
        self.sender_id = sender_id
        self.timestamp = timestamp
        self.position = position
        self.size = size
        self.is_attack = is_attack
        self.packet_id = packet_id
        self.expected_receivers = 0

class SensingData:
    """表示感知数据"""
    def __init__(self, slot_id, subchannel, pRsvp, sender_id, timestamp, sender_position,expected_receivers):
        self.slot_id = slot_id
        self.subchannel = subchannel
        self.pRsvp = pRsvp
        self.sender_id = sender_id
        self.timestamp = timestamp
        self.sender_position = sender_position  # 新增发送者位置信息
        self.expected_receivers = expected_receivers  # 新增期望接收者数量

class ResourceInfo:
    """表示资源块 (时隙+子信道)"""
    def __init__(self, slot_id, subchannel):
        self.slot_id = int(slot_id)
        self.subchannel = int(subchannel)
    def __eq__(self, other):
        if not isinstance(other, ResourceInfo):
            return False
        return (self.slot_id == other.slot_id and
                self.subchannel == other.subchannel)
    def __hash__(self):
        return hash((self.slot_id, self.subchannel))
    def __repr__(self):
        return f"(slot:{self.slot_id}, subchannel:{self.subchannel})"

class SINRCalculator:
    """SINR计算器 - 封装SINR计算逻辑"""

    def __init__(self, tx_power=23.0, noise_power=-95.0, use_simple_interference=False,attacker_interference_boost=5.0,normal_interference_boost=1.0):
        self.tx_power = tx_power  # 发射功率 (dBm)
        self.noise_power = noise_power  # 噪声功率 (dBm)
        self.use_simple_interference = use_simple_interference  # 是否使用简化干扰计算
        self.attacker_interference_boost = attacker_interference_boost  # 攻击者干扰增强
        self.normal_interference_boost = normal_interference_boost
    def calculate_path_loss(self, distance):
        """
        优化的路径损耗模型 - 确保远距离单发送者也能接收成功
        """
        if distance < 1:
            distance = 1

        # 修正的路径损耗模型，减少远距离损耗
        if distance <= 50:
            # 近距离使用标准模型
            return 32.45 + 20 * np.log10(distance) + 20 * np.log10(5.9)
        elif distance <= 200:
            # 中距离优化
            return 32.45 + 20 * np.log10(50) + 20 * np.log10(5.9) + 15 * np.log10(distance / 50)
        else:
            # 远距离进一步优化，确保300m+也能接收
            return 32.45 + 20 * np.log10(50) + 20 * np.log10(5.9) + 15 * np.log10(200 / 50) + 10 * np.log10(distance / 200)


    def calculate_sinr_optimized(self, receiver_pos, sender_pos, interferers_pos, tx_power=None, has_attack=False):
        tx_power = tx_power if tx_power is not None else self.tx_power
        distance = np.linalg.norm(receiver_pos - sender_pos)
        path_loss = self.calculate_path_loss(distance)
        rx_power = self.tx_power - path_loss  # dBm

        interference_power_mw = 0
        for intf_pos in interferers_pos:
            intf_distance = np.linalg.norm(receiver_pos - intf_pos)
            intf_path_loss = self.calculate_path_loss(intf_distance)
            # 普通车辆干扰也加权
            interference_factor = self.attacker_interference_boost if has_attack else self.normal_interference_boost
            intf_rx_power = self.tx_power - intf_path_loss  # dBm
            interference_power_mw += (10 ** (intf_rx_power / 10)) * interference_factor

        noise_mw = 10 ** (self.noise_power / 10)
        total_interference_mw = interference_power_mw + noise_mw

        signal_mw = 10 ** (rx_power / 10)
        sinr_linear = signal_mw / total_interference_mw
        sinr_db = 10 * np.log10(sinr_linear) if sinr_linear > 0 else -100

        return sinr_db

    def calculate_sinr_simple(self, receiver_pos, sender_pos, num_interferers, tx_power=None):
        """
        简化的SINR计算：仅基于发送者数量和路径损耗
        """
        # 计算目标信号接收功率
        tx_power = tx_power if tx_power is not None else self.tx_power
        distance = np.linalg.norm(receiver_pos - sender_pos)
        path_loss = self.calculate_path_loss(distance)
        rx_power = self.tx_power - path_loss  # dBm

        # 简化干扰计算：基于干扰者数量
        if num_interferers == 0:
            # 没有干扰 - 只有噪声
            interference_power_db = self.noise_power
        else:
            # 有干扰 - 假设平均干扰功率
            avg_interference_power = self.tx_power - 60  # 假设平均60dB路径损耗
            total_interference_mw = num_interferers * (10 ** (avg_interference_power / 10))
            noise_mw = 10 ** (self.noise_power / 10)
            interference_power_db = 10 * np.log10(total_interference_mw + noise_mw)

        # 计算SINR
        signal_mw = 10 ** (rx_power / 10)
        interference_mw = 10 ** (interference_power_db / 10)
        sinr_linear = signal_mw / interference_mw
        sinr_db = 10 * np.log10(sinr_linear) if sinr_linear > 0 else -100

        return sinr_db

class Message:
    """表示从发送者到接收者的V2X消息"""
    def __init__(self, sender_id, receiver_id, packet_id):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.packet_id = packet_id
        self.resources_received = 0  # 已接收的资源块数量
        self.success = True  # 初始假设成功，一旦失败就变为False
        self.completed = False  # 是否已完成（两个资源块都处理）

    def record_reception(self, success):
        """记录资源块接收情况"""
        self.resources_received += 1
        if not success:
            self.success = False  # 一旦失败就不可逆

    def is_completed(self):
        """检查消息是否完成（两个资源块都处理）"""
        return self.resources_received >= 2

class Vehicle:
    """表示具有V2X功能的车辆"""

    def __init__(self, vehicle_id, initial_position, initial_velocity, sim, resource_selection_mode='Combine',enable_collision_triggered_reselection=True):
        self.id = vehicle_id
        self.position = initial_position
        self.velocity = initial_velocity
        self.sim = sim
        self.resource_selection_mode = resource_selection_mode

        # 获取资源池参数
        self.num_slots = sim.resource_pool.num_slots
        self.num_subchannels = sim.resource_pool.num_subchannels

        # 资源选择参数
        self.resel_counter = 0
        self.prob_resource_keep = random.uniform(0.2, 0.8)
        self.current_resources = None
        self.sensing_data = []
        self.next_transmission_time = 0
        self.sent_resources_count = 0
        self.current_packet_id = 0
        # 感知窗口参数
        self.sensing_window_duration = 1000
        self.has_transmitted = False
        self.total_faild = 0
        self.enable_collision_triggered_reselection = enable_collision_triggered_reselection
        self.current_resources_collision = False  # 记录当前资源是否冲突
        # 初始化统计
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_received_succeed = 0
        self.collisions = 0
        self.successful_transmissions = 0
        self.prr = 0.0
        self.expected_receptions = 0
        self.successful_receptions = 0

    def reset(self):
        """重置车辆状态"""
        self.resel_counter = 0
        self.packets_received_succeed = 0
        self.current_resources = None
        self.sensing_data = []
        self.next_transmission_time = 0
        self.sent_resources_count = 0
        self.current_packet_id = 0
        self.packets_sent = 0
        self.packets_received = 0
        self.collisions = 0
        self.successful_transmissions = 0
        self.prr = 0.0
        self.expected_receptions = 0
        self.successful_receptions = 0
        self.total_faild = 0

    def move(self, delta_time):
        """基于速度和时间增量更新车辆位置"""
        self.position = self.position + self.velocity * delta_time

        # 处理边界条件（反射）
        if self.position[0] >= 1000:
            self.position[0] = 1000 - (self.position[0] - 1000)
            self.velocity = -self.velocity
            self.position[1] = 10.0
        if self.position[0] <= 0:
            self.position[0] = -self.position[0]
            self.velocity = -self.velocity
            self.position[1] = 5.0

    def get_sensed_resource_occupancy(self):
        """获取监听窗中的资源占用状态矩阵"""
        occupancy = np.zeros((self.num_slots, self.num_subchannels), dtype=int)

        for data in self.sensing_data:
            slot = data.slot_id % self.num_slots
            if 0 <= slot < self.num_slots and 0 <= data.subchannel < self.num_subchannels:
                occupancy[slot, data.subchannel] = 1

        return occupancy

    def select_future_resource(self, current_time):
        """选择未来资源 - 根据模式选择资源块"""
        self._update_sensing_window(current_time)
        selection_window = self._create_selection_window(current_time)

        # 创建已占用资源集合
        occupied_resources = set()
        for data in self.sensing_data:
            resource_key = (data.slot_id, data.subchannel)
            occupied_resources.add(resource_key)

        # 根据模式选择资源
        if self.resource_selection_mode == 'Combine':
            selected_resources = self._select_combined_resources(selection_window, occupied_resources)
        else:  # Separate模式
            selected_resources = self._select_separate_resources(selection_window, occupied_resources)

        self.resel_counter = random.randint(5, 15)
        return selected_resources

    def _select_separate_resources(self, selection_window, occupied_resources):
        """Separate模式：选择两个独立的资源块"""
        candidate_resources = []
        for resource in selection_window:
            resource_key = (resource.slot_id, resource.subchannel)
            if resource_key not in occupied_resources:
                candidate_resources.append(resource)

        min_candidates = max(1, int(0.2 * len(selection_window)))
        if len(candidate_resources) < min_candidates:
            candidate_resources = selection_window[:]

        selected_resources = []
        if len(candidate_resources) >= 2:
            selected = random.sample(candidate_resources, 2)
            selected_resources = selected
        elif len(candidate_resources) == 1:
            selected_resources = [candidate_resources[0], random.choice(selection_window)]
        else:
            slot1 = random.randint(0, self.num_slots-1)
            subchannel1 = random.randint(0, self.num_subchannels-1)
            slot2 = random.randint(0, self.num_slots-1)
            subchannel2 = random.randint(0, self.num_subchannels-1)
            selected_resources = [ResourceInfo(slot1, subchannel1), ResourceInfo(slot2, subchannel2)]

        return selected_resources

    def _select_combined_resources(self, selection_window, occupied_resources):
        """Combine模式：选择同一时隙的两个相邻子信道"""
        slot_resources = defaultdict(list)
        for resource in selection_window:
            slot_resources[resource.slot_id].append(resource)

        valid_slots = []
        for slot_id, resources in slot_resources.items():
            free_subchannels = [r.subchannel for r in resources
                               if (slot_id, r.subchannel) not in occupied_resources]

            adjacent_pairs = []
            for i in range(self.num_subchannels - 1):
                if i in free_subchannels and (i+1) in free_subchannels:
                    adjacent_pairs.append((i, i+1))

            if adjacent_pairs:
                valid_slots.append((slot_id, adjacent_pairs))

        if valid_slots:
            slot_id, adjacent_pairs = random.choice(valid_slots)
            sc1, sc2 = random.choice(adjacent_pairs)
            return [ResourceInfo(slot_id, sc1), ResourceInfo(slot_id, sc2)]

        if slot_resources:
            slot_id = random.choice(list(slot_resources.keys()))
            sc_pair = random.choice([(i, i+1) for i in range(self.num_subchannels - 1)])
            sc1, sc2 = sc_pair
            return [ResourceInfo(slot_id, sc1), ResourceInfo(slot_id, sc2)]

        return []

    def _create_selection_window(self, current_time):
        """创建选择窗口 (T1=4到T2=100)"""
        selection_window = []
        current_slot = current_time % self.num_slots
        start_slot = (current_slot + 4) % self.num_slots
        end_slot = (current_slot + 100) % self.num_slots

        if start_slot < end_slot:
            slots = range(start_slot, end_slot)
        else:
            slots = list(range(start_slot, self.num_slots)) + list(range(0, end_slot))

        for slot in slots:
            for subchannel in range(self.num_subchannels):
                selection_window.append(ResourceInfo(slot, subchannel))

        return selection_window

    def _update_sensing_window(self, current_time):
        """通过移除旧条目更新感知窗口"""
        sensing_window_start = current_time - self.sensing_window_duration
        self.sensing_data = [data for data in self.sensing_data
                            if data.timestamp >= sensing_window_start]

    def add_sensing_data(self, slot_id, subchannel, pRsvp, sender_id, timestamp, sender_position,expected_receivers):
        """添加感知数据"""
        sensing_data = SensingData(
            slot_id=slot_id,
            subchannel=subchannel,
            pRsvp=pRsvp,
            sender_id=sender_id,
            timestamp=timestamp,
            sender_position=sender_position,  # 新增发送者位置
            expected_receivers =expected_receivers
        )
        self.sensing_data.append(sensing_data)

    def handle_periodic_resource_reselection(self, current_time):
        """在周期开始时处理资源重选"""
        self.prob_resource_keep = random.uniform(0.2, 0.8)
        if not self.enable_collision_triggered_reselection:
            # print("执行")
            if self.resel_counter <= 0:
                if random.random() < self.prob_resource_keep:
                    self.resel_counter = random.randint(5, 15)
                else:
                    self.current_resources = None
                    self.resel_counter = random.randint(5, 15)
            if self.current_resources is None:
                self.current_resources = self.select_future_resource(current_time)
                self.sent_resources_count = 0
                self.current_packet_id += 1
        else:
            if self.resel_counter <= 0 or self.current_resources_collision:

                if random.random() < self.prob_resource_keep and not self.current_resources_collision:
                    self.resel_counter = random.randint(5, 15)
                else:
                    self.current_resources = None
                    self.resel_counter = random.randint(5, 15)
            if self.current_resources is None:
                self.current_resources = self.select_future_resource(current_time)
                self.current_resources_collision = False  # 重置冲突状态
                self.sent_resources_count = 0
                self.current_packet_id += 1


    def send_packet(self, current_time):
        """使用选定的资源发送数据包（现在使用两个资源块）"""
        if self.current_resources is None:
            return None

        current_slot = current_time % self.num_slots

        resources_to_send = []
        for resource in self.current_resources:
            if resource.slot_id == current_slot:
                resources_to_send.append(resource)

        if not resources_to_send:
            return None

        packet = Packet(self.id, current_time, self.position, self.current_packet_id)
        packet.expected_receivers = self._calculate_expected_receivers()
        # for resource in resources_to_send:
        #     print(self.id, "发送数据包", packet.packet_id, "到资源", resource.slot_id," ",resource.subchannel)
        transmissions = []
        for resource in resources_to_send:
            transmissions.append((packet, resource))

        self.sent_resources_count += len(resources_to_send)
        self.packets_sent += len(resources_to_send)

        if self.sent_resources_count >= 2:
            self.has_transmitted = True
            self.resel_counter -= 1
            self.sent_resources_count = 0

        return transmissions

    def _calculate_expected_receivers(self):
        """计算当前时刻能接收到该包的车辆数量（通信范围内）"""
        count = 0
        for vehicle in self.sim.vehicles:
            if vehicle.id != self.id and vehicle.should_receive_packet(self.position):
                count += 1
        return count


    def calculate_prr(self):
        """计算个人PRR"""
        if self.expected_receptions > 0:
            return self.successful_receptions/self.expected_receptions
        return 0.0

    def receive_packet(self, packet, resource, success):
        """处理接收到的数据包 - 修改为始终添加感知数据"""
        if hasattr(packet, 'is_attack') and packet.is_attack:
            pRsvp = 100
        else:
            pRsvp = 100

        self.add_sensing_data(
            resource.slot_id,
            resource.subchannel,
            pRsvp,
            packet.sender_id,
            packet.timestamp,
            packet.position,  # 新增发送者位置
            packet.expected_receivers
        )
        if not packet.is_attack:
            self.packets_received += 1
        if not packet.is_attack and success:
            self.packets_received_succeed += 1
            return True
        return False

    def should_receive_packet(self, sender_position):
        """确定该车辆是否应接收来自发送者的数据包"""
        distance = np.linalg.norm(self.position - sender_position)
        return distance <= self.sim.communication_range

class RLVehicle(Vehicle):
    """
    一个由RL驱动的车辆，它学习选择资源来抵抗攻击并与其他RL车辆合作。
    """
    def __init__(self, vehicle_id, initial_position, initial_velocity, sim, resource_selection_mode='Combine'):
        super().__init__(vehicle_id, initial_position, initial_velocity, sim, resource_selection_mode)

        # RL-specific attributes
        self.action_history = deque(maxlen=5)
        self.transmission_cycle = 100  # 每100ms传输一次
        self.sensing_window_duration = 100
        self.next_transmission_time = random.randint(0, self.transmission_cycle -1) # 错开初始传输时间
        self.is_rl_agent = True # 识别RL车辆的标志

    def reset(self):
        """为新的回合重置车辆状态。"""
        super().reset()
        self.action_history.clear()
        self.next_transmission_time = random.randint(0, self.transmission_cycle - 1)

    def is_ready_to_transmit(self, current_time):
        """检查是否到了选择动作和传输的时间。"""
        if current_time >= self.next_transmission_time:
            return True
        return False

    def select_resources_from_action(self, action):
        """
        根据RL智能体的动作选择未来的传输资源。
        动作决定了下一个100ms周期的资源。
        """
        # 动作空间现在是 Discrete(100 * 4)
        num_subchannel_choices = self.num_subchannels - 1

        # 将离散动作解码为时隙索引和子信道
        absolute_slot = action // num_subchannel_choices  # 结果: 0-99
        subchannel_start = action % num_subchannel_choices   # 结果: 0-3

        # 记录归一化前的动作，用于状态表示
        self.action_history.append((absolute_slot, subchannel_start))

        # 在“Combine”模式下选择两个相邻的资源
        sc1 = subchannel_start
        sc2 = (subchannel_start + 1) % self.num_subchannels

        resource1 = ResourceInfo(absolute_slot, sc1)
        resource2 = ResourceInfo(absolute_slot, sc2)

        self.current_resources = [resource1, resource2]
        self.sent_resources_count = 0
        self.current_packet_id += 1

        # 设置下一次动作/传输的时间
        self.next_transmission_time += self.transmission_cycle
        return self.current_resources

    def get_state(self, current_time):
        """
        为RL智能体生成状态观测。
        由于决策周期为100ms，状态表示覆盖所有100个时隙。
        """
        self._update_sensing_window(current_time)
        
        # 状态矩阵覆盖整个100ms资源池
        resource_state = np.zeros((self.num_slots, self.num_subchannels))
        attacker_state = np.zeros((self.num_slots, self.num_subchannels))

        for data in self.sensing_data:
            slot_idx = data.slot_id % self.num_slots
            
            # 检查子信道是否在有效范围内
            if not (0 <= data.subchannel < self.num_subchannels):
                continue
            
            # 将距离归一化作为特征
            distance = np.linalg.norm(self.position - data.sender_position)
            normalized_distance = max(0, 1 - min(1, distance / self.sim.communication_range))

            if data.sender_id >= self.sim.num_vehicles: # 是攻击者
                attacker_state[slot_idx, data.subchannel] = 1.0 # 标记攻击者存在
            else: # 是普通车辆
                # 只在当前值更优时更新
                if normalized_distance > resource_state[slot_idx, data.subchannel]:
                    resource_state[slot_idx, data.subchannel] = normalized_distance

        # 展平状态矩阵
        resource_state_flat = resource_state.flatten()
        attacker_state_flat = attacker_state.flatten()

        # 归一化并包含动作历史
        action_history_state = np.zeros(5 * 2)
        for i, action in enumerate(self.action_history):
            if action is not None:
                # 动作归一化 (slot/99, subchannel/3)
                norm_slot = action[0] / (self.num_slots - 1.0) if self.num_slots > 1 else 0
                norm_subchannel = action[1] / (self.num_subchannels - 2.0) if self.num_subchannels > 2 else 0
                action_history_state[i*2] = norm_slot
                action_history_state[i*2+1] = norm_subchannel

        # 组合成最终的状态向量
        full_state = np.concatenate([
            resource_state_flat,    # 车辆资源占用图 (100*5)
            # attacker_state_flat,    # 攻击者占用图 (100*5)
            action_history_state      # 动作历史 (10)
        ])

        return full_state.astype(np.float32)

    def handle_periodic_resource_reselection(self, current_time):
        """
        覆盖基类方法。对于RLVehicle，资源选择由智能体的动作驱动，
        而不是这个周期性的重选逻辑。
        """
        pass # RL智能体通过 `select_resources_from_action` 处理

class RLAttacker:
    """基于RL的攻击者"""

    def __init__(self, attacker_id, initial_position, initial_velocity, sim, action_num=2):
        self.id = attacker_id
        self.position = initial_position
        self.velocity = initial_velocity
        self.sim = sim
        self.last_collison = 0
        self.communication_range = 320
        self.num_slots = sim.resource_pool.num_slots
        self.num_subchannels = sim.resource_pool.num_subchannels
        self.action_num = action_num  # 每个智能体选择的资源块数量
        self.total_resources = self.num_slots * self.num_subchannels  # 资源块总数
        self.last_attack_success_rate = 0.0  # 上次攻击成功率
        self.next_transmission_time = 0
        self.transmission_cycle = 100
        self.current_resource = None
        self.attack_packets_sent = 0
        self.attack_success_count = 0
        self.collisions_caused = 0
        self.target_slot = -1
        self.sensing_data = []
        self.sensing_window_duration = 100
        self.last_action = None
        self.last_reward = 0
        self.resources_changed = False
        self.action_history = deque(maxlen=5)
        self.last_action_indices = np.zeros(5, dtype=np.int32)  # 存储最近5个动作的索引
        self.attack_success_this_step = False  # 新增：标记本次传输是否造成碰撞
        self.target_vehicle_id = 0
        self.target_vehicle_resources = []
        self.target_vehicle_tracking_time = 0
        self.targeted_resources = []
        self.last_prr = 1.0
        self.last_attack_count = 0
        self.last_fail_counts = 0
        # 新增：保存上次选择的资源
        self.last_selected_resources = set()
        self.last_action_index = -1  # 存储上一动作的资源索引

    def reset(self):
        """重置攻击者状态"""
        self.last_collison = 0
        self.next_transmission_time = 0
        self.current_resource = None
        self.attack_packets_sent = 0
        self.attack_success_count = 0
        self.collisions_caused = 0
        self.target_slot = -1
        self.resources_changed = False
        self.sensing_data = []
        self.last_action = None
        self.last_reward = 0
        self.attack_packets_sent = 0
        self.attack_success_count = 0
        self.last_prr = 1.0
        self.last_attack_count = 0
        self.last_fail_counts = 0
        self.action_history.clear()
        self.attack_success_this_step = False  # 新增：标记本次传输是否造成碰撞
        self.target_vehicle_id = -1
        self.target_vehicle_resources = []
        self.target_vehicle_tracking_time = 0
        self.targeted_resources = []
        self.last_action_index = -1
        self.last_selected_resources = set()
        # self.last_action_indices = np.zeros(self.action_num, dtype=np.int32)
        self.last_action_indices = np.zeros(5, dtype=np.int32)  # 存储最近5个动作的索引
        self.last_attack_success_rate = 0.0

    def move(self, delta_time):
        """更新攻击者位置"""
        self.position = self.position + self.velocity * delta_time

    def _action_to_tuple(self, a):
        if isinstance(a, np.ndarray):
            if a.ndim == 0:
                return (a.item(),)
            else:
                return tuple(a.tolist())
        if isinstance(a, (list, tuple)):
            return tuple(a)
        return (a,)

    def get_individual_resource_state(self, current_time):
        """获取单个攻击者的资源状态矩阵 (20x5)"""
        self._update_sensing_window(current_time)
        resource_state = np.zeros((20, self.num_subchannels))
        current_slot = current_time % self.num_slots
        window_start = current_slot
        window_end = (current_slot + 20) % self.num_slots

        for sensing_data in self.sensing_data:
            slot_id = sensing_data.slot_id % self.num_slots
            slot_in_window = False
            if window_start < window_end:
                if window_start <= slot_id < window_end:
                    slot_in_window = True
            else:
                if slot_id >= window_start or slot_id < window_end:
                    slot_in_window = True

            if slot_in_window and 0 <= sensing_data.subchannel < self.num_subchannels:
                if window_start <= slot_id:
                    slot_index = slot_id - window_start
                else:
                    slot_index = slot_id + (self.num_slots - window_start)
                slot_index = slot_index % 20

                sender_position = sensing_data.sender_position
                distance = np.linalg.norm(self.position - sender_position)
                normalized_distance = max(0, 1 - min(1, distance / self.sim.communication_range))
                if normalized_distance > resource_state[slot_index, sensing_data.subchannel]:
                    resource_state[slot_index, sensing_data.subchannel] = normalized_distance
        return resource_state

    def get_state(self, current_time):
        """获取RL代理的当前状态 - 包含信道状态、上一动作和时间步"""
        resource_state_flat = self.get_individual_resource_state(current_time).flatten()

        # 2. 添加动作历史（最近5个动作）
        action_history_state = np.zeros(5 * 2)  # 5个动作，每个动作2个值（slot_idx, subchannel）

        for i, action in enumerate(self.action_history):
            if action is not None:
                # 归一化动作值
                norm_slot = action[0] / 19.0  # slot_idx范围0-19
                norm_subchannel = action[1] / (self.num_subchannels -1) if self.num_subchannels > 1 else 0 # subchannel范围0-3
                action_history_state[i*2] = norm_slot
                action_history_state[i*2+1] = norm_subchannel

        # 3. 添加时间步（归一化）
        time_norm = np.array([current_time / self.sim.episode_duration])

        # 组合完整状态
        full_state = np.concatenate([
            resource_state_flat,         # 信道状态
            action_history_state,      # 动作历史 (10维)
            time_norm                   # 时间步 (1维)
        ])

        return full_state.astype(np.float32)

    def select_attack_resources(self, action):
        """基于动作选择攻击资源 - 改进版（使用20ms窗口）"""
        num_subchannel_choices = 4
    
        slot_idx = action // num_subchannel_choices
        subchannel = action % num_subchannel_choices
        self.action_history.append((slot_idx, subchannel))

        current_slot = self.sim.current_time % self.num_slots
        slot = (current_slot + slot_idx) % self.num_slots

        sc1 = subchannel
        sc2 = (subchannel + 1) % self.num_subchannels

        resource1 = ResourceInfo(slot, sc1)
        resource2 = ResourceInfo(slot, sc2)

        current_resources = [resource1, resource2]
        current_set = set(current_resources)

        resources_changed = current_set != self.last_selected_resources
        self.last_selected_resources = current_set

        return current_resources, resources_changed


    def send_attack_packet_with_action(self, current_time):
        """发送攻击数据包 - 使用解码后的资源块"""
        current_slot = current_time % self.num_slots
        attack_packets = []

        if self.current_resources:
            for resource in self.current_resources:
                if resource.slot_id == current_slot:
                    attack_packet = Packet(self.id, current_time, self.position, packet_id=0, is_attack=True)
                    attack_packets.append((attack_packet, resource))
                    self.attack_packets_sent += 1
        return attack_packets


    def calculate_reward(self, collision_count, affect_vehicles):
        reward = 0.0

        if collision_count > 0:
            reward += 3.0 * collision_count

        hub_bonus = 0
        if affect_vehicles:
            median_expected_receptions = np.median([v.expected_receptions for v in self.sim.vehicles if v.expected_receptions > 0] or [0])
            for vid in affect_vehicles:
                if 0 <= vid < len(self.sim.vehicles):
                    vehicle = self.sim.vehicles[vid]
                    if vehicle.expected_receptions > median_expected_receptions:
                        hub_bonus += 2.0
        reward += hub_bonus

        distance_reward = 0.0
        for affected_vehicle_id in affect_vehicles:
            if 0 <= affected_vehicle_id < len(self.sim.vehicles):
                dist = np.linalg.norm(np.array(self.position) - np.array(self.sim.vehicles[affected_vehicle_id].position))

                if dist < 50:
                    distance_reward += 1.0
                elif dist < 320:
                    distance_reward += 1.0 * (1 - (dist - 50) / 320)
        reward += distance_reward

        if len(affect_vehicles)>1:
            reward += 2.0*(len(affect_vehicles)-1)

        current_success_rate = self.attack_success_count / max(1, self.attack_packets_sent)
        reward += 10.0 * (current_success_rate - self.last_attack_success_rate)

        if collision_count == 0:
            reward -= 3.0

        self.last_attack_success_rate = current_success_rate
        return np.clip(reward, -10.0, 15.0)

    def record_attack_success(self, collision_occurred):
        """记录攻击成功"""
        if collision_occurred:
            self.collisions_caused += 1

    def should_receive_packet(self, sender_position):
        """攻击者可以接收通信范围内的数据包"""
        distance = np.linalg.norm(self.position - sender_position)
        return distance <= self.communication_range

    def add_sensing_data(self, slot_id, subchannel, pRsvp, sender_id, timestamp, sender_position,expected_receivers):
        """添加来自接收传输的感知数据"""
        sensing_data = SensingData(
            slot_id=slot_id,
            subchannel=subchannel,
            pRsvp=pRsvp,
            sender_id=sender_id,
            timestamp=timestamp,
            sender_position=sender_position,
            expected_receivers=expected_receivers
        )
        self.sensing_data.append(sensing_data)

        if TARGETED_ATTACK_MODE and sender_id == self.target_vehicle_id:
            resource = ResourceInfo(slot_id, subchannel)
            self.target_vehicle_resources.append(resource)

    def _update_sensing_window(self, current_time):
        """更新监听窗，移除过期数据"""
        sensing_window_start = current_time - self.sensing_window_duration

        self.sensing_data = [data for data in self.sensing_data
                            if data.timestamp >= sensing_window_start]

class FixAttacker(Vehicle):
    """固定策略攻击者"""

    def __init__(self, attacker_id, initial_position, initial_velocity, sim,
                 attack_cycle=20, num_subchannels=2, resource_selection_mode='Combine',attack_nearest=False,attack_most_and_nearest=False):
        super().__init__(attacker_id, initial_position, initial_velocity, sim, resource_selection_mode)
        self.is_attack = True
        self.attack_cycle = attack_cycle
        self.num_subchannels = num_subchannels
        self.next_attack_time = 0
        self.attack_packets_sent = 0
        self.collisions_caused = 0
        self.has_transmitted = False
        self.attack_success_this_step = False  # 新增：标记本次传输是否造成碰撞
        self.num_slots = sim.resource_pool.num_slots
        self.sensing_window_duration = 100
        self.prob_resource_keep = 0.2
        self.communication_range = 320
        self.cycle_groups = self._calculate_cycle_groups() if not TARGETED_ATTACK_MODE else []
        self.current_packet_id = 0
        self.attack_packets_sent = 0
        self.attack_success_count = 0
        self.target_vehicle_id = 0  # 默认攻击车辆0
        self.target_vehicle_resources = []
        self.target_vehicle_tracking_time = 0
        self.targeted_resources = []
        self.num_vehicles = 20
        # 目标攻击相关
        self.target_tracking_enabled = True
        self.last_target_update_time = 0

        self.attack_nearest = attack_nearest
        self.attack_most_and_nearest = attack_most_and_nearest
        self.nearest_vehicle_id = -1
        self.nearest_vehicle_resources = []
        self.best_target_vehicle_id = -1
        self.best_target_resources = []

    def reset(self):
        """重置攻击者状态"""
        super().reset()
        self.next_attack_time = 0
        self.attack_packets_sent = 0
        self.collisions_caused = 0
        self.current_packet_id = 0
        self.attack_packets_sent = 0
        self.attack_success_count = 0
        self.sensing_data = []
        self.cycle_groups = self._calculate_cycle_groups() if not TARGETED_ATTACK_MODE else []
        self.attack_success_this_step = False  # 重置标记
        self.target_vehicle_id = 1  # 重置后仍然攻击车辆0
        self.target_vehicle_resources = []
        self.target_vehicle_tracking_time = 0
        self.targeted_resources = []
        self.last_target_update_time = 0

        # 新增：重置最近车辆相关状态
        self.nearest_vehicle_id = -1
        self.nearest_vehicle_resources = []

    def _calculate_cycle_groups(self):
        """根据攻击周期计算时隙组"""
        num_groups = self.num_slots // self.attack_cycle
        groups = []
        start = 0

        for _ in range(num_groups):
            end = start + self.attack_cycle
            groups.append((start, end))
            start = end

        if start < self.num_slots:
            groups.append((start, self.num_slots))

        return groups

    def _get_current_cycle_group(self, current_time):
        """获取当前时间所属的周期组"""
        if not self.cycle_groups:
            return (0, self.num_slots)

        current_slot = current_time % self.num_slots

        for start, end in self.cycle_groups:
            if start <= current_slot < end:
                return (start, end)

        return self.cycle_groups[-1]

    def send_packet(self, current_time):
        """重写发送方法实现攻击逻辑"""
        if self.attack_most_and_nearest:
            # 攻击期望接收数最多且距离最近的车辆
            self.current_packet_id += 1
            return self._send_most_and_nearest_attack(current_time)
        elif self.attack_nearest:
            # 攻击最近车辆模式
            self.current_packet_id += 1
            return self._send_nearest_attack(current_time)
        elif TARGETED_ATTACK_MODE:
            # 目标攻击模式
            self.current_packet_id += 1
            return self._send_targeted_attack(current_time)
        else:
            # 周期组攻击模式
            return self._send_cycle_group_attack(current_time)

    def _send_most_and_nearest_attack(self, current_time):
        """执行攻击期望接收数最多且距离最近的车辆"""
        # 更新最佳目标车辆信息
        self._update_best_target_vehicle()

        current_slot = current_time % self.num_slots
        transmissions = []

        # 使用最佳目标车辆的资源进行攻击
        for resource in self.best_target_resources:
            if resource.slot_id == current_slot:
                packet = Packet(self.id, current_time, self.position, self.current_packet_id, is_attack=True)
                transmissions.append((packet, resource))
                self.attack_packets_sent += 1

        return transmissions

    def _update_best_target_vehicle(self):
        """更新最佳目标车辆（综合考虑期望接收数和距离）"""
        if not self.sensing_data:
            self.best_target_vehicle_id = -1
            self.best_target_resources = []
            return

        # 收集所有感知到的车辆信息
        vehicle_info = {}
        for data in self.sensing_data:
            if data.sender_id < self.num_vehicles:  # 只考虑正常车辆
                if data.sender_id not in vehicle_info:
                    vehicle_info[data.sender_id] = {
                        'position': data.sender_position,
                        'timestamp': data.timestamp,
                        'expected_receivers': data.expected_receivers,
                        'resources': []
                    }
                # 记录资源使用情况
                resource = ResourceInfo(data.slot_id, data.subchannel)
                vehicle_info[data.sender_id]['resources'].append(resource)

        # 如果没有车辆信息，直接返回
        if not vehicle_info:
            self.best_target_vehicle_id = -1
            self.best_target_resources = []
            return

        # 计算每个车辆的评分（综合考虑期望接收数和距离）
        best_score = -float('inf')
        best_vehicle_id = -1

        for vehicle_id, info in vehicle_info.items():
            # 计算距离
            distance = np.linalg.norm(self.position - info['position'])

            # 计算期望接收数（如果没有期望接收数信息，使用默认值）
            expected_receivers = info.get('expected_receivers', 0)

            # 计算评分：期望接收数越多越好，距离越近越好
            # 使用加权公式：score = α * expected_receivers - β * distance
            # 这里α和β是权重系数，可以根据需要调整
            alpha = 2.0  # 期望接收数的权重
            beta = 0.2   # 距离的权重

            score = alpha * expected_receivers - beta * distance

            if score > best_score:
                best_score = score
                best_vehicle_id = vehicle_id

        self.best_target_vehicle_id = best_vehicle_id

        # 更新最佳目标车辆的资源使用情况
        if best_vehicle_id != -1:
            # 只保留最佳目标车辆的资源
            self.best_target_resources = vehicle_info[best_vehicle_id]['resources']

            # 只保留最新的两个资源（假设每个包使用两个资源）
            if len(self.best_target_resources) > 2:
                # 按时间戳排序（这里简化处理，假设后面的数据更新）
                self.best_target_resources = self.best_target_resources[-2:]

    def _send_cycle_group_attack(self, current_time):
        """执行周期组攻击模式"""
        if self.current_resources is None:
            return None

        if self.has_transmitted:
            return None

        current_slot = current_time % self.num_slots

        resources_to_send = []
        for resource in self.current_resources:
            if resource.slot_id == current_slot:
                resources_to_send.append(resource)

        if not resources_to_send:
            return None

        packet = Packet(self.id, current_time, self.position, self.current_packet_id, is_attack=True)

        transmissions = []
        for resource in resources_to_send:
            transmissions.append((packet, resource))

        self.sent_resources_count += len(transmissions)
        self.attack_packets_sent += len(transmissions)

        if self.sent_resources_count >= 2:
            self.resel_counter -= 1
            if self.resel_counter <= 0:
                self.current_resources = None
            self.sent_resources_count = 0

        self.has_transmitted = True

        return transmissions

    def _send_targeted_attack(self, current_time):
        """执行目标侧链攻击模式 - 改进版"""

        current_slot = current_time % self.num_slots
        transmissions = []

        for resource in self.targeted_resources:
            if resource.slot_id == current_slot:
                packet = Packet(self.id, current_time, self.position, self.current_packet_id, is_attack=True)
                transmissions.append((packet, resource))
                self.attack_packets_sent += 1
        return transmissions

    def _send_nearest_attack(self, current_time):
        """执行攻击最近车辆模式"""
        self._update_nearest_vehicle()

        current_slot = current_time % self.num_slots
        transmissions = []

        for resource in self.nearest_vehicle_resources:
            if resource.slot_id == current_slot:
                packet = Packet(self.id, current_time, self.position, self.current_packet_id, is_attack=True)
                transmissions.append((packet, resource))
                self.attack_packets_sent += 1

        return transmissions


    def _update_nearest_vehicle(self):
        """更新最近车辆的信息和资源使用情况"""
        if not self.sensing_data:
            self.nearest_vehicle_id = -1
            self.nearest_vehicle_resources = []
            return

        min_distance = float('inf')
        nearest_vehicle_id = -1

        vehicle_positions = {}
        for data in self.sensing_data:
            if data.sender_id < self.num_vehicles:
                if data.sender_id not in vehicle_positions:
                    vehicle_positions[data.sender_id] = {
                        'position': data.sender_position,
                        'timestamp': data.timestamp,
                        'resources': []
                    }
                resource = ResourceInfo(data.slot_id, data.subchannel)
                vehicle_positions[data.sender_id]['resources'].append(resource)

        for vehicle_id, info in vehicle_positions.items():
            distance = np.linalg.norm(self.position - info['position'])
            if distance < min_distance:
                min_distance = distance
                nearest_vehicle_id = vehicle_id

        self.nearest_vehicle_id = nearest_vehicle_id

        if nearest_vehicle_id != -1:
            self.nearest_vehicle_resources = vehicle_positions[nearest_vehicle_id]['resources']
            if len(self.nearest_vehicle_resources) > 2:
                self.nearest_vehicle_resources = self.nearest_vehicle_resources[-2:]

    def _update_target_resources(self):
        """增强目标资源选择逻辑 - 无数据时随机选择或使用其他车辆"""
        target_resources = []
        for data in self.sensing_data:
            if data.sender_id == self.target_vehicle_id:
                target_resources.append((data.timestamp, data))

        if len(target_resources) >= 2:
            self.targeted_resources = []
            target_resources.sort(key=lambda x: x[0], reverse=True)
            latest_resources_data = [res_data for ts, res_data in target_resources[:2]]
            self.targeted_resources = [ResourceInfo(d.slot_id, d.subchannel) for d in latest_resources_data]
            return

    def add_sensing_data(self, slot_id, subchannel, pRsvp, sender_id, timestamp, sender_position,expected_receivers):
        """添加感知数据 - 扩展以支持目标攻击"""
        super().add_sensing_data(slot_id, subchannel, pRsvp, sender_id, timestamp, sender_position,expected_receivers)

        if TARGETED_ATTACK_MODE and sender_id == self.target_vehicle_id:
            logger.debug(f"攻击者 {self.id} 感知到目标车辆 {self.target_vehicle_id} 使用资源 (时隙:{slot_id}, 子信道:{subchannel})")

    def should_receive_packet(self, sender_position):
        """攻击者可以接收通信范围内的数据包"""
        distance = np.linalg.norm(self.position - sender_position)
        return distance <= self.communication_range

    def _update_sensing_window(self, current_time):
        """更新监听窗，移除过期数据"""
        sensing_window_start = current_time - self.sensing_window_duration

        self.sensing_data = [data for data in self.sensing_data
                            if data.timestamp >= sensing_window_start]

class ResourcePool:
    """管理V2X通信的侧链路资源池"""

    def __init__(self, num_slots=100, num_subchannels=5, subchannel_size=10):
        self.num_slots = num_slots
        self.num_subchannels = num_subchannels
        self.subchannel_size = subchannel_size
        self.total_rbs = num_subchannels * num_slots

class V2XRLEnvironment(gym.Env):
    """V2X攻击优化的RL环境 - 支持优化的SINR碰撞检测"""

    def __init__(self, kwargs):
        super(V2XRLEnvironment, self).__init__()
        
        # General parameters
        self.num_vehicles = kwargs.get('num_vehicles', 20)
        self.num_attackers = kwargs.get('num_attackers', 1)
        self.episode_duration = kwargs.get('episode_duration', 20000)
        self.communication_range = kwargs.get('communication_range', 320.0)
        self.vehicle_resource_mode = kwargs.get('vehicle_resource_mode', 'Combine')
        self.use_sinr = kwargs.get('use_sinr', True)
        # RL agent type parameters
        self.attacker_type = kwargs.get('attacker_type', 'RL')
        self.use_rl_vehicles = kwargs.get('use_rl_vehicles', False)
        
        # Attacker specific parameters
        self.fix_attacker_params = kwargs.get('fix_attacker_params', {'cycle': 20, 'num_subchannels': 2, 'attack_nearest': False, 'attack_most_and_nearest': False})
        
        # Resource and SINR parameters
        self.num_slots = kwargs.get('num_slots', 100)
        self.num_subchannels = kwargs.get('num_subchannels', 5)
        self.tx_power = kwargs.get('tx_power', 23.0)
        self.attacker_tx_power = 30.0
        self.noise_power = kwargs.get('noise_power', -95.0)
        self.sinr_threshold = kwargs.get('sinr_threshold', 0.0)
        self.use_simple_interference = kwargs.get('use_simple_interference', False)
        
        # 确保只有一种类型的RL智能体被激活
        if self.use_rl_vehicles and self.attacker_type == 'RL':
            raise ValueError("配置错误：不能同时激活RL驱动的车辆和RL驱动的攻击者。")

        ### MODIFICATION START ###
        # 设置多智能体数量
        if self.use_rl_vehicles:
            # 如果使用RL车辆，所有车辆都是智能体
            self.n_agents = self.num_vehicles
            self.rl_mode = 'vehicle'
        elif self.attacker_type == 'RL':
            self.n_agents = self.num_attackers
            self.rl_mode = 'attacker'
        else:
            self.n_agents = 0
            self.rl_mode = 'none'
        ### MODIFICATION END ###

        self.multi_agent_coordination = True if self.n_agents > 1 else False
        self.num_agents = self.n_agents # 兼容性别名

        self.total_resources = self.num_slots * self.num_subchannels
        self.step_count = 0

        self.sinr_calculator = SINRCalculator(
            tx_power=self.tx_power,
            noise_power=self.noise_power,
            use_simple_interference=self.use_simple_interference
        )

        self.resource_pool = ResourcePool(num_slots=self.num_slots, num_subchannels=self.num_subchannels, subchannel_size=10)
        self.initial_vehicle_states = None
        self.initial_attacker_states = None

        self.recent_collision_queue = deque(maxlen=100)
        self.recent_collision_rate = 0.0

        # 初始化仿真组件
        self.vehicles = []
        self.attackers = []
        self._initialize_vehicles()
        self._initialize_attackers()
        
        # 根据RL模式定义观测和动作空间
        if self.rl_mode == 'none':
            self.action_space = []
            self.observation_space = []
            self.share_observation_space = []
        else:
            if self.rl_mode == 'attacker':
                # RLAttacker: 20ms视窗, 动作空间 Discrete(20 * 4 = 80)
                total_actions = 20 * (self.num_subchannels -1) if self.num_subchannels > 1 else 20
                # 状态: 资源图(20*5) + 动作历史(10) + 时间(1)
                obs_dim = (20 * self.num_subchannels) + (5 * 2) + 1
                self.action_space = [spaces.Discrete(total_actions) for _ in range(self.n_agents)]

            elif self.rl_mode == 'vehicle':
                # RLVehicle: 100ms视窗, 动作空间 Discrete(100 * 4 = 400)
                total_actions = self.num_slots * (self.num_subchannels - 1) if self.num_subchannels > 1 else self.num_slots
                # 状态: 车辆资源图(100*5) + 攻击者图(100*5) + 动作历史(10)
                obs_dim = (self.num_slots * self.num_subchannels) + (5 * 2)
                # obs_dim = (self.num_slots * self.num_subchannels) * 2 + (5 * 2)
                self.action_space = [spaces.Discrete(total_actions) for _ in range(self.n_agents)]

            self.observation_space = [
                spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
                for _ in range(self.n_agents)
            ]

            share_obs_dim = self.num_slots * self.num_subchannels
            self.share_observation_space = [
                 spaces.Box(low=0.0, high=1.0, shape=(share_obs_dim,), dtype=np.float32)
                 for _ in range(self.n_agents)
            ]

        self.current_time = 0
        self.message_status_dict = {}
        self.message_pool = {}
        self.sinr_records = []
        self.resource_transmission_count = 0
        self.attack_effect_records = []
        self.verbose_attack_output = True
        self.reset_stats()


    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def _get_global_state(self):
        """
        生成一个真实的全局状态，表示所有正常车辆的资源预选情况。
        返回一个扁平化的一维向量。
        """
        resource_grid = np.zeros((self.num_slots, self.num_subchannels), dtype=np.float32)

        for vehicle in self.vehicles:
            if vehicle.current_resources:
                for resource in vehicle.current_resources:
                    if 0 <= resource.slot_id < self.num_slots and 0 <= resource.subchannel < self.num_subchannels:
                        resource_grid[resource.slot_id, resource.subchannel] = 1.0

        return resource_grid.flatten()
    
    def render_sensing_view(self, vehicle_id=0):
        """渲染指定车辆的监听窗视图（支持正常车辆和攻击者）"""
        if not hasattr(self, 'sensing_fig') or not hasattr(self, 'sensing_ax'):
            plt.ion()
            self.sensing_fig, self.sensing_ax = plt.subplots(figsize=(15, 8))
            self.sensing_cbar = None

        total_vehicles = len(self.vehicles) + len(self.attackers)
        if vehicle_id < 0 or vehicle_id >= total_vehicles:
            print(f"Invalid vehicle ID: {vehicle_id}. Valid range: 0-{total_vehicles-1}")
            return

        if vehicle_id < len(self.vehicles):
            vehicle = self.vehicles[vehicle_id]
            vehicle_type = "Vehicle"
        else:
            vehicle = self.attackers[vehicle_id - len(self.vehicles)]
            vehicle_type = "Attacker"

        occupancy = vehicle.get_sensed_resource_occupancy()

        attacker_occupancy = np.zeros((self.num_slots, self.num_subchannels), dtype=int)
        for data in vehicle.sensing_data:
            if data.sender_id >= len(self.vehicles):
                slot_idx = data.slot_id % self.num_slots
                if 0 <= slot_idx < self.num_slots and 0 <= data.subchannel < self.num_subchannels:
                    attacker_occupancy[slot_idx, data.subchannel] = 1

        combined_view = np.zeros_like(occupancy)
        combined_view[occupancy == 1] = 1
        combined_view[attacker_occupancy == 1] = 2

        self.sensing_ax.clear()
        cmap = matplotlib.colors.ListedColormap(['white', 'blue', 'red'])
        norm = matplotlib.colors.BoundaryNorm([0, 1, 2, 3], cmap.N)

        im = self.sensing_ax.imshow(
            combined_view.T, 
            cmap=cmap, 
            norm=norm,
            aspect='auto',
            origin='lower',
            extent=[0, self.num_slots, 0, self.num_subchannels]
        )

        current_slot = self.current_time % self.num_slots
        self.sensing_ax.axvline(x=current_slot, color='r', linestyle='-', linewidth=1)

        self.sensing_ax.set_xlabel(f'Slot (0-{self.num_slots-1})')
        self.sensing_ax.set_ylabel(f'Subchannel (0-{self.num_subchannels-1})')
        self.sensing_ax.set_xticks(np.arange(0, self.num_slots+1, max(1, self.num_slots//10)))
        self.sensing_ax.set_yticks(np.arange(0, self.num_subchannels+1, 1))
        self.sensing_ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

        min_distance = float('inf')
        if self.attackers:
            positions = [attacker.position for attacker in self.attackers]
            distances = [self._distance(vehicle.position, pos) for pos in positions]
            min_distance = min(distances)

        title = (
            f"{vehicle_type} {vehicle_id} Sensing View at {self.current_time}ms\n"
            f"Slots: {self.num_slots}, Subchannels: {self.num_subchannels}"
        )
        if self.attackers:
            title += f", Min distance to attacker: {min_distance:.2f}m"

        self.sensing_ax.set_title(title)

        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Free'),
            Patch(facecolor='blue', edgecolor='black', label='Normal Occupancy'),
            Patch(facecolor='red', edgecolor='black', label='Attacker Occupancy')
        ]
        self.sensing_ax.legend(handles=legend_elements, loc='upper right')

        plt.draw()
        plt.pause(0.001)                    
  
    def render(self, mode='human'):
        """渲染资源选择图"""
        if mode != 'human' or self.current_time % 1 != 0:
            return

        if not hasattr(self, 'fig') or not hasattr(self, 'ax'):
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(15, 8))
            self.cbar = None

        self.ax.clear()
        
        if not hasattr(self, 'resource_grid'):
             self.resource_grid = [[[] for _ in range(self.num_subchannels)] for _ in range(self.num_slots)]


        grid_data = np.zeros((self.num_subchannels, self.num_slots))
        attack_data = np.zeros((self.num_subchannels, self.num_slots))

        for slot in range(self.num_slots):
            for sc in range(self.num_subchannels):
                users = self.resource_grid[slot][sc]
                if users:
                    grid_data[sc, slot] = len(users)
                    if any(uid >= self.num_vehicles for uid in users):
                        attack_data[sc, slot] = 1

        im = self.ax.imshow(grid_data, cmap='viridis', aspect='auto', origin='lower',
                           vmin=0, vmax=3, extent=[0, self.num_slots, 0, self.num_subchannels])

        current_slot = self.current_time % self.num_slots
        self.ax.axvline(x=current_slot, color='r', linestyle='-', linewidth=1)

        for slot in range(self.num_slots):
            for sc in range(self.num_subchannels):
                users = self.resource_grid[slot][sc]
                if users:
                    user_text = ','.join(str(uid) for uid in users)
                    text_color = 'red' if any(uid >= self.num_vehicles for uid in users) else 'white'
                    self.ax.text(slot + 0.5, sc + 0.5, user_text,
                                 ha='center', va='center', fontsize=8, color=text_color)

        for slot in range(self.num_slots):
            for sc in range(self.num_subchannels):
                if attack_data[sc, slot] == 1:
                    rect = plt.Rectangle((slot, sc), 1, 1, 
                                        fill=False, edgecolor='red', linewidth=2)
                    self.ax.add_patch(rect)

        self.ax.set_xticks(np.arange(0, self.num_slots+1, max(1, self.num_slots//10)))
        self.ax.set_yticks(np.arange(0, self.num_subchannels+1, 1))
        self.ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

        self.ax.set_xlabel(f'Slot (0-{self.num_slots-1})')
        self.ax.set_ylabel(f'Subchannel (0-{self.num_subchannels-1})')
        title = f'Resource Allocation at Time: {self.current_time} ms (Current Slot: {current_slot})'
        self.ax.set_title(title)
        
        if self.cbar is None:
            self.cbar = self.fig.colorbar(im, ax=self.ax)
            self.cbar.set_label('Number of Users')
        else:
            self.cbar.update_normal(im)

        plt.draw()
        plt.pause(0.001)
        self.fig.canvas.flush_events()

    def reset(self):
        """为新的回合重置环境。"""
        self.current_time = 0
        self.recent_collision_queue.clear()
        self.recent_collision_rate = 0.0
        self.message_status_dict = {}
        self.sinr_records = []
        self.message_pool = {}
        self.reset_stats()
        self.step_count = 0
        self.attack_effect_records = []
        self.resource_transmission_count = 0

        if self.initial_vehicle_states:
            for i, vehicle in enumerate(self.vehicles):
                vehicle.reset()
                vehicle.position, vehicle.velocity = self.initial_vehicle_states[i][0].copy(), self.initial_vehicle_states[i][1].copy()
        if self.initial_attacker_states:
             for i, attacker in enumerate(self.attackers):
                attacker.reset()
                attacker.position, attacker.velocity = self.initial_attacker_states[i][0].copy(), self.initial_attacker_states[i][1].copy()

        obs = []
        # 根据哪个智能体是RL驱动的来获取观测值
        if self.rl_mode == 'attacker':
            for attacker in self.attackers:
                obs.append(attacker.get_state(self.current_time))
        elif self.rl_mode == 'vehicle':
            ### MODIFICATION START ###
            # 确保为所有车辆（现在都是RL智能体）获取状态
            for i in range(self.n_agents):
                obs.append(self.vehicles[i].get_state(self.current_time))
            ### MODIFICATION END ###
        else: # 没有RL智能体
            obs = [np.zeros(self.observation_space[0].shape, dtype=np.float32) for _ in range(self.n_agents)] if self.n_agents > 0 else []

        global_state = self._get_global_state()
        state = [global_state for _ in range(self.n_agents)]

        total_actions = self.action_space[0].n if self.n_agents > 0 else 0
        avail_actions = [np.ones(total_actions, dtype=np.float32) for _ in range(self.n_agents)]
        return obs, state, avail_actions

    def step(self, actions):
        """在环境中执行一个时间步。"""
        # 根据当前的RL训练模式，调用相应的step函数
        if self.rl_mode == 'attacker':
            return self._attacker_step(actions)
        elif self.rl_mode == 'vehicle':
            return self._vehicle_step(actions)
        else:
            # 处理没有RL智能体的情况，可能只是为了仿真
            self.current_time += 1
            done = self.current_time >= self.episode_duration
            return [], [], [], [done for _ in range(self.n_agents)], [{} for _ in range(self.n_agents)], []
        
    def _vehicle_step(self, actions):
        """处理RLVehicle是智能体的步骤。"""
        self.step_count += 1
        
        # 动作阶段: 将动作应用于所有RL车辆
        ### MODIFICATION START ###
        for i in range(self.n_agents):
             self.vehicles[i].select_resources_from_action(actions[i])
        ### MODIFICATION END ###

        # 仿真阶段 (100ms)
        step_outcomes = defaultdict(lambda: {'success': False, 'collided': False})
        
        for _ in range(100): # 仿真100ms以匹配车辆的周期
            # 在此模式下，所有车辆都是RL驱动的，它们已经选择了资源。
            # 普通车辆的SPS逻辑不再需要。

            # 移动
            for vehicle in self.vehicles:
                vehicle.move(0.001)
            for attacker in self.attackers:
                attacker.move(0.001)

            # 传输和碰撞检测
            transmission_outcomes = self._process_transmissions_with_rl()
            
            # 汇总结果用于奖励计算
            ### MODIFICATION START ###
            for vehicle_id, outcome in transmission_outcomes['outcomes'].items():
                # 确保只处理RL车辆的结果
                if vehicle_id < self.n_agents:
                    if outcome['collided']:
                        step_outcomes[vehicle_id]['collided'] = True
                    # 只有在没有碰撞的情况下才记录成功
                    elif outcome['success'] and not step_outcomes[vehicle_id]['collided']:
                        step_outcomes[vehicle_id]['success'] = True
            ### MODIFICATION END ###
            
            self.current_time += 1
            if self.current_time >= self.episode_duration:
                break
        
        # 奖励阶段
        rewards = self._calculate_rl_vehicle_rewards(step_outcomes)
        
        # 观测阶段
        next_obs = []
        ### MODIFICATION START ###
        for i in range(self.n_agents):
            next_obs.append(self.vehicles[i].get_state(self.current_time))
        ### MODIFICATION END ###

        next_global_state = self._get_global_state()
        next_state = [next_global_state for _ in range(self.n_agents)]

        done = self.current_time >= self.episode_duration
        dones = [done for _ in range(self.n_agents)]
        
        info = { 'prr': self._calculate_current_prr(), 'total_collisions': self.collision_count }
        infos = [info for _ in range(self.n_agents)]

        total_actions = self.action_space[0].n
        avail_actions = [np.ones(total_actions, dtype=np.float32) for _ in range(self.n_agents)]
        
        return next_obs, next_state, rewards, dones, infos, avail_actions

    def _calculate_rl_vehicle_rewards(self, step_outcomes):
        """为每个RLVehicle计算奖励。"""
        rewards = []
        
        # 定义奖励和惩罚值
        SUCCESS_REWARD = 3.0
        COLLISION_PENALTY = -5.0
        COOPERATIVE_BONUS = 10.0

        ### MODIFICATION START ###
        # 检查合作奖励的条件：是否有任何RL车辆发生碰撞
        any_rl_vehicle_collided = any(outcome['collided'] for vehicle_id, outcome in step_outcomes.items() if vehicle_id < self.n_agents)

        for i in range(self.n_agents):
            reward = 0.0
            vehicle_outcome = step_outcomes.get(i)
            
            if vehicle_outcome:
                if vehicle_outcome['collided']:
                    reward += COLLISION_PENALTY
                elif vehicle_outcome['success']:
                    reward += SUCCESS_REWARD
            
            # 如果没有RL车辆发生碰撞，则应用合作奖励
            if not any_rl_vehicle_collided:
                reward += COOPERATIVE_BONUS

            rewards.append([reward])
        ### MODIFICATION END ###
            
        return rewards

    def _attacker_step(self, actions):
        """处理RLAttackers是智能体的步骤。"""
        self.step_count += 1
        
        attacker_action_resources = defaultdict(list)
        
        for i, attacker in enumerate(self.attackers):
            if isinstance(attacker, RLAttacker):
                resources, changed = attacker.select_attack_resources(actions[i])
                attacker.current_resources = resources
                attacker.resources_changed = changed
                attacker_action_resources[attacker.id] = resources
            elif isinstance(attacker, FixAttacker):
                if not TARGETED_ATTACK_MODE:
                    attacker.current_resources = attacker.select_future_resource(self.current_time)
                else:
                    attacker._update_target_resources()
            attacker.sent_resources_count = 0

        total_collision_caused_due_to_attack = 0
        total_attacker_impact = defaultdict(set)

        # 在攻击者模式下，所有车辆都是普通车辆
        for vehicle in self.vehicles:
            vehicle._update_sensing_window(self.current_time)
            vehicle.handle_periodic_resource_reselection(self.current_time)
        
        # 在攻击者模式下，仿真运行20ms
        for _ in range(20):
            for vehicle in self.vehicles:
                vehicle.move(0.001)
            for attacker in self.attackers:
                attacker.move(0.001)
            
            collision_info = self._process_transmissions_with_rl()
            total_collision_caused_due_to_attack += collision_info.get('collisions_caused', 0)
            for attacker_id, affected_vehicles in collision_info.get('attacker_impact_map', {}).items():
                total_attacker_impact[attacker_id].update(affected_vehicles)

            self.current_time += 1

        next_obs = []
        for attacker in self.attackers:
             next_obs.append(attacker.get_state(self.current_time))
        
        next_global_state = self._get_global_state()
        next_state = [next_global_state for _ in range(self.n_agents)]

        team_reward = 0
        if self.attackers:
            all_affected_vehicles = set.union(*total_attacker_impact.values()) if total_attacker_impact else set()
            base_reward = self.attackers[0].calculate_reward(total_collision_caused_due_to_attack, all_affected_vehicles)
            team_reward += base_reward
            
            if self.n_agents > 1 and self.multi_agent_coordination:
                resource_list = [res for res_list in attacker_action_resources.values() for res in res_list]
                num_overlaps = (len(resource_list) - len(set(resource_list))) / 2
                team_reward -= num_overlaps * 3.0
                if len(total_attacker_impact) >= 2:
                    team_reward += 3.0
        
        rewards = [[team_reward] for _ in range(self.n_agents)]

        done = self.current_time >= self.episode_duration
        dones = [done for _ in range(self.n_agents)]

        info = {'total_collisions': self.collision_count, 'prr': self._calculate_current_prr()}
        infos = [info for _ in range(self.n_agents)]

        total_actions = self.action_space[0].n
        avail_actions = [np.ones(total_actions, dtype=np.float32) for _ in range(self.n_agents)]
        
        return next_obs, next_state, rewards, dones, infos, avail_actions

    def _process_transmissions_with_rl(self):
        """
        处理单个1ms时间片内的所有传输。
        返回详细结果用于奖励计算。
        """
        transmissions = []
        attack_transmissions = []
        
        # 从所有车辆收集传输
        for vehicle in self.vehicles:
            tx_result = vehicle.send_packet(self.current_time)
            if tx_result:
                transmissions.extend([(vehicle, packet, resource) for packet, resource in tx_result])

        # 从攻击者收集传输
        for attacker in self.attackers:
            if isinstance(attacker, RLAttacker):
                attack_result = attacker.send_attack_packet_with_action(self.current_time)
            else: # FixAttacker
                attack_result = attacker.send_packet(self.current_time)

            if attack_result:
                attack_transmissions.extend([(attacker, packet, resource) for packet, resource in attack_result])

        all_transmissions = transmissions + attack_transmissions
        if all_transmissions:
            # 核心SINR逻辑现在返回详细的结果
            return self._handle_transmissions_with_enhanced_sinr(all_transmissions)

        # 如果没有传输，返回空结果
        return {
            'collisions_caused': 0,
            'attacker_impact_map': defaultdict(set),
            'outcomes': defaultdict(dict)
        }

    def _distance(self, pos1, pos2):
        """计算两点之间的欧几里得距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _handle_transmissions_with_enhanced_sinr(self, transmissions):
        """处理传输并使用SINR进行碰撞检测"""
        tx_by_slot = defaultdict(list)
        for sender, packet, resource in transmissions:
            tx_by_slot[resource.slot_id].append((sender, packet, resource))
        
        # 这将保存所有车辆（包括RL车辆）的结果
        transmission_outcomes = defaultdict(lambda: {'success': False, 'collided': False, 'jammed': False})

        collision_info = {
            'collisions_caused': 0,
            'attacker_impact_map': defaultdict(set),
            'outcomes': transmission_outcomes # 返回详细结果
        }

        for slot_id, slot_transmissions in tx_by_slot.items():
            subchannel_usage = defaultdict(list)
            for sender, packet, resource in slot_transmissions:
                subchannel_usage[resource.subchannel].append((sender, packet, resource))

            receiver_message_status = defaultdict(lambda: {
                'success': False,
                'resources_used': 0,
                'failed_resources': set()
            })
            slot_sinr_records = []

            for subchannel, users in subchannel_usage.items():
                has_attacker = any(isinstance(sender, (RLAttacker, FixAttacker)) for sender, _, _ in users)
                has_normal = any(not isinstance(sender, (RLAttacker, FixAttacker)) for sender, _, _ in users)
                normal_users = [user for user in users if not isinstance(user[0], (RLAttacker, FixAttacker))]

                if has_attacker and has_normal:
                    attackers_on_resource = [sender for sender, _, _ in users if isinstance(sender, (RLAttacker, FixAttacker))]
                    normal_vehicles_on_resource = [sender for sender, _, _ in users if not isinstance(sender, (RLAttacker, FixAttacker))]

                    collision_info['collisions_caused'] += len(normal_vehicles_on_resource)
                    
                    for attacker in attackers_on_resource:
                        for vehicle in normal_vehicles_on_resource:
                            collision_info['attacker_impact_map'][attacker.id].add(vehicle.id)
                            # 标记此车辆的传输被干扰，用于奖励计算
                            collision_info['outcomes'][vehicle.id]['jammed'] = True
                            collision_info['outcomes'][vehicle.id]['collided'] = True # 干扰也是一种碰撞


                collision_occurred = len(normal_users) > 1
                if collision_occurred:
                    for sender, _, _ in normal_users:
                         collision_info['outcomes'][sender.id]['collided'] = True


                if self.use_sinr:
                    resource_record = {'time': self.current_time, 'resource': (slot_id, subchannel), 'senders': [], 'receivers': []}

                    for sender, packet, resource in users:
                        if len(users)>1 and not isinstance(sender, (RLAttacker, FixAttacker)):
                            sender.current_resources_collision = True
                        resource_record['senders'].append({'sender_id': sender.id, 'sender_position': sender.position.tolist(), 'is_attacker': isinstance(sender, (RLAttacker, FixAttacker))})

                    for receiver in self.vehicles:
                        receiver_sinr_info = {'receiver_id': receiver.id, 'receiver_position': receiver.position.tolist(), 'sinr_values': [], 'sender_ids': [], 'distances': []}
                        if not any(sender.id == receiver.id for sender, _, _ in users):
                            for sender, packet, resource in users:
                                if receiver.should_receive_packet(sender.position):
                                    interferers_pos = [other_sender.position for other_sender, _, _ in users if other_sender.id != sender.id]
                                    has_attack_in_interferers = any(isinstance(other_sender, (RLAttacker, FixAttacker)) for other_sender, _, _ in users if other_sender.id != sender.id)
                                    
                                    sinr = self.sinr_calculator.calculate_sinr_optimized(receiver.position, sender.position, interferers_pos, self.attacker_tx_power, has_attack_in_interferers)
                                    success = sinr >= self.sinr_threshold

                                    self.resource_transmission_count += 1
                                    if success:
                                        msg_key = (sender.id, packet.packet_id, receiver.id)
                                        receiver_message_status[msg_key]['success'] = True
                                    else:
                                        if not isinstance(sender, (RLAttacker, FixAttacker)):
                                            self.collision_count += 1
                                            receiver.total_faild += 1
                                        msg_key = (sender.id, packet.packet_id, receiver.id)
                                        receiver_message_status[msg_key]['failed_resources'].add(receiver.id)

                                    if isinstance(receiver, Vehicle):
                                        receiver.receive_packet(packet, resource, success)

                                    receiver_sinr_info['sinr_values'].append(sinr)
                                    receiver_sinr_info['sender_ids'].append(sender.id)
                                    receiver_sinr_info['distances'].append(self._distance(receiver.position, sender.position))

                        if receiver_sinr_info['sinr_values']:
                            resource_record['receivers'].append(receiver_sinr_info)
                    
                    if resource_record['receivers']:
                        slot_sinr_records.append(resource_record)

                    for receiver in self.attackers:
                        if not any(sender.id == receiver.id for sender, _, _ in users):
                            for sender, packet, resource in normal_users:
                                if receiver.should_receive_packet(sender.position):
                                    receiver.add_sensing_data(resource.slot_id, resource.subchannel, 100, sender.id, packet.timestamp, packet.position, packet.expected_receivers)
                
                for sender, packet, resource in normal_users:
                    for vehicle in self.vehicles + self.attackers:
                        if vehicle.id != sender.id and vehicle.should_receive_packet(sender.position):
                            msg_key = (sender.id, packet.packet_id, vehicle.id)
                            if msg_key not in receiver_message_status:
                                receiver_message_status[msg_key] = {'success': False, 'resources_used': 0, 'failed_resources': set()}
                            receiver_message_status[msg_key]['resources_used'] += 1
            
            self.sinr_records.extend(slot_sinr_records)

            finished_messages = set()
            for msg_key, status in receiver_message_status.items():
                sender_id, packet_id, receiver_id = msg_key
                if sender_id >= self.num_vehicles or receiver_id >= self.num_vehicles:
                    continue
                if status['resources_used'] == 2:
                    if status['failed_resources']: status['success'] = False

                    sender = next((v for v in self.vehicles if v.id == sender_id), None)
                    if sender:
                        sender.expected_receptions += 1
                        self.total_expected_packets += 1
                        if status['success']:
                            sender.successful_receptions += 1
                            self.total_received_packets += 1
                            # 标记此发送者成功，除非它已经被标记为碰撞
                            if not collision_info['outcomes'][sender_id]['collided']:
                                collision_info['outcomes'][sender_id]['success'] = True
                        else:
                            self.message_failures += 1
                            # 消息失败意味着碰撞/干扰事件
                            collision_info['outcomes'][sender_id]['collided'] = True
                        if not isinstance(sender, (RLAttacker, FixAttacker)) and not status['success']:
                            sender.collisions += 1
                    finished_messages.add(msg_key)

            for msg_key in finished_messages:
                if msg_key in receiver_message_status:
                    del receiver_message_status[msg_key]
        return collision_info

    def _calculate_current_prr(self):
        """计算当前分组接收率(PRR)"""
        if self.total_expected_packets > 0:
            return self.total_received_packets / self.total_expected_packets
        return 0.0

    def get_vehicle_prrs(self):
        """获取所有车辆的个人PRR"""
        vehicle_prrs = {}
        for vehicle in self.vehicles:
            vehicle_prrs[vehicle.id] = vehicle.calculate_prr()
        return vehicle_prrs

    def get_episode_stats(self):
        """获取当前轮的统计信息"""
        vehicle_prrs = self.get_vehicle_prrs()

        return {
            'total_collisions': self.collision_count,
            'total_transmissions': self.transmission_count,
            'prr': self._calculate_current_prr(),
            'attack_success_rate': self.total_attack_success / max(1, self.attack_transmission_count),
            'collision_rate': self.collision_count / max(1, self.resource_transmission_count),
            'message_failures': self.message_failures,
            'vehicle_prrs': vehicle_prrs,
            'step_count': self.step_count
        }

    def reset_stats(self):
        """重置所有统计"""
        self.collision_count = 0
        self.transmission_count = 0
        self.total_expected_packets = 0
        self.total_received_packets = 0
        self.attack_transmission_count = 0
        self.total_attack_success = 0
        self.message_failures = 0
        self.resource_block_attacks = 0
        self.resource_block_collisions = 0

    def _initialize_vehicles(self):
        """初始化车辆位置和速度，并保存初始状态"""
        lane1_y = 5.0
        lane2_y = 10.0
        highway_length = 1000.0
        self.vehicles = []
        vehicle_states = []

        ### MODIFICATION START ###
        # 根据 use_rl_vehicles 标志，统一创建所有车辆的类型
        for i in range(self.num_vehicles):
            lane_y = lane1_y if i % 2 == 0 else lane2_y
            pos_x = random.uniform(0, highway_length)
            position = np.array([pos_x, lane_y])
            velocity = np.array([10.0 if lane_y == lane1_y else -10.0, 0.0])

            if self.use_rl_vehicles:
                # 创建一个RLVehicle实例
                vehicle = RLVehicle(i, position.copy(), velocity.copy(), self, self.vehicle_resource_mode)
            else:
                # 创建一个标准的Vehicle实例
                vehicle = Vehicle(i, position.copy(), velocity.copy(), self, self.vehicle_resource_mode)

            self.vehicles.append(vehicle)
            vehicle_states.append((position.copy(), velocity.copy()))
        ### MODIFICATION END ###
        
        self.initial_vehicle_states = vehicle_states

    def _initialize_attackers(self):
        """初始化攻击者并保存初始状态，根据攻击者数量均匀分配其初始位置。"""
        highway_length = 1000.0
        self.attackers = []
        attacker_states = []
        
        num_attackers = self.num_attackers
        for i in range(num_attackers):
            attacker_id = self.num_vehicles + i
            
            pos_x = (i + 1) * highway_length / (num_attackers + 1)
            position = np.array([pos_x, 7.5]) 
            velocity = np.array([0.0, 0.0])

            if self.attacker_type == 'RL':
                attacker = RLAttacker(attacker_id, position.copy(), velocity.copy(), self)
            else:
                attacker = FixAttacker(
                    attacker_id,
                    position.copy(),
                    velocity.copy(),
                    self,
                    attack_cycle=self.fix_attacker_params.get('cycle', 20),
                    num_subchannels=self.fix_attacker_params.get('num_subchannels', 2),
                    attack_nearest=self.fix_attacker_params.get('attack_nearest', False),
                    attack_most_and_nearest=self.fix_attacker_params.get('attack_most_and_nearest', False)
                )
            self.attackers.append(attacker)
            attacker_states.append((position.copy(), velocity.copy()))
            
        self.initial_attacker_states = attacker_states

    def close(self):
        """关闭环境，清理资源"""
        if hasattr(self, 'fig'):
            plt.ioff()
            plt.close()
        if hasattr(self, 'sensing_fig'):
            plt.close(self.sensing_fig)