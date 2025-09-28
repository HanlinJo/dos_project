# # -*- coding: utf-8 -*-s
# from matplotlib.patches import Patch
# import numpy as np
# import random
# import time
# from collections import defaultdict, deque
# import logging
# import gym
# from gym import spaces
# import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend
# import matplotlib.pyplot as plt
# import math
# import re

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger('V2X-RL-Environment-Enhanced-SINR')
# np.random.seed(42)
# # Global attack mode switch
# TARGETED_ATTACK_MODE = True

# class Packet:
#     """Represents a V2X packet for transmission"""

#     def __init__(self, sender_id, timestamp, position, packet_id, size=190, is_attack=False):
#         self.sender_id = sender_id
#         self.timestamp = timestamp
#         self.position = position
#         self.size = size
#         self.is_attack = is_attack
#         self.packet_id = packet_id
#         self.expected_receivers = 0

# class SensingData:
#     """Represents sensing data"""
#     def __init__(self, slot_id, subchannel, pRsvp, sender_id, timestamp, sender_position,expected_receivers):
#         self.slot_id = slot_id
#         self.subchannel = subchannel
#         self.pRsvp = pRsvp
#         self.sender_id = sender_id
#         self.timestamp = timestamp
#         self.sender_position = sender_position  # Added sender position information
#         self.expected_receivers = expected_receivers  # Added number of expected receivers

# class ResourceInfo:
#     """Represents a resource block (slot + subchannel)"""
#     def __init__(self, slot_id, subchannel):
#         self.slot_id = int(slot_id)
#         self.subchannel = int(subchannel)
#     def __eq__(self, other):
#         if not isinstance(other, ResourceInfo):
#             return False
#         return (self.slot_id == other.slot_id and
#                 self.subchannel == other.subchannel)
#     def __hash__(self):
#         return hash((self.slot_id, self.subchannel))
#     def __repr__(self):
#         return f"(slot:{self.slot_id}, subchannel:{self.subchannel})"

# class SINRCalculator:
#     """SINR Calculator - Encapsulates SINR calculation logic"""

#     def __init__(self, tx_power=23.0, noise_power=-95.0, use_simple_interference=False,attacker_interference_boost=10.0,normal_interference_boost=10.0):
#         self.tx_power = tx_power  # Transmit power (dBm)
#         self.noise_power = noise_power  # Noise power (dBm)
#         self.use_simple_interference = use_simple_interference  # Whether to use simplified interference calculation
#         self.attacker_interference_boost = attacker_interference_boost  # Attacker interference enhancement
#         self.normal_interference_boost = normal_interference_boost
#     def calculate_path_loss(self, distance):
#         """
#         Optimized path loss model - ensures that a single sender at a long distance can also be received successfully
#         """
#         if distance < 1:
#             distance = 1

#         # Corrected path loss model to reduce long-distance loss
#         if distance <= 50:
#             # Use standard model for short distances
#             return 32.45 + 20 * np.log10(distance) + 20 * np.log10(5.9)
#         elif distance <= 200:
#             # Medium distance optimization
#             return 32.45 + 20 * np.log10(50) + 20 * np.log10(5.9) + 15 * np.log10(distance / 50)
#         else:
#             # Further optimization for long distances to ensure reception at 300m+
#             return 32.45 + 20 * np.log10(50) + 20 * np.log10(5.9) + 15 * np.log10(200 / 50) + 10 * np.log10(distance / 200)


#     def calculate_sinr_optimized(self, receiver_pos, sender_pos, interferers_pos, tx_power=None, has_attack=False):
#         tx_power = tx_power if tx_power is not None else self.tx_power
#         distance = np.linalg.norm(receiver_pos - sender_pos)
#         path_loss = self.calculate_path_loss(distance)
#         rx_power = self.tx_power - path_loss  # dBm

#         interference_power_mw = 0
#         for intf_pos in interferers_pos:
#             intf_distance = np.linalg.norm(receiver_pos - intf_pos)
#             intf_path_loss = self.calculate_path_loss(intf_distance)
#             # Regular vehicle interference is also weighted
#             interference_factor = self.attacker_interference_boost if has_attack else self.normal_interference_boost
#             intf_rx_power = self.tx_power - intf_path_loss  # dBm
#             interference_power_mw += (10 ** (intf_rx_power / 10)) * interference_factor

#         noise_mw = 10 ** (self.noise_power / 10)
#         total_interference_mw = interference_power_mw + noise_mw

#         signal_mw = 10 ** (rx_power / 10)
#         sinr_linear = signal_mw / total_interference_mw
#         sinr_db = 10 * np.log10(sinr_linear) if sinr_linear > 0 else -100

#         return sinr_db

#     def calculate_sinr_simple(self, receiver_pos, sender_pos, num_interferers, tx_power=None):
#         """
#         Simplified SINR calculation: based only on the number of senders and path loss
#         """
#         # Calculate target signal received power
#         tx_power = tx_power if tx_power is not None else self.tx_power
#         distance = np.linalg.norm(receiver_pos - sender_pos)
#         path_loss = self.calculate_path_loss(distance)
#         rx_power = self.tx_power - path_loss  # dBm

#         # Simplified interference calculation: based on the number of interferers
#         if num_interferers == 0:
#             # No interference - only noise
#             interference_power_db = self.noise_power
#         else:
#             # Interference - assume average interference power
#             avg_interference_power = self.tx_power - 60  # Assume average 60dB path loss
#             total_interference_mw = num_interferers * (10 ** (avg_interference_power / 10))
#             noise_mw = 10 ** (self.noise_power / 10)
#             interference_power_db = 10 * np.log10(total_interference_mw + noise_mw)

#         # Calculate SINR
#         signal_mw = 10 ** (rx_power / 10)
#         interference_mw = 10 ** (interference_power_db / 10)
#         sinr_linear = signal_mw / interference_mw
#         sinr_db = 10 * np.log10(sinr_linear) if sinr_linear > 0 else -100

#         return sinr_db

# class Message:
#     """Represents a V2X message from sender to receiver"""
#     def __init__(self, sender_id, receiver_id, packet_id):
#         self.sender_id = sender_id
#         self.receiver_id = receiver_id
#         self.packet_id = packet_id
#         self.resources_received = 0  # Number of resource blocks received
#         self.success = True  # Initially assume success, becomes False upon failure
#         self.completed = False  # Whether it is completed (both resource blocks are processed)

#     def record_reception(self, success):
#         """Record resource block reception status"""
#         self.resources_received += 1
#         if not success:
#             self.success = False  # Once failed, it is irreversible

#     def is_completed(self):
#         """Check if the message is complete (both resource blocks have been processed)"""
#         return self.resources_received >= 2

# class Vehicle:
#     """Represents a vehicle with V2X capabilities"""

#     def __init__(self, vehicle_id, initial_position, initial_velocity, sim, resource_selection_mode='Combine',enable_collision_triggered_reselection=False):
#         self.id = vehicle_id
#         self.position = initial_position
#         self.velocity = initial_velocity
#         self.sim = sim
#         self.resource_selection_mode = resource_selection_mode

#         # Get resource pool parameters
#         self.num_slots = sim.resource_pool.num_slots
#         self.num_subchannels = sim.resource_pool.num_subchannels

#         # Resource selection parameters
#         self.resel_counter = 0
#         self.prob_resource_keep = random.uniform(0.2, 0.8)
#         self.current_resources = None
#         self.sensing_data = []
#         self.next_transmission_time = 0
#         self.sent_resources_count = 0
#         self.current_packet_id = 0
#         # Sensing window parameters
#         self.sensing_window_duration = 1000
#         self.has_transmitted = False
#         self.total_faild = 0
#         self.enable_collision_triggered_reselection = enable_collision_triggered_reselection
#         self.current_resources_collision = False  # Record whether the current resource is in conflict
#         # Initialize statistics
#         self.packets_sent = 0
#         self.packets_received = 0
#         self.packets_received_succeed = 0
#         self.collisions = 0
#         self.successful_transmissions = 0
#         self.prr = 0.0
#         self.expected_receptions = 0
#         self.total_successful_receptions = 0
#         self.total_expected_receptions = 0
#         self.transmission_count = 0
#         self.transmission_on_current_resource_counter = 0

#     def reset(self):
#         """Reset vehicle state"""
#         self.resel_counter = 0
#         self.packets_received_succeed = 0
#         self.current_resources = None
#         self.sensing_data = []
#         self.next_transmission_time = 0
#         self.sent_resources_count = 0
#         self.current_packet_id = 0
#         self.packets_sent = 0
#         self.packets_received = 0
#         self.collisions = 0
#         self.successful_transmissions = 0
#         self.prr = 0.0
#         self.expected_receptions = 0
#         self.total_successful_receptions = 0
#         self.total_expected_receptions = 0
#         self.total_faild = 0
#         self.transmission_count = 0
#         self.transmission_on_current_resource_counter = 0

#     def move(self, delta_time):
#         """Update vehicle position based on velocity and time delta"""
#         self.position = self.position + self.velocity * delta_time

#         # Handle boundary conditions (reflection)
#         if self.position[0] >= 1000:
#             self.position[0] = 1000 - (self.position[0] - 1000)
#             self.velocity = -self.velocity
#             self.position[1] = 10.0
#         if self.position[0] <= 0:
#             self.position[0] = -self.position[0]
#             self.velocity = -self.velocity
#             self.position[1] = 5.0

#     def get_sensed_resource_occupancy(self):
#         """Get the resource occupancy status matrix in the listening window"""
#         occupancy = np.zeros((self.num_slots, self.num_subchannels), dtype=int)

#         for data in self.sensing_data:
#             slot = data.slot_id % self.num_slots
#             if 0 <= slot < self.num_slots and 0 <= data.subchannel < self.num_subchannels:
#                 occupancy[slot, data.subchannel] = 1
#             # print(occupancy)
#         return occupancy

#     def select_future_resource(self, current_time):
#         """Select future resources - select resource blocks based on the mode"""
#         self._update_sensing_window(current_time)
#         selection_window = self._create_selection_window(current_time)

#         # Create a set of occupied resources
#         occupied_resources = set()
#         for data in self.sensing_data:
#             resource_key = (data.slot_id, data.subchannel)
#             occupied_resources.add(resource_key)

#         # Select resources according to the mode
#         if self.resource_selection_mode == 'Combine':
#             selected_resources = self._select_combined_resources(selection_window, occupied_resources)
#         else:  # Separate mode
#             selected_resources = self._select_separate_resources(selection_window, occupied_resources)

#         self.resel_counter = random.randint(5, 15)
#         # print(selected_resources[0].slot_id,selected_resources[0].subchannel)
#         return selected_resources

#     def _select_separate_resources(self, selection_window, occupied_resources):
#         """Separate mode: select two independent resource blocks"""
#         candidate_resources = []
#         for resource in selection_window:
#             resource_key = (resource.slot_id, resource.subchannel)
#             if resource_key not in occupied_resources:
#                 candidate_resources.append(resource)

#         min_candidates = max(1, int(0.2 * len(selection_window)))
#         if len(candidate_resources) < min_candidates:
#             candidate_resources = selection_window[:]

#         selected_resources = []
#         if len(candidate_resources) >= 2:
#             selected = random.sample(candidate_resources, 2)
#             selected_resources = selected
#         elif len(candidate_resources) == 1:
#             selected_resources = [candidate_resources[0], random.choice(selection_window)]
#         else:
#             slot1 = random.randint(0, self.num_slots-1)
#             subchannel1 = random.randint(0, self.num_subchannels-1)
#             slot2 = random.randint(0, self.num_slots-1)
#             subchannel2 = random.randint(0, self.num_subchannels-1)
#             selected_resources = [ResourceInfo(slot1, subchannel1), ResourceInfo(slot2, subchannel2)]

#         return selected_resources

#     def _select_combined_resources(self, selection_window, occupied_resources):
#         """Combine mode: select two adjacent subchannels in the same slot"""
#         slot_resources = defaultdict(list)
#         for resource in selection_window:
#             slot_resources[resource.slot_id].append(resource)

#         valid_slots = []
#         for slot_id, resources in slot_resources.items():
#             free_subchannels = [r.subchannel for r in resources
#                                if (slot_id, r.subchannel) not in occupied_resources]

#             adjacent_pairs = []
#             for i in range(self.num_subchannels - 1):
#                 if i in free_subchannels and (i+1) in free_subchannels:
#                     adjacent_pairs.append((i, i+1))

#             if adjacent_pairs:
#                 valid_slots.append((slot_id, adjacent_pairs))

#         if valid_slots:
#             slot_id, adjacent_pairs = random.choice(valid_slots)
#             sc1, sc2 = random.choice(adjacent_pairs)
#             return [ResourceInfo(slot_id, sc1), ResourceInfo(slot_id, sc2)]

#         if slot_resources:
#             slot_id = random.choice(list(slot_resources.keys()))
#             sc_pair = random.choice([(i, i+1) for i in range(self.num_subchannels - 1)])
#             sc1, sc2 = sc_pair
#             return [ResourceInfo(slot_id, sc1), ResourceInfo(slot_id, sc2)]

#         return []

#     def _create_selection_window(self, current_time):
#         """Create a selection window (T1=4 to T2=100)"""
#         selection_window = []
#         current_slot = current_time % self.num_slots
#         start_slot = (current_slot + 0) % self.num_slots
#         end_slot = (current_slot + self.num_slots-1) % self.num_slots

#         if start_slot < end_slot:
#             slots = range(start_slot, end_slot)
#         else:
#             slots = list(range(start_slot, self.num_slots)) + list(range(0, end_slot))

#         for slot in slots:
#             for subchannel in range(self.num_subchannels):
#                 selection_window.append(ResourceInfo(slot, subchannel))

#         return selection_window

#     def _update_sensing_window(self, current_time):
#         """Update the sensing window by removing old entries"""
#         sensing_window_start = current_time - self.sensing_window_duration
#         self.sensing_data = [data for data in self.sensing_data
#                             if data.timestamp >= sensing_window_start]

#     def add_sensing_data(self, slot_id, subchannel, pRsvp, sender_id, timestamp, sender_position,expected_receivers):
#         """Add sensing data"""
#         sensing_data = SensingData(
#             slot_id=slot_id,
#             subchannel=subchannel,
#             pRsvp=pRsvp,
#             sender_id=sender_id,
#             timestamp=timestamp,
#             sender_position=sender_position,  # Added sender position
#             expected_receivers =expected_receivers
#         )
#         self.sensing_data.append(sensing_data)

#     def handle_periodic_resource_reselection(self, current_time):
#         """Maneja la nueva selección de recursos basándose en el modo del vehículo."""

#         # Esta rama es para el modo donde las colisiones NO provocan una nueva selección de recursos.
#         if not self.enable_collision_triggered_reselection:
#             should_reselect = False

#             # Condición 1: Se deben seleccionar recursos si no existen.
#             if self.current_resources is None:
#                 should_reselect = True

#             # Condición 2: Nueva selección forzada después de 15 mensajes en el mismo recurso.
#             # Nota: self.transmission_on_current_resource_counter debe ser incrementado después de cada transmisión completa.
#             elif self.transmission_on_current_resource_counter >= 15:
#                 should_reselect = True

#             # Condición 3: Si el contador inicial ha expirado, se vuelve a seleccionar según la probabilidad.
#             # Nota: self.resel_counter debe ser decrementado después de cada transmisión durante su período de gracia.
#             elif self.resel_counter <= 0:
#                 # La probabilidad de volver a seleccionar está entre 0.2 y 0.8.
#                 prob_to_reselect = random.uniform(0.2, 0.8)
#                 if random.random() < prob_to_reselect:
#                     should_reselect = True

#             # Si se cumple alguna condición, se realiza la nueva selección y se reinician todos los contadores.
#             if should_reselect:
#                 self.current_resources = self.select_future_resource(current_time)
#                 # Se establece una nueva cuenta atrás inicial (período de gracia) de 5 a 15.
#                 self.resel_counter = random.randint(5, 15)
#                 # Se reinicia el contador de mensajes enviados en el recurso actual.
#                 self.transmission_on_current_resource_counter = 0
#                 self.sent_resources_count = 0
#                 # self.current_packet_id += 1 # Se genera un nuevo paquete para los nuevos recursos

#         else:
#             # Esta rama se activa cuando las colisiones SÍ provocan una nueva selección de recursos.
#             should_reselect = False

#             # Condición 1: Se deben seleccionar recursos si no existen.
#             if self.current_resources is None:
#                 should_reselect = True

#             # Condición 2: Nueva selección forzada si se detectó una colisión en los recursos actuales.
#             elif self.current_resources_collision:
#                 should_reselect = True

#             # Condición 3: Nueva selección forzada después de 15 mensajes en el mismo recurso.
#             elif self.transmission_on_current_resource_counter >= 15:
#                 should_reselect = True

#             # Condición 4: Si el contador inicial ha expirado, se vuelve a seleccionar según la probabilidad.
#             elif self.resel_counter <= 0:
#                 # La probabilidad de volver a seleccionar está entre 0.2 y 0.8.
#                 prob_to_reselect = random.uniform(0.2, 0.8)
#                 if random.random() < prob_to_reselect:
#                     should_reselect = True

#             # Si se cumple alguna condición, se realiza la nueva selección y se reinician todos los contadores.
#             if should_reselect:
#                 self.current_resources = self.select_future_resource(current_time)
#                 # Se restablece el indicador de colisión ya que estamos eligiendo nuevos recursos.
#                 self.current_resources_collision = False
#                 # Se establece una nueva cuenta atrás inicial (período de gracia) de 5 a 15.
#                 self.resel_counter = random.randint(5, 15)
#                 # Se reinicia el contador de mensajes enviados en el recurso actual.
#                 self.transmission_on_current_resource_counter = 0
#                 self.sent_resources_count = 0
#                 # self.current_packet_id += 1


#     def send_packet(self, current_time):
#         """Send a packet using the selected resources (now using two resource blocks)"""
#         if self.current_resources is None:
#             return None

#         current_slot = current_time % self.num_slots
#         # print(current_slot)
#         resources_to_send = []
#         for resource in self.current_resources:
#             if resource.slot_id == current_slot:
#                 print("fale")
#                 resources_to_send.append(resource)

#         if not resources_to_send:
#             return None

#         packet = Packet(self.id, current_time, self.position, self.current_packet_id)
#         packet.expected_receivers = self._calculate_expected_receivers()
#         # self.expected_receivers = 
#         self.expected_receivers = packet.expected_receivers
#         # for resource in resources_to_send:
#         #     print(self.id, "sending packet", packet.packet_id, "to resource", resource.slot_id," ",resource.subchannel)
#         transmissions = []
#         for resource in resources_to_send:
#             transmissions.append((packet, resource))

#         self.sent_resources_count += len(resources_to_send)
#         self.packets_sent += len(resources_to_send)
#         if self.sent_resources_count >= 2:
#             self.has_transmitted = True
            
#             # --- LÓGICA AÑADIDA/MODIFICADA ---
#             # Si el período de gracia no ha terminado, decrémentalo
#             if self.resel_counter > 0:
#                 self.resel_counter -= 1
            
#             # Incrementa el contador de transmisiones en el recurso actual
#             self.transmission_on_current_resource_counter += 1
#             # --- FIN DE LA LÓGICA AÑADIDA/MODIFICADA ---
#             self.current_packet_id += 1
#             self.sent_resources_count = 0

#             # self.sent_resources_count = 0
#         # if self.sent_resources_count >= 2:
#         #     self.has_transmitted = True
#         #     self.resel_counter -= 1
            

#         return transmissions

#     def _calculate_expected_receivers(self):
#         """Calculate the number of vehicles that can receive this packet at the current time (within communication range)"""
#         count = 0
#         for vehicle in self.sim.vehicles:
#             if vehicle.id != self.id and vehicle.should_receive_packet(self.position):
#                 count += 1
#         return count


#     def calculate_prr(self):
#         """Calculate personal PRR"""
#         if self.total_expected_receptions > 0:
#             return self.total_successful_receptions/self.total_expected_receptions
#         return 0.0

#     def receive_packet(self, packet, resource, success):
#         """Process received packets - modified to always add sensing data"""
#         if hasattr(packet, 'is_attack') and packet.is_attack:
#             pRsvp = 100
#         else:
#             pRsvp = 100

#         self.add_sensing_data(
#             resource.slot_id,
#             resource.subchannel,
#             pRsvp,
#             packet.sender_id,
#             packet.timestamp,
#             packet.position,  # Added sender position
#             packet.expected_receivers
#         )
#         if not packet.is_attack:
#             self.packets_received += 1
#         if not packet.is_attack and success:
#             self.packets_received_succeed += 1
#             return True
#         return False

#     def should_receive_packet(self, sender_position):
#         """Determine whether this vehicle should receive a packet from the sender"""
#         distance = np.linalg.norm(self.position - sender_position)
#         return distance <= self.sim.communication_range

# class RLVehicle:
#     """RL-based sender"""

#     def __init__(self, attacker_id, initial_position, initial_velocity, sim, action_num=2):
#         self.id = attacker_id
#         self.position = initial_position
#         self.velocity = initial_velocity
#         self.sim = sim
#         self.last_collison = 0
#         self.communication_range = 320
#         self.num_slots = sim.resource_pool.num_slots
#         self.num_subchannels = sim.resource_pool.num_subchannels
#         self.action_num = action_num  # Number of resource blocks selected by each agent
#         self.total_resources = self.num_slots * self.num_subchannels  # Total number of resource blocks
        
#         self.next_transmission_time = 0
#         self.transmission_cycle = 100
#         self.current_resources = None
#         self.normal_packets_sent = 0
#         self.normal_success_count = 0
#         self.collisions_caused = 0
#         self.target_slot = -1
#         self.sensing_data = []
#         self.sensing_window_duration = 100
#         self.last_action = None
#         self.last_reward = 0
#         self.resources_changed = False
#         self.action_history = deque(maxlen=5)
#         self.last_action_indices = np.zeros(5, dtype=np.int32)  # Store the indices of the last 5 actions
#         self.send_success_this_step = False  # Added: flag whether this transmission caused a collision
#         self.target_vehicle_id = 0
#         self.target_vehicle_resources = []
#         self.target_vehicle_tracking_time = 0
#         self.targeted_resources = []
#         self.last_prr = 1.0
#         self.last_send_count = 0
#         self.last_fail_counts = 0
#         self.packets_received =0
#         self.packets_received_succeed =0
#         self.total_successful_receptions = 0
#         self.total_expected_receptions = 0
#         # Added: save the last selected resources
#         self.last_selected_resources = set()
#         self.last_action_index = -1  # Store the resource index of the previous action
#         self.packets_sent = 0
#         self.packets_received = 0
#         self.packets_received_succeed = 0
#         self.collisions = 0
#         self.successful_transmissions = 0
#         self.prr = 0.0
#         self.expected_receptions = 0
#         self.successful_receptions = 0
#         self.total_faild = 0
#         self.successful_sends = 0
#         self.total_sends = 0
#         self.last_success_rate = 1.0
#         self.current_packet_id = 0
#         self.transmission_count = 0

#     def reset(self):
#         """Reset attacker state"""
#         self.last_collison = 0
#         self.next_transmission_time = 0
#         self.current_resources = None
#         self.normal_packets_sent = 0
#         self.normal_success_count = 0
#         self.collisions_caused = 0
#         self.target_slot = -1
#         self.resources_changed = False
#         self.sensing_data = []
#         self.last_action = None
#         self.last_reward = 0

#         self.last_prr = 1.0
#         self.last_send_count = 0
#         self.last_fail_counts = 0
#         self.total_successful_receptions = 0
#         self.total_expected_receptions = 0
#         self.action_history.clear()
#         self.send_success_this_step = False  # Added: flag whether this transmission caused a collision
#         self.target_vehicle_id = 0
#         self.target_vehicle_resources = []
#         self.target_vehicle_tracking_time = 0
#         self.targeted_resources = []
#         self.last_action_index = -1
#         self.packets_received =0
#         self.packets_received_succeed =0
#         self.packets_sent = 0
#         self.packets_received = 0
#         self.packets_received_succeed = 0
#         self.collisions = 0
#         self.successful_transmissions = 0
#         self.prr = 0.0
#         self.expected_receptions = 0
#         self.successful_receptions = 0
#         self.last_selected_resources = set()
#         # self.last_action_indices = np.zeros(self.action_num, dtype=np.int32)
#         self.last_action_indices = np.zeros(5, dtype=np.int32)  # Store the indices of the last 5 actions
#         self.total_faild = 0
#         self.successful_sends = 0
#         self.total_sends = 0
#         self.last_success_rate = 1.0
#         self.current_packet_id = 0
#         self.transmission_count = 0

#     def move(self, delta_time):
#         """Update vehicle position based on velocity and time delta"""
#         self.position = self.position + self.velocity * delta_time

#         # Handle boundary conditions (reflection)
#         if self.position[0] >= 1000:
#             self.position[0] = 1000 - (self.position[0] - 1000)
#             self.velocity = -self.velocity
#             self.position[1] = 10.0
#         if self.position[0] <= 0:
#             self.position[0] = -self.position[0]
#             self.velocity = -self.velocity
#             self.position[1] = 5.0

#     def _action_to_tuple(self, a):
#         if isinstance(a, np.ndarray):
#             if a.ndim == 0:
#                 return (a.item(),)
#             else:
#                 return tuple(a.tolist())
#         if isinstance(a, (list, tuple)):
#             return tuple(a)
#         return (a,)

#     def get_individual_resource_state(self, current_time):
#         """
#         MODIFICATION: Simplifies the state representation.
#         A resource is marked as 1 if any transmission is sensed, 0 otherwise.
#         The state is an absolute representation of the entire pool.
#         """
#         self._update_sensing_window(current_time)
#         # State matrix covers all slots, initialized to 0 (free)
#         resource_state = np.zeros((self.num_slots, self.num_subchannels))
        
#         # Iterate through all sensed data points
#         for data in self.sensing_data:
#             slot_id = data.slot_id % self.num_slots
#             subchannel_id = data.subchannel
            
#             # Check for valid subchannel index
#             if 0 <= subchannel_id < self.num_subchannels:
#                 # Mark the resource as occupied (1)
#                 resource_state[slot_id, subchannel_id] = 1
                
#         return resource_state

#     def get_state(self, current_time):
#         """Get the current state of the RL agent - including channel state, previous action, and time step"""
       
#         resource_state_flat = self.get_individual_resource_state(current_time).flatten()

#         # 2. Add action history (last 5 actions) - MODIFIED
#         action_history_state = np.zeros(5)  # 5 actions, each action is a single value

#         all_last_actions_state = np.full(self.sim.num_vehicles, -1.0, dtype=np.float32)
#         total_actions = self.sim.action_space[self.id].n
#         for vehicle in self.sim.vehicles:
#             # This part about other vehicles' actions remains the same as it's about observable outcomes
#             if isinstance(vehicle, RLVehicle) and vehicle.action_history:
#                 last_action = vehicle.action_history[-1]
#                 action_norm = last_action/total_actions
                
#                 all_last_actions_state[vehicle.id] = action_norm

#         # MODIFIED: New logic for self action history
        
#         for i, action in enumerate(self.action_history):
#             if action is not None:
#                 action_history_state[i] = action / total_actions

#         # 3. Add time step (normalized)
#         time_norm = np.array([current_time / self.sim.episode_duration])

#         full_state = np.concatenate([
#             resource_state_flat,         # Channel state
#             action_history_state,      # Action history (5-dimensional)
#             time_norm,                 # Time step (1-dimensional)
#             all_last_actions_state     # Previous actions of other vehicles
#         ])

#         return full_state.astype(np.float32)

#     def select_send_resources(self, action):
#         """Select send resources based on the action - improved version (using 100ms window)"""
#         # MODIFIED: Store the raw action in history
#         self.action_history.append(action)

#         # Action decoding: action index to resource index
#         num_subchannel_choices = self.sim.num_subchannels - 1
#         if num_subchannel_choices <= 0: num_subchannel_choices = 1

#         slot_idx = action // num_subchannel_choices
#         subchannel = action % num_subchannel_choices
        
#         # Get the current slot
#         current_slot = self.sim.current_time % self.num_slots
#         # Calculate the absolute slot (current slot + action index)
#         slot = (current_slot + slot_idx) % self.num_slots

#         # Select two adjacent resources
#         sc1 = subchannel
#         sc2 = subchannel + 1

#         # Create resource objects
#         resource1 = ResourceInfo(slot, sc1)
#         resource2 = ResourceInfo(slot, sc2)

#         # Save the currently selected resources
#         current_resources = [resource1, resource2]
#         current_set = set(current_resources)
#         # Check if the resources have changed
#         resources_changed = current_set != self.last_selected_resources
#         self.last_selected_resources = current_set

#         return current_resources, resources_changed

#     def record_send_success(self, success_send):
#         """Record successful sending"""
#         if success_send:
#             self.normal_success_count += 1

#     def should_receive_packet(self, sender_position):
#         """The attacker can receive packets within the communication range"""
#         distance = np.linalg.norm(self.position - sender_position)
#         return distance <= self.communication_range
    
#     def calculate_reward(self, was_successful):
#         """Calculates the base reward based on transmission outcome."""
#         current_success_rate = self.successful_sends / self.total_sends if self.total_sends > 0 else 0.0
#         self.last_success_rate = current_success_rate
#         success_rate_reward = (current_success_rate - self.last_success_rate) * 5.0 # Weighting factor
#         if was_successful:
#             return 2.0 + success_rate_reward  # Reward for successful transmission
#         else:
#             return -5.0 + success_rate_reward # Penalty for collision
        
#     def add_sensing_data(self, slot_id, subchannel, pRsvp, sender_id, timestamp, sender_position,expected_receivers):
#         """Add sensing data from received transmissions"""
#         sensing_data = SensingData(
#             slot_id=slot_id,
#             subchannel=subchannel,
#             pRsvp=pRsvp,
#             sender_id=sender_id,
#             timestamp=timestamp,
#             sender_position=sender_position,  # Added sender position
#             expected_receivers=expected_receivers
#         )
#         self.sensing_data.append(sensing_data)

#     def receive_packet(self, packet, resource, success):
#         """Process received packets - modified to always add sensing data"""
#         if hasattr(packet, 'is_attack') and packet.is_attack:
#             pRsvp = 100
#         else:
#             pRsvp = 100

#         self.add_sensing_data(
#             resource.slot_id,
#             resource.subchannel,
#             pRsvp,
#             packet.sender_id,
#             packet.timestamp,
#             packet.position,  # Added sender position
#             packet.expected_receivers
#         )
#         if not packet.is_attack:
#             self.packets_received += 1
#         if not packet.is_attack and success:
#             self.packets_received_succeed += 1
#             return True
#         return False
    
#     def _update_sensing_window(self, current_time):
#         """Update the listening window, remove expired data"""
#         sensing_window_start = current_time - self.sensing_window_duration

#         self.sensing_data = [data for data in self.sensing_data
#                             if data.timestamp >= sensing_window_start]


# class RLAttacker:
#     """RL-based attacker"""

#     def __init__(self, attacker_id, initial_position, initial_velocity, sim, action_num=2):
#         self.id = attacker_id
#         self.position = initial_position
#         self.velocity = initial_velocity
#         self.sim = sim
#         self.last_collison = 0
#         self.communication_range = 320
#         self.num_slots = sim.resource_pool.num_slots
#         self.num_subchannels = sim.resource_pool.num_subchannels
#         self.action_num = action_num  # Number of resource blocks selected by each agent
#         self.total_resources = self.num_slots * self.num_subchannels  # Total number of resource blocks
#         self.last_attack_success_rate = 0.0  # Last attack success rate
#         self.next_transmission_time = 0
#         self.transmission_cycle = 100
#         self.current_resources = None
#         self.attack_packets_sent = 0
#         self.attack_success_count = 0
#         self.collisions_caused = 0
#         self.target_slot = -1
#         self.sensing_data = []
#         self.sensing_window_duration = 100
#         self.last_action = None
#         self.last_reward = 0
#         self.resources_changed = False
#         self.action_history = deque(maxlen=5)
#         self.last_action_indices = np.zeros(5, dtype=np.int32)  # Store the indices of the last 5 actions
#         self.attack_success_this_step = False  # Added: flag whether this transmission caused a collision
#         self.target_vehicle_id = 0
#         self.target_vehicle_resources = []
#         self.target_vehicle_tracking_time = 0
#         self.targeted_resources = []
#         self.last_prr = 1.0
#         self.last_attack_count = 0
#         self.last_fail_counts = 0
#         self.current_packet_id = 0
#         # Added: save the last selected resources
#         self.last_selected_resources = set()
#         self.last_action_index = -1  # Store the resource index of the previous action
#         self.transmission_count = 0

#     def reset(self):
#         """Reset attacker state"""
#         self.last_collison = 0
#         self.next_transmission_time = 0
#         self.current_resources = None
#         self.attack_packets_sent = 0
#         self.attack_success_count = 0
#         self.collisions_caused = 0
#         self.target_slot = -1
#         self.resources_changed = False
#         self.sensing_data = []
#         self.last_action = None
#         self.last_reward = 0
#         self.attack_packets_sent = 0
#         self.attack_success_count = 0
#         self.current_packet_id = 0
#         self.last_prr = 1.0
#         self.last_attack_count = 0
#         self.last_fail_counts = 0
#         self.action_history.clear()
#         self.attack_success_this_step = False  # Added: flag whether this transmission caused a collision
#         self.target_vehicle_id = -1
#         self.target_vehicle_resources = []
#         self.target_vehicle_tracking_time = 0
#         self.targeted_resources = []
#         self.last_action_index = -1
#         self.last_selected_resources = set()
#         self.last_action_indices = np.zeros(5, dtype=np.int32)  # Store the indices of the last 5 actions
#         self.last_attack_success_rate = 0.0
#         self.transmission_count = 0

#     def move(self, delta_time):
#         """Update attacker position"""
#         self.position = self.position + self.velocity * delta_time

#     def _action_to_tuple(self, a):
#         if isinstance(a, np.ndarray):
#             if a.ndim == 0:
#                 return (a.item(),)
#             else:
#                 return tuple(a.tolist())
#         if isinstance(a, (list, tuple)):
#             return tuple(a)
#         return (a,)

#     def get_individual_resource_state(self, current_time):
#         """Get the resource state matrix of a single attacker (20xN_subchannels)"""
#         self._update_sensing_window(current_time)
#         resource_state = np.zeros((20, self.num_subchannels))
#         current_slot = current_time % self.num_slots
#         window_start = current_slot
#         window_end = (current_slot + 20) % self.num_slots

#         for sensing_data in self.sensing_data:
#             slot_id = sensing_data.slot_id % self.num_slots
            
#             # Check if the slot is within the [current_slot, current_slot + 20) window
#             if window_start < window_end: # Window does not wrap around
#                  if not (window_start <= slot_id < window_end):
#                     continue
#             else: # Window wraps around
#                 if not (slot_id >= window_start or slot_id < window_end):
#                     continue
            
#             if 0 <= sensing_data.subchannel < self.num_subchannels:
#                 # Calculate the index relative to the window start
#                 slot_index = (slot_id - window_start + self.num_slots) % self.num_slots

#                 if 0 <= slot_index < 20:
#                     sender_position = sensing_data.sender_position
#                     distance = np.linalg.norm(self.position - sender_position)
#                     normalized_distance = max(0, 1 - min(1, distance / self.sim.communication_range))
#                     if normalized_distance > resource_state[slot_index, sensing_data.subchannel]:
#                         resource_state[slot_index, sensing_data.subchannel] = normalized_distance
#         return resource_state


#     def get_state(self, current_time):
#         """Get the current state of the RL agent - including channel state, previous action, and time step"""
#         resource_state_flat = self.get_individual_resource_state(current_time).flatten()

#         # 2. Add action history (last 5 actions) - MODIFIED
#         action_history_state = np.zeros(5)  # 5 actions, each action is a single value

#         # MODIFIED: New logic for self action history
#         attacker_agent_index = self.id - self.sim.num_vehicles
#         total_actions = self.sim.action_space[attacker_agent_index].n
#         for i, action in enumerate(self.action_history):
#             if action is not None:
#                 action_history_state[i] = action / total_actions

#         # 3. Add time step (normalized)
#         time_norm = np.array([current_time / self.sim.episode_duration])

#         # Combine the full state
#         full_state = np.concatenate([
#             resource_state_flat,         # Channel state
#             action_history_state,      # Action history (5-dimensional)
#             time_norm                   # Time step (1-dimensional)
#         ])

#         return full_state.astype(np.float32)

#     def select_attack_resources(self, action):
#         """Select attack resources based on the action - improved version (using 20ms window)"""
#         # MODIFIED: Store the raw action in history
#         self.action_history.append(action)
        
#         # Action decoding: action index to resource index
#         num_subchannel_choices = self.sim.num_subchannels - 1
#         if num_subchannel_choices <= 0: num_subchannel_choices = 1
    
#         slot_idx = action // num_subchannel_choices
#         subchannel = action % num_subchannel_choices

#         # Get the current slot
#         current_slot = self.sim.current_time % self.num_slots
#         # Calculate the absolute slot (current slot + action index)
#         slot = (current_slot + slot_idx) % self.num_slots

#         # Select two adjacent resources
#         sc1 = subchannel
#         sc2 = subchannel + 1 

#         # Create resource objects
#         resource1 = ResourceInfo(slot, sc1)
#         resource2 = ResourceInfo(slot, sc2)

#         # Save the currently selected resources
#         current_resources = [resource1, resource2]
#         current_set = set(current_resources)

#         # Check if the resources have changed
#         resources_changed = current_set != self.last_selected_resources
#         self.last_selected_resources = current_set

#         return current_resources, resources_changed

#     def calculate_reward(self, collision_count, affect_vehicles):
#         reward = 0.0

#         # 1. Core collision reward (weighted by the number of affected vehicles)
#         if collision_count > 0:
#             reward += 3.0 * collision_count

#         # 2. Added: high-value target reward (attacking communication hubs)
#         hub_bonus = 0
#         if affect_vehicles: # Ensure the set is not empty
#             median_expected_receptions = np.median([v.expected_receptions for v in self.sim.vehicles if v.expected_receptions > 0] or [0])
#             for vid in affect_vehicles:
#                 # Check if vid is valid
#                 if 0 <= vid < len(self.sim.vehicles):
#                     vehicle = self.sim.vehicles[vid]
#                     if vehicle.expected_receptions > median_expected_receptions:
#                         hub_bonus += 2.0
#         reward += hub_bonus

#         distance_reward = 0.0
#         for affected_vehicle_id in affect_vehicles:
#             if 0 <= affected_vehicle_id < len(self.sim.vehicles):
#                 # Calculate the distance between the attacker and the affected vehicle
#                 dist = np.linalg.norm(np.array(self.position) - np.array(self.sim.vehicles[affected_vehicle_id].position))

#                 # Give a reward based on the distance
#                 if dist < 50:
#                     distance_reward += 1.0  # High reward for close distance
#                 elif dist < 320:
#                     # Linearly decay between 50-300 meters
#                     distance_reward += 1.0 * (1 - (dist - 50) / 320)
#         reward += distance_reward

#         # if len(affect_vehicles)>1:
#         #     reward -= 2.0*(len(affect_vehicles)-1)
#         # 3. Efficiency reward (simplified version)
#         current_success_rate = self.attack_success_count / max(1, self.attack_packets_sent)
#         reward += 10.0 * (current_success_rate - self.last_attack_success_rate)

#         # 4. Penalty term
#         if collision_count == 0:
#             reward -= 5.0  # Penalty for not causing a collision

#         self.last_attack_success_rate = current_success_rate
#         return np.clip(reward, -10.0, 15.0)

#     def record_attack_success(self, collision_occurred):
#         """Record successful attacks"""
#         if collision_occurred:
#             self.collisions_caused += 1

#     def should_receive_packet(self, sender_position):
#         """The attacker can receive packets within the communication range"""
#         distance = np.linalg.norm(self.position - sender_position)
#         return distance <= self.communication_range

#     def add_sensing_data(self, slot_id, subchannel, pRsvp, sender_id, timestamp, sender_position,expected_receivers):
#         """Add sensing data from received transmissions"""
#         sensing_data = SensingData(
#             slot_id=slot_id,
#             subchannel=subchannel,
#             pRsvp=pRsvp,
#             sender_id=sender_id,
#             timestamp=timestamp,
#             sender_position=sender_position,  # Added sender position
#             expected_receivers=expected_receivers
#         )
#         self.sensing_data.append(sensing_data)

#         if TARGETED_ATTACK_MODE and sender_id == self.target_vehicle_id:
#             resource = ResourceInfo(slot_id, subchannel)
#             self.target_vehicle_resources.append(resource)

#     def _update_sensing_window(self, current_time):
#         """Update the listening window, remove expired data"""
#         sensing_window_start = current_time - self.sensing_window_duration

#         self.sensing_data = [data for data in self.sensing_data
#                             if data.timestamp >= sensing_window_start]

# class FixAttacker(Vehicle):
#     """Fixed strategy attacker"""

#     def __init__(self, attacker_id, initial_position, initial_velocity, sim,
#                  attack_cycle=20, num_subchannels=2, resource_selection_mode='Combine',
#                  attack_nearest=False, attack_most_and_nearest=False,
#                  attack_fixed_subchannel=False, attack_periodic_group=False, is_target_attacker=True):
#         super().__init__(attacker_id, initial_position, initial_velocity, sim, resource_selection_mode)
#         self.is_attack = True
#         self.is_target_attacker = is_target_attacker
#         self.attack_cycle = attack_cycle
#         self.num_subchannels_param = num_subchannels 
#         self.next_attack_time = 0
#         self.attack_packets_sent = 0
#         self.collisions_caused = 0
#         self.has_transmitted = False
#         self.attack_success_this_step = False
#         self.num_slots = sim.resource_pool.num_slots
#         self.sensing_window_duration = 200
#         self.prob_resource_keep = 0.2
#         self.communication_range = 320
#         self.cycle_groups = self._calculate_cycle_groups() if not TARGETED_ATTACK_MODE else []
#         self.current_packet_id = 0
#         self.attack_success_count = 0
#         self.target_vehicle_id = 1
#         self.target_vehicle_resources = []
#         self.target_vehicle_tracking_time = 0
#         self.targeted_resources = []
#         self.num_vehicles = sim.num_vehicles
#         self.target_tracking_enabled = True
#         self.last_target_update_time = 0
#         self.attack_nearest = attack_nearest
#         self.attack_most_and_nearest = attack_most_and_nearest
#         self.nearest_vehicle_id = -1
#         self.nearest_vehicle_resources = []
#         self.best_target_vehicle_id = -1
#         self.best_target_resources = []
#         self.attack_fixed_subchannel = attack_fixed_subchannel
#         self.attack_periodic_group = attack_periodic_group
#         self.transmission_count = 0

#     def reset(self):
#         """Reset attacker state"""
#         super().reset()
#         self.next_attack_time = 0
#         self.attack_packets_sent = 0
#         self.collisions_caused = 0
#         self.current_packet_id = 0
#         self.attack_success_count = 0
#         self.sensing_data = []
#         self.cycle_groups = self._calculate_cycle_groups() if not TARGETED_ATTACK_MODE else []
#         self.attack_success_this_step = False
#         self.target_vehicle_id = 1
#         self.target_vehicle_resources = []
#         self.target_vehicle_tracking_time = 0
#         self.targeted_resources = []
#         self.last_target_update_time = 0
#         self.nearest_vehicle_id = -1
#         self.nearest_vehicle_resources = []
#         self.transmission_count = 0

#     def _calculate_cycle_groups(self):
#         """Calculate slot groups based on the attack cycle"""
#         num_groups = self.num_slots // self.attack_cycle
#         groups = []
#         start = 0
#         for _ in range(num_groups):
#             end = start + self.attack_cycle
#             groups.append((start, end))
#             start = end
#         if start < self.num_slots:
#             groups.append((start, self.num_slots))
#         return groups
    
#     def _send_fixed_subchannel_attack(self, current_time):
#         """Send jamming messages on subchannels 0 and 4 in all slots"""
#         transmissions = []
#         current_slot = current_time % self.num_slots
#         attack_subchannels = [0, 4]
#         packet = Packet(self.id, current_time, self.position, self.current_packet_id, is_attack=True)
#         for sc in attack_subchannels:
#             if sc < self.sim.num_subchannels:
#                 resource = ResourceInfo(current_slot, sc)
#                 self.current_resources = [resource]
#                 return
    
#     def _send_periodic_group_attack(self, current_time):
#         """Send jamming messages on all subchannels in slots 0, 10, 20...90"""
#         current_slot = current_time % self.num_slots
#         if current_slot % 10 == 0:
#             resources = []
#             for sc in range(self.sim.num_subchannels):
#                 resources.append(ResourceInfo(current_slot, sc))
#             self.current_resources = resources
#             return

#     def select_and_set_resources(self, current_time):
#         """ New method for the step-based logic. Selects resources but doesn't send. """
#         self.current_packet_id += 1
#         self.current_resources = [] # Clear previous selection
#         #attacker0
#         # return 
#         if self.is_target_attacker:
#             self._update_target_resources() # This updates self.targeted_resources
#             self.current_resources = self.targeted_resources
#         elif self.attack_fixed_subchannel:
#             # This attack happens every time slot, so we need to determine resources based on current time
#             attack_subchannels = [0, self.sim.num_subchannels -1] # e.g. [0, 4]
#             # In the new logic, we pre-select all resources for the entire step duration
#             for t_offset in range(self.sim.step_duration):
#                 slot = (current_time + t_offset) % self.num_slots
#                 for sc in attack_subchannels:
#                     if sc < self.sim.num_subchannels:
#                         self.current_resources.append(ResourceInfo(slot, sc))
#         elif self.attack_periodic_group:
#             # Pre-select all resources for the entire step duration
#             for t_offset in range(self.sim.step_duration):
#                 slot = (current_time + t_offset) % self.num_slots
#                 if slot % 10 == 0:
#                     for sc in range(self.sim.num_subchannels):
#                         self.current_resources.append(ResourceInfo(slot, sc))
#         elif self.attack_most_and_nearest:
#             self._update_best_target_vehicle()
#             self.current_resources = self.best_target_resources
#         elif self.attack_nearest:
#             self._update_nearest_vehicle()
#             self.current_resources = self.nearest_vehicle_resources
#         else: # Fallback to cycle group
#             # This logic needs adaptation for step-based selection
#             # For simplicity, let's make it select one random resource for now
#             slot = random.randint(0, self.num_slots-1)
#             sc1 = random.randint(0, self.num_subchannels-2)
#             self.current_resources = [ResourceInfo(slot, sc1), ResourceInfo(slot, sc1+1)]

#     def _update_best_target_vehicle(self):
#         """Update the best target vehicle (comprehensively considering the number of expected receivers and distance)"""
#         if not self.sensing_data:
#             self.best_target_vehicle_id = -1
#             self.best_target_resources = []
#             return

#         vehicle_info = {}
#         for data in self.sensing_data:
#             if data.sender_id < self.num_vehicles:
#                 if data.sender_id not in vehicle_info:
#                     vehicle_info[data.sender_id] = {
#                         'position': data.sender_position,
#                         'timestamp': data.timestamp,
#                         'expected_receivers': data.expected_receivers,
#                         'resources': []
#                     }
#                 resource = ResourceInfo(data.slot_id, data.subchannel)
#                 vehicle_info[data.sender_id]['resources'].append(resource)

#         if not vehicle_info:
#             self.best_target_vehicle_id = -1
#             self.best_target_resources = []
#             return

#         best_score = -float('inf')
#         best_vehicle_id = -1
#         for vehicle_id, info in vehicle_info.items():
#             distance = np.linalg.norm(self.position - info['position'])
#             expected_receivers = info.get('expected_receivers', 0)
#             alpha = 2.0
#             beta = 0.2
#             score = alpha * expected_receivers - beta * distance
#             if score > best_score:
#                 best_score = score
#                 best_vehicle_id = vehicle_id

#         self.best_target_vehicle_id = best_vehicle_id
#         if best_vehicle_id != -1:
#             self.best_target_resources = list(set(vehicle_info[best_vehicle_id]['resources'])) # Use set to get unique resources
#             if len(self.best_target_resources) > 2:
#                 self.best_target_resources = self.best_target_resources[-2:]

#     def _update_nearest_vehicle(self):
#         """Update the information and resource usage of the nearest vehicle"""
#         if not self.sensing_data:
#             self.nearest_vehicle_id = -1
#             self.nearest_vehicle_resources = []
#             return

#         vehicle_positions = {}
#         for data in self.sensing_data:
#             if data.sender_id < self.num_vehicles:
#                 if data.sender_id not in vehicle_positions:
#                     vehicle_positions[data.sender_id] = {
#                         'position': data.sender_position,
#                         'timestamp': data.timestamp,
#                         'resources': []
#                     }
#                 resource = ResourceInfo(data.slot_id, data.subchannel)
#                 vehicle_positions[data.sender_id]['resources'].append(resource)

#         min_distance = float('inf')
#         nearest_vehicle_id = -1
#         for vehicle_id, info in vehicle_positions.items():
#             distance = np.linalg.norm(self.position - info['position'])
#             if distance < min_distance:
#                 min_distance = distance
#                 nearest_vehicle_id = vehicle_id
        
#         self.nearest_vehicle_id = nearest_vehicle_id
#         if nearest_vehicle_id != -1:
#             self.nearest_vehicle_resources = list(set(vehicle_positions[nearest_vehicle_id]['resources']))
#             if len(self.nearest_vehicle_resources) > 2:
#                 self.nearest_vehicle_resources = self.nearest_vehicle_resources[-2:]

#     def _update_target_resources(self):
#         """Enhanced target resource selection logic - random selection or use of other vehicles when no data is available"""
#         target_resources = []
#         for data in self.sensing_data:
#             if data.sender_id == self.target_vehicle_id:
#                 target_resources.append((data.timestamp, ResourceInfo(data.slot_id, data.subchannel)))
        
#         if target_resources:
#             target_resources.sort(key=lambda x: x[0], reverse=True)
#             # Get unique resources from the sorted list
#             unique_latest_resources = []
#             seen_resources = set()
#             for _, res in target_resources:
#                 if res not in seen_resources:
#                     unique_latest_resources.append(res)
#                     seen_resources.add(res)
            
#             # Take the 2 most recent unique resources
#             self.targeted_resources = unique_latest_resources[:2]
#         else:
#             # MODIFIED: If target is not sensed, select a random resource pair to attack.
#             # This keeps the attacker active instead of idle.
#             random_slot = random.randint(0, self.sim.num_slots - 1)
#             if self.sim.num_subchannels > 1:
#                 # Select two adjacent subchannels
#                 sc1 = random.randint(0, self.sim.num_subchannels - 2)
#                 sc2 = sc1 + 1
#                 self.targeted_resources = [ResourceInfo(random_slot, sc1), ResourceInfo(random_slot, sc2)]
#             elif self.sim.num_subchannels == 1:
#                 # Handle the edge case of only one subchannel
#                 self.targeted_resources = [ResourceInfo(random_slot, 0)]
#             else:
#                 self.targeted_resources = []


#     def add_sensing_data(self, slot_id, subchannel, pRsvp, sender_id, timestamp, sender_position,expected_receivers):
#         """Add sensing data - extended to support target attacks"""
#         super().add_sensing_data(slot_id, subchannel, pRsvp, sender_id, timestamp, sender_position,expected_receivers)
#         if self.is_target_attacker and sender_id == self.target_vehicle_id:
#             logger.debug(f"Attacker {self.id} sensed target vehicle {self.target_vehicle_id} using resource (slot:{slot_id}, subchannel:{subchannel})")

#     def should_receive_packet(self, sender_position):
#         """The attacker can receive packets within the communication range"""
#         distance = np.linalg.norm(self.position - sender_position)
#         return distance <= self.communication_range

#     def _update_sensing_window(self, current_time):
#         """Update the listening window, remove expired data"""
#         sensing_window_start = current_time - self.sensing_window_duration
#         self.sensing_data = [data for data in self.sensing_data if data.timestamp >= sensing_window_start]


# class ResourcePool:
#     """Manages the sidelink resource pool for V2X communications"""

#     def __init__(self, num_slots=100, num_subchannels=5, subchannel_size=10):
#         self.num_slots = num_slots
#         self.num_subchannels = num_subchannels
#         self.subchannel_size = subchannel_size
#         self.total_rbs = num_subchannels * num_slots


# class V2XRLEnvironment(gym.Env):
#     """V2X攻击优化的RL环境 - 支持优化的SINR碰撞检测"""

#     def __init__(self, kwargs):
#         super(V2XRLEnvironment, self).__init__()

#         self.num_vehicles = kwargs.get('num_vehicles', 20)
#         self.num_attackers = kwargs.get('num_attackers', 1)
#         self.episode_duration = kwargs.get('episode_duration', 20000)
#         self.communication_range = kwargs.get('communication_range', 320.0)
#         self.vehicle_resource_mode = kwargs.get('vehicle_resource_mode', 'Separate')
#         self.attacker_type = kwargs.get('attacker_type', 'RL')
#         # MODIFICATION: Add is_target_attacker flag to fix_attacker_params
#         self.fix_attacker_params = kwargs.get('fix_attacker_params', {'cycle': 20, 'num_subchannels': 2, 'attack_nearest': False, 'attack_most_and_nearest': False})
#         self.num_slots = kwargs.get('num_slots', 100)
#         self.num_subchannels = kwargs.get('num_subchannels', 5)
#         self.tx_power = kwargs.get('tx_power', 23.0)
#         self.attacker_tx_power = 30.0
#         self.noise_power = kwargs.get('noise_power', -95.0)
#         self.action_num = kwargs.get('action_num', 2)
#         self.use_sinr = kwargs.get('use_sinr', True)
#         self.sinr_threshold = kwargs.get('sinr_threshold', 5.0)
#         self.use_simple_interference = kwargs.get('use_simple_interference', False)
#         self.print_detail = False
#         self.use_rl_vehicles = kwargs.get('use_rl_vehicles', False)
        
#         if self.use_rl_vehicles:
#             self.n_agents = self.num_vehicles
#         else:
#             self.n_agents = self.num_attackers
        
#         self.multi_agent_coordination = True if self.n_agents > 1 else False
#         self.num_agents = self.n_agents

#         self.total_resources = self.num_slots * self.num_subchannels
#         self.step_count = 0

#         self.sinr_calculator = SINRCalculator(
#             tx_power=self.tx_power,
#             noise_power=self.noise_power,
#             use_simple_interference=self.use_simple_interference
#         )

#         self.resource_pool = ResourcePool(num_slots=self.num_slots, num_subchannels=self.num_subchannels, subchannel_size=10)
#         self.initial_vehicle_states = None
#         self.initial_attacker_states = None

#         self.recent_collision_queue = deque(maxlen=100)
#         self.recent_collision_rate = 0.0
#         self.last_global_prr = 1.0

#         self.resource_grid = [ [ [] for _ in range(self.num_subchannels)] for _ in range(self.num_slots) ]

#         self.vehicles = []
#         self.attackers = []
#         self._initialize_vehicles()
#         self._initialize_attackers()
        
#         # <<< MODIFICATION START: Define state and action spaces based on fixed requirements >>>
#         if self.num_subchannels <= 1:
#             num_subchannel_choices = 1
#         else:
#             num_subchannel_choices = self.num_subchannels - 1

#         if self.use_rl_vehicles:
#             # --- Configuration for RLVehicle agents ---
#             self.step_duration = self.num_slots  # ms
            
#             # STATE SPACE for RLVehicle: num_slots * num_subchannels
#             obs_dim = self.num_slots * self.num_subchannels
            
#             # ACTION SPACE for RLVehicle: num_slots * (num_subchannels - 1)
#             total_actions = self.num_slots * num_subchannel_choices
            
#             # Additional state space dimensions (action history, time, other agent actions)
#             action_history_dim = 5
#             time_dim = 1
#             other_actions_dim = self.num_vehicles
#             final_obs_dim = obs_dim + action_history_dim + time_dim + other_actions_dim
#         else:
#             # --- Configuration for RLAttacker agents ---
#             self.step_duration = 20  # ms

#             # STATE SPACE for RLAttacker: 20 * num_subchannels
#             obs_dim = 20 * self.num_subchannels

#             # ACTION SPACE for RLAttacker: 20 * (num_subchannels - 1)
#             total_actions = 20 * num_subchannel_choices

#             # Additional state space dimensions (action history, time)
#             action_history_dim = 5
#             time_dim = 1
#             final_obs_dim = obs_dim + action_history_dim + time_dim

#         # Handle the case where there are no RL agents to avoid errors
#         if (self.attacker_type != 'RL' and not self.use_rl_vehicles):
#             final_obs_dim = 1
#             total_actions = 1

#         self.action_space = [spaces.Discrete(total_actions) for _ in range(self.n_agents)]
#         self.observation_space = [
#             spaces.Box(low=-np.inf, high=np.inf, shape=(final_obs_dim,), dtype=np.float32)
#             for _ in range(self.n_agents)
#         ]
#         # <<< MODIFICATION END >>>

#         share_obs_dim = self.num_slots * self.num_subchannels
#         self.share_observation_space = [
#              spaces.Box(low=0.0, high=1.0, shape=(share_obs_dim,), dtype=np.float32)
#              for _ in range(self.n_agents)]
#         # print(len(self.attackers),len(self.vehicles),self.num_agents)
#         # exit(0)
#         self.current_time = 0
#         self.message_status_dict = {}
#         self.message_pool = {}
#         self.sinr_records = []
#         self.resource_transmission_count = 0
#         self.attack_effect_records = []
#         self.verbose_attack_output = True
#         self.reset_stats()

#     def seed(self, seed=None):
#         if seed is not None:
#             np.random.seed(seed)
#             random.seed(seed)

#     def _get_global_state(self):
#         """
#         生成一个真实的全局状态，表示所有正常车辆的资源预选情况。
#         返回一个扁平化的一维向量。
#         """
#         resource_grid = np.zeros((self.num_slots, self.num_subchannels), dtype=np.float32)

#         for vehicle in self.vehicles:
#             if vehicle.current_resources:
#                 for resource in vehicle.current_resources:
#                     if 0 <= resource.slot_id < self.num_slots and 0 <= resource.subchannel < self.num_subchannels:
#                         resource_grid[resource.slot_id, resource.subchannel] = 1.0 # 标记为1

#         return resource_grid.flatten()
    
#     def render(self, mode='human'):
#         pass # Render logic can be added here if needed

#     def reset(self):
#         # print("\nSimulation finished.")
    
#         # # ===============================================================
#         # # ===== AÑADE ESTE BLOQUE PARA IMPRIMIR LAS ESTADÍSTICAS FINALES =====
#         # # ===============================================================
#         # print("\n" + "="*40)
#         # print(" " * 8 + "Final Episode Statistics Dictionary")
#         # print("="*40)

#         # # Llama a la función para obtener el diccionario de estadísticas
#         # final_stats = self.get_episode_stats()

#         # # Imprime el contenido del diccionario clave por clave
#         # print(f"  'step_count': {final_stats['step_count']}")
#         # print(f"  'total_collisions': {final_stats['total_collisions']}")
#         # print(f"  'message_failures': {final_stats['message_failures']}")
#         # print(f"  'prr': {final_stats['prr']:.4f}")
#         # print(f"  'attack_success_rate': {final_stats['attack_success_rate']:.4f}")

#         # # Imprime el diccionario anidado 'vehicle_prrs' de forma legible
#         # print("  'vehicle_prrs': {")
#         # if final_stats['vehicle_prrs']:
#         #     for vehicle_id, prr in final_stats['vehicle_prrs'].items():
#         #         print(f"    Vehicle {vehicle_id}: {prr:.4f},")
#         # else:
#         #     print("    (No data)")
#         # print("  }")
#         self.current_time = 0
#         self.recent_collision_queue.clear()
#         self.recent_collision_rate = 0.0
#         self.message_status_dict = {}
#         self.sinr_records = []
#         self.message_pool = {}
#         self.reset_stats()
#         self.step_count = 0
#         self.attack_effect_records = []
#         self.resource_transmission_count = 0

#         if self.initial_vehicle_states:
#             for i, vehicle in enumerate(self.vehicles):
#                 vehicle.reset()
#                 vehicle.position, vehicle.velocity = self.initial_vehicle_states[i]
#         if self.initial_attacker_states:
#              for i, attacker in enumerate(self.attackers):
#                 attacker.reset()
#                 attacker.position, attacker.velocity = self.initial_attacker_states[i]


#         obs = []
#         agent_list = self.vehicles if self.use_rl_vehicles else self.attackers
#         rl_agent_class = RLVehicle if self.use_rl_vehicles else RLAttacker

#         for agent in agent_list:
#             if isinstance(agent, rl_agent_class):
#                 obs.append(agent.get_state(self.current_time))
#             else:
#                 obs.append(np.zeros(self.observation_space[0].shape, dtype=np.float32))

#         global_state = self._get_global_state()
#         state = [global_state for _ in range(self.n_agents)]

#         total_actions = self.action_space[0].n
#         avail_actions = [np.ones(total_actions, dtype=np.float32) for _ in range(self.n_agents)]
#         return obs, state, avail_actions

    
#     def step(self, actions):
#         """
#         Executes one time step in the environment using a step-based resolution.
#         """
#         self.step_count += 1
        
#         # --- 1. Action and Resource Selection Phase ---
#         attacker_action_resources = defaultdict(list)
        
#         # RL agents select resources based on actions
#         if self.use_rl_vehicles:
#             for i, vehicle in enumerate(self.vehicles):
#                 if isinstance(vehicle, RLVehicle):
#                     resources, _ = vehicle.select_send_resources(actions[i])
#                     vehicle.current_resources = resources
#         else: # RL Attackers
#             for i, attacker in enumerate(self.attackers):
#                 if isinstance(attacker, RLAttacker):
#                     resources, _ = attacker.select_attack_resources(actions[i])
#                     attacker.current_resources = resources
#                     attacker_action_resources[attacker.id] = resources
#                     attacker.sent_resources_count = 0

#         # Non-RL agents select their resources
#         for vehicle in self.vehicles:
#             if isinstance(vehicle, Vehicle):
#                 if self.current_time%self.num_slots==0:
#                     vehicle.has_transmitted = False
#                 if not vehicle.has_transmitted:
#                     vehicle.handle_periodic_resource_reselection(self.current_time)
#                     vehicle.has_transmitted = True

#                     if vehicle.resel_counter > 0:
#                         vehicle.resel_counter -= 1


#                     vehicle.transmission_on_current_resource_counter += 1
#                     vehicle.sent_resources_count = 0
#                 # tx_results = vehicle.send_packet(self.current_time)
#                 # # print("fale",self.current_time)
#                 # if tx_results:
#                 #     print("sender:",tx_results[0][1].slot_id,tx_results[0][1].subchannel ,self.current_time)
        
#         for attacker in self.attackers:
#             if isinstance(attacker, FixAttacker):
#                 attacker.select_and_set_resources(self.current_time)
#                 # tx_results = attacker.send_packet(self.current_time)
#                 # if tx_results:
#                 #     print("**attacker:",tx_results[0][1].slot_id,tx_results[0][1].subchannel,self.current_time)

#         # --- 2. Transmission Aggregation & Outcome Resolution Phase ---
#         resource_usage_map = defaultdict(list)
#         # print(f"\n--- [Step {self.step_count} at Time {self.current_time}ms] Sender Resource Selections ---")
#         all_agents = self.vehicles + self.attackers
#         # for agent in sorted(all_agents, key=lambda x: x.id):
#         #     if agent.current_resources:
#         #         # 将资源对象转换为更易读的字符串列表
#         #         res_str_list = [str(r) for r in agent.current_resources]
#         #         agent_type = "Attacker" if isinstance(agent, (RLAttacker, FixAttacker)) else "Vehicle"
#         #         print(f"  - {agent_type} ID {agent.id}: {', '.join(res_str_list)}")
#         # print("-" * (60 + len(str(self.step_count)) + len(str(self.current_time))))
#         for agent in all_agents:
#             if agent.current_resources:
#                 # Create a single packet for this agent for the whole step
#                 is_attack = isinstance(agent, (RLAttacker, FixAttacker))
                
#                 packet = Packet(agent.id, self.current_time, agent.position, agent.current_packet_id, is_attack=is_attack)
                
#                 for resource in agent.current_resources:
#                      # Only consider resources within the current step's time window for resolution
#                     resource_usage_map[resource].append((agent, packet))
#             agent.transmission_count+=1
#         # Resolve all outcomes for this step
#         step_outcomes = self._resolve_step_transmissions(resource_usage_map)

#         total_collision_caused_due_to_attack = 0
#         if step_outcomes['attacker_impact']:
#             affected_vehicles = set()
#             for vehicle_set in step_outcomes['attacker_impact'].values():
#                 affected_vehicles.update(vehicle_set)
#             total_collision_caused_due_to_attack = len(affected_vehicles)
            
#         # --- 3. Reward Calculation Phase ---
#         rewards = []
#         if self.use_rl_vehicles:
#             current_prr = self._calculate_current_prr()
#             prr_reward = (current_prr - self.last_global_prr) * 20.0
#             self.last_global_prr = current_prr
            
#             successful_tx = sum(1 for success in step_outcomes['rl_vehicle_status'].values() if success)
#             team_reward = 1 * (successful_tx - 15) 

#             for vehicle in self.vehicles:
#                 if isinstance(vehicle, RLVehicle):
#                     was_successful = step_outcomes['rl_vehicle_status'].get(vehicle.id, False)
#                     vehicle.total_sends += 1
#                     if was_successful: vehicle.successful_sends += 1
                    
#                     individual_reward = vehicle.calculate_reward(was_successful)
#                     final_reward = individual_reward + team_reward + prr_reward
#                     rewards.append([final_reward])
#                 else:
#                     rewards.append([0.0])
#         elif self.attacker_type == 'RL':
#             team_reward = 0
#             if self.n_agents > 1 and self.multi_agent_coordination:
#                 # resource_list = [res for res_list in attacker_action_resources.values() for res in res_list]
#                 # num_overlaps = len(resource_list) - len(set(resource_list))
#                 # team_reward -= num_overlaps * 3.0
#                 if len(step_outcomes['attacker_impact']) >= self.n_agents:
#                     team_reward += 3.0

#             for attacker in self.attackers:
#                 if isinstance(attacker, RLAttacker):
#                     affected_vehicles = step_outcomes['attacker_impact'].get(attacker.id, set())
#                     collision_count = len(affected_vehicles)
#                     attacker.attack_success_count += 1 if collision_count > 0 else 0
#                     attacker.attack_packets_sent += len(attacker.current_resources)
                    
#                     individual_reward = attacker.calculate_reward(collision_count, affected_vehicles)
#                     rewards.append([individual_reward + team_reward])
#         else:
#             rewards = [[0.0] for _ in range(self.n_agents)]

#         # --- 4. State Update Phase ---
#         delta_time_sec = self.step_duration / 1000.0
#         for agent in all_agents:
#             agent.move(delta_time_sec)
#         self.current_time += self.step_duration

#         next_obs = []
#         agent_list = self.vehicles if self.use_rl_vehicles else self.attackers
#         rl_agent_class = RLVehicle if self.use_rl_vehicles else RLAttacker
#         for agent in agent_list:
#             if isinstance(agent, rl_agent_class):
#                 next_obs.append(agent.get_state(self.current_time))
#             else:
#                 next_obs.append(np.zeros(self.observation_space[0].shape, dtype=np.float32))

#         next_global_state = self._get_global_state()
#         next_state = [next_global_state for _ in range(self.n_agents)]

#         # --- 5. Return values ---
#         done = self.current_time >= self.episode_duration
#         dones = [done for _ in range(self.n_agents)]
        
#         # <<< MODIFICATION START: Format the info dictionary as requested >>>
#         info = {
#             'collisions_caused_this_step': total_collision_caused_due_to_attack,
#             'total_collisions': self.collision_count,
#             'attack_success_rate': self.total_attack_success / max(1, self.attack_transmission_count),
#             'prr': self._calculate_current_prr(),
#             'message_failures': self.message_failures,
#             'collision_rate': self.collision_count / max(1, self.transmission_count)
#         }
#         # <<< MODIFICATION END >>>
#         infos = [info for _ in range(self.n_agents)]

#         total_actions = self.action_space[0].n
#         avail_actions = [np.ones(total_actions, dtype=np.float32) for _ in range(self.n_agents)]

#         return next_obs, next_state, rewards, dones, infos, avail_actions

#     # def _resolve_step_transmissions(self, resource_usage_map):
#     #     """
#     #     Resolves outcomes for all transmissions in a step based on resource usage.
#     #     This function correctly handles two-block messages for SINR-based PRR calculation
#     #     and identifies collisions for rewards.
#     #     """
#     #     outcomes = {
#     #         'attacker_impact': defaultdict(set),
#     #         'rl_vehicle_status': {v.id: True for v in self.vehicles if isinstance(v, RLVehicle)}
#     #     }
        
#     #     # MODIFICATION: Use a dictionary to track the success of each block for each message-receiver pair.
#     #     # Key: (sender_id, packet_id, receiver_id), Value: list of success booleans [True, False, ...]
#     #     message_reception_status = defaultdict(list)

#     #     # print("--- [Step Collision Report] ---")
#     #     for resource, users in resource_usage_map.items():
#     #         senders = [user[0] for user in users]
            
#     #         # --- Print Collision Information ---
#     #         # if len(senders) > 1:
#     #         #     sender_ids = [s.id for s in senders]
#     #             # print(f"  >>> COLLISION DETECTED at Resource {str(resource)} with Senders: {sender_ids}")

#     #         normal_vehicles = [s for s in senders if not isinstance(s, (RLAttacker, FixAttacker))]
#     #         attackers = [s for s in senders if isinstance(s, (RLAttacker, FixAttacker))]

#     #         # --- Update outcomes for reward calculation ---
#     #         if len(senders) > 1:
#     #             for sender in senders:
#     #                 if isinstance(sender, RLVehicle):
#     #                     outcomes['rl_vehicle_status'][sender.id] = False
            
#     #         if attackers:
#     #             if normal_vehicles:
#     #                 self.total_attack_success += 1
#     #                 for attacker in attackers:
#     #                     for vehicle in normal_vehicles:
#     #                         outcomes['attacker_impact'][attacker.id].add(vehicle.id)
#     #             self.attack_transmission_count +=1

#     #         # --- SINR Calculation: Gather success status for each resource block ---
#     #         for sender, packet in users:
#     #             interferers = [u[0] for u in users if u[0].id != sender.id]
#     #             interferer_positions = [i.position for i in interferers]
#     #             has_attack_in_interferers = any(isinstance(i, (RLAttacker, FixAttacker)) for i in interferers)
                
#     #             for receiver in self.vehicles:
#     #                 if receiver.id == sender.id:
#     #                     continue
                    
#     #                 if receiver.should_receive_packet(sender.position):
                        
#     #                     sinr = self.sinr_calculator.calculate_sinr_optimized(
#     #                         receiver.position, sender.position, interferer_positions, 
#     #                         self.attacker_tx_power, has_attack_in_interferers
#     #                     )
#     #                     success = sinr >= self.sinr_threshold

#     #                     # Update receiver's sensing data regardless of message type
#     #                     receiver.receive_packet(packet, resource, success)

#     #                     # Only track reception status for non-attack packets for PRR calculation
#     #                     if not packet.is_attack:
#     #                         msg_key = (sender.id, packet.packet_id, receiver.id)
#     #                         message_reception_status[msg_key].append(success)
#     #                     if not isinstance(sender, (RLAttacker, FixAttacker)):   
#     #                         self.transmission_count +=1
#     #                         if not success:
#     #                             self.collision_count += 1
#     #                             sender.collisions+=1
#     #             for receiver in self.attackers:
#     #                 if receiver.should_receive_packet(sender.position):
#     #                                 pRsvp = 100 if not isinstance(sender, (RLAttacker, FixAttacker)) else 20
#     #                                 receiver.add_sensing_data(
#     #                                     resource.slot_id,
#     #                                     resource.subchannel,
#     #                                     pRsvp,
#     #                                     sender.id,
#     #                                     packet.timestamp,
#     #                                     packet.position,
#     #                                     packet.expected_receivers
#     #                                 )
#     #     # --- Final Tally: Process the tracked message statuses to update counters correctly ---
#     #     # A sender's message is defined by a unique packet_id. We need to avoid double-counting expected packets.
#     #     processed_expected_packets = set() # (sender_id, packet_id)

#     #     for msg_key, success_list in message_reception_status.items():
#     #         sender_id, packet_id, _ = msg_key
#     #         sender = self.vehicles[sender_id] # Assuming vehicle IDs are 0 to num_vehicles-1

#     #         # Increment expected packets only once per transmitted message
#     #         if (sender_id, packet_id) not in processed_expected_packets:
#     #             # A message is expected to be received by all vehicles in range.
#     #             # Here we are processing on a per-receiver basis, which is correct for PRR.
#     #             sender.expected_receptions += 1
#     #             self.total_expected_packets += 1
#     #             # Mark this sender's packet as processed for expected counts to avoid recounting
#     #             # Note: This logic assumes one receiver is processed at a time. A better way is to count all receivers in range once.
#     #             # However, this per-receiver counting is standard for PRR calculation.
            
#     #         # A message is successful if both its blocks were received successfully.
#     #         # It's a failure if it's incomplete (less than 2 blocks) or if any block failed.
#     #         is_complete = len(success_list) == 2
#     #         is_successful = all(success_list)

#     #         if is_complete:
#     #             if is_successful:
#     #                 sender.successful_receptions += 1
#     #                 self.total_received_packets += 1
#     #             else:
#     #                 self.message_failures += 1
#     #             # We only count a message failure once, even if it's for the same sender-receiver pair
#     #             # The current loop structure naturally handles this.
#     #             # Failures are tallied at the end
            
#     #     # <<< MODIFICATION START: Per-attacker success counting moved here >>>
#     #     total_vehicles_impacted_by_attack = 0
#     #     # Iterate through the outcomes to update each attacker's individual counter
#     #     for attacker_id, impacted_vehicle_ids in outcomes['attacker_impact'].items():
#     #         collision_count = len(impacted_vehicle_ids)
#     #         if collision_count > 0:
#     #             # Find the attacker object by its ID to update its counter
#     #             # Attacker IDs start after vehicle IDs, so their index is id - num_vehicles
#     #             attacker_index = attacker_id - self.num_vehicles
#     #             if 0 <= attacker_index < len(self.attackers):
#     #                 attacker = self.attackers[attacker_index]
#     #                 attacker.attack_success_count += collision_count
#     #         total_vehicles_impacted_by_attack += collision_count
        
#     #     # Update the global counter for the info dict
#     #     # self.total_attack_success += total_vehicles_impacted_by_attack
#     #     outcomes['collisions_caused_this_step'] = total_vehicles_impacted_by_attack
#     #     # Now, calculate total message failures based on the difference
#     #     # between expected receptions and successful receptions.
#     #     # self.message_failures = self.total_expected_packets - self.total_received_packets
#     #     # Also update individual sender collision stats

#     #     # Remove the old, incorrect division-by-two logic.
        
#     #     return outcomes
    
#     def _resolve_step_transmissions(self, resource_usage_map):
#         """
#         Resolves outcomes for all transmissions in a step based on resource usage.
#         This function correctly handles two-block messages for SINR-based PRR calculation
#         and identifies collisions for rewards.
#         """
#         outcomes = {
#             'attacker_impact': defaultdict(set),
#             'rl_vehicle_status': {v.id: True for v in self.vehicles if isinstance(v, RLVehicle)}
#         }
        
#         # MODIFICATION: Use a dictionary to track the success of each block for each message-receiver pair.
#         # Key: (sender_id, packet_id, receiver_id), Value: list of success booleans [True, False, ...]
#         message_reception_status = defaultdict(list)

#         # ADDED: Main title for the report section
#         # print("\n--- [SINR Resolution Report] ---")
#         for resource, users in resource_usage_map.items():
#             senders = [user[0] for user in users]
#             sender_ids = [s.id for s in senders]  # ADDED: Get sender IDs

#             # ADDED: Print resource block and which vehicles are using it
#             # print(f"\nResource Block: {str(resource)}")
#             # print(f"  Users (Senders): {sender_ids}")
            
#             normal_vehicles = [s for s in senders if not isinstance(s, (RLAttacker, FixAttacker))]
#             attackers = [s for s in senders if isinstance(s, (RLAttacker, FixAttacker))]
            
#             # if attackers:
#             #     print(f"  'slotid': {resource.slot_id} 'subchannel': {resource.subchannel}")
#             #     for sender in senders:
#             #         print(f"  'sender': {sender.id}")
#             #     for attacker in attackers:
#             #         print(f"  'sender': {attacker.id}")
#             # --- Update outcomes for reward calculation ---
#             if len(senders) > 1:
#                 for sender in senders:
#                     if isinstance(sender, RLVehicle):
#                         outcomes['rl_vehicle_status'][sender.id] = False
#                     if isinstance(sender,Vehicle):
#                         sender.current_resources_collision = True
            
#             if attackers:
#                 if normal_vehicles:
#                     self.total_attack_success += 1
#                     for attacker in attackers:
#                         for vehicle in normal_vehicles:
#                             outcomes['attacker_impact'][attacker.id].add(vehicle.id)
#                 self.attack_transmission_count +=1

#             # --- SINR Calculation: Gather success status for each resource block ---
#             for sender, packet in users:
#                 interferers = [u[0] for u in users if u[0].id != sender.id]
#                 interferer_positions = [i.position for i in interferers]
#                 has_attack_in_interferers = any(isinstance(i, (RLAttacker, FixAttacker)) for i in interferers)
                
#                 # ADDED: Print a sub-header for the current sender being analyzed
#                 # print(f"  - Analyzing Sender ID: {sender.id}")
                
#                 for receiver in self.vehicles:
#                     if receiver.id == sender.id:
#                         continue
                    
#                     if receiver.should_receive_packet(sender.position):
                        
#                         # ADDED: Calculate sender-receiver distance
#                         sender_dist = np.linalg.norm(receiver.position - sender.position)

#                         # ADDED: Calculate interferer distances and format them for printing
#                         interferer_dists_str = []
#                         for interferer in interferers:
#                             dist = np.linalg.norm(receiver.position - interferer.position)
#                             interferer_dists_str.append(f"ID {interferer.id}: {dist:.1f}m")
#                         interferer_info_str = f"Interferers: [{', '.join(interferer_dists_str)}]" if interferers else "Interferers: [None]"

#                         sinr = self.sinr_calculator.calculate_sinr_optimized(
#                             receiver.position, sender.position, interferer_positions, 
#                             self.attacker_tx_power, has_attack_in_interferers
#                         )
#                         success = sinr >= self.sinr_threshold
                        
#                         status_str = "Success" if success else "Fail"
                        
#                         # # MODIFIED: Update the print statement to include distances
#                         # print(f"    - Rx ID: {receiver.id} | Tx Dist: {sender_dist:.1f}m | SINR: {sinr:.2f} dB | Status: {status_str}")
#                         # print(f"      L {interferer_info_str}")


#                         # Update receiver's sensing data regardless of message type
#                         receiver.receive_packet(packet, resource, success)

#                         # Only track reception status for non-attack packets for PRR calculation
#                         if not packet.is_attack:
#                             msg_key = (sender.id, packet.packet_id, receiver.id)
#                             message_reception_status[msg_key].append(success)
                            
#                         if not isinstance(sender, (RLAttacker, FixAttacker)):   
#                             self.transmission_count +=1
#                             # sender.total_expected_receptions+=1
#                             if not success:
#                                 self.collision_count += 1
#                                 sender.collisions+=1
#                             # else:
#                                 # sender.total_successful_receptions +=1 
#                 for receiver in self.attackers:
#                     if receiver.should_receive_packet(sender.position):
#                                     pRsvp = 100 if not isinstance(sender, (RLAttacker, FixAttacker)) else 20
#                                     receiver.add_sensing_data(
#                                         resource.slot_id,
#                                         resource.subchannel,
#                                         pRsvp,
#                                         sender.id,
#                                         packet.timestamp,
#                                         packet.position,
#                                         packet.expected_receivers
#                                     )
        
#         # print("--- [End SINR Resolution Report] ---\n")  # ADDED: Footer for the report section

#         # --- Final Tally: Process the tracked message statuses to update counters correctly ---
#         # A sender's message is defined by a unique packet_id. We need to avoid double-counting expected packets.
#         processed_expected_packets = set() # (sender_id, packet_id)

#         for msg_key, success_list in message_reception_status.items():
#             # print(f"Message Key: {msg_key}, Success List: {success_list}")
#             sender_id, packet_id, _ = msg_key
#             sender = self.vehicles[sender_id] # Assuming vehicle IDs are 0 to num_vehicles-1

#             # Increment expected packets only once per transmitted message
#             if (sender_id, packet_id) not in processed_expected_packets:
#                 # A message is expected to be received by all vehicles in range.
#                 # Here we are processing on a per-receiver basis, which is correct for PRR.
#                 sender.expected_receptions += 1
#                 self.total_expected_packets += 1
#                 # Mark this sender's packet as processed for expected counts to avoid recounting
#                 # Note: This logic assumes one receiver is processed at a time. A better way is to count all receivers in range once.
#                 # However, this per-receiver counting is standard for PRR calculation.
            
#             # A message is successful if both its blocks were received successfully.
#             # It's a failure if it's incomplete (less than 2 blocks) or if any block failed.
#             is_complete = len(success_list) == 2
#             is_successful = all(success_list)

#             if is_complete:
#                 sender.total_expected_receptions+=1
#                 if is_successful:
#                     sender.total_successful_receptions += 1
#                     self.total_received_packets += 1
#                 else:
#                     self.message_failures += 1
#                 # We only count a message failure once, even if it's for the same sender-receiver pair
#                 # The current loop structure naturally handles this.
#                 # Failures are tallied at the end
            
#         # <<< MODIFICATION START: Per-attacker success counting moved here >>>
#         total_vehicles_impacted_by_attack = 0
#         # Iterate through the outcomes to update each attacker's individual counter
#         for attacker_id, impacted_vehicle_ids in outcomes['attacker_impact'].items():
#             collision_count = len(impacted_vehicle_ids)
#             if collision_count > 0:
#                 # Find the attacker object by its ID to update its counter
#                 # Attacker IDs start after vehicle IDs, so their index is id - num_vehicles
#                 attacker_index = attacker_id - self.num_vehicles
#                 if 0 <= attacker_index < len(self.attackers):
#                     attacker = self.attackers[attacker_index]
#                     attacker.attack_success_count += collision_count
#             total_vehicles_impacted_by_attack += collision_count
        
#         # Update the global counter for the info dict
#         # self.total_attack_success += total_vehicles_impacted_by_attack
#         outcomes['collisions_caused_this_step'] = total_vehicles_impacted_by_attack
#         # Now, calculate total message failures based on the difference
#         # between expected receptions and successful receptions.
#         # self.message_failures = self.total_expected_packets - self.total_received_packets
#         # Also update individual sender collision stats

#         # Remove the old, incorrect division-by-two logic.
        
#         return outcomes
    
#     def _distance(self, pos1, pos2):
#         """计算两点之间的欧几里得距离"""
#         return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

#     def _calculate_current_prr(self):
#         """计算当前分组接收率(PRR)"""
#         if self.total_expected_packets > 0:
#             return self.total_received_packets / self.total_expected_packets
#         # Return 1.0 if no packets were expected, to avoid penalizing inaction
#         return 1.0

#     def get_vehicle_prrs(self):
#         """获取所有车辆的个人PRR"""
#         vehicle_prrs = {}
#         for vehicle in self.vehicles:
#             vehicle_prrs[vehicle.id] = vehicle.calculate_prr()
#         return vehicle_prrs

#     def get_episode_stats(self):
#         """获取当前轮的统计信息"""
#         vehicle_prrs = self.get_vehicle_prrs()
#         return {
#             'total_collisions': self.collision_count,
#             'prr': self._calculate_current_prr(),
#             'attack_success_rate': self.total_attack_success / max(1, self.attack_transmission_count),
#             'message_failures': self.message_failures,
#             'vehicle_prrs': vehicle_prrs,
#             'step_count': self.step_count
#         }

#     def reset_stats(self):
#         """重置所有统计"""
#         self.collision_count = 0
#         self.transmission_count = 0
#         self.total_expected_packets = 0
#         self.total_received_packets = 0
#         self.attack_transmission_count = 0
#         self.total_attack_success = 0
#         self.message_failures = 0
#         self.resource_block_attacks = 0
#         self.resource_block_collisions = 0
#         self.last_global_prr = 1.0

#     def _initialize_vehicles(self):
#         """初始化车辆位置和速度，并保存初始状态"""
#         lane1_y = 5.0
#         lane2_y = 10.0
#         highway_length = 1000.0
#         self.vehicles = []
#         vehicle_states = []
#         for i in range(self.num_vehicles):
#             lane_y = lane1_y if i % 2 == 0 else lane2_y
#             pos_x = random.uniform(0, highway_length)
#             position = np.array([pos_x, lane_y])
#             velocity = np.array([10.0 if lane_y == lane1_y else -10.0, 0.0])
#             if self.use_rl_vehicles:
#                 vehicle = RLVehicle(i, position.copy(), velocity.copy(), self, self.action_num)
#             else:
#                 vehicle = Vehicle(i, position.copy(), velocity.copy(), self, self.vehicle_resource_mode)
#             self.vehicles.append(vehicle)
#             vehicle_states.append((position.copy(), velocity.copy()))
#         self.initial_vehicle_states = vehicle_states

#     def _initialize_attackers(self):
#         """初始化攻击者并保存初始状态，根据攻击者数量均匀分配其初始位置。"""
#         highway_length = 1000.0
#         self.attackers = []
#         attacker_states = []
        
#         num_attackers = self.num_attackers
#         for i in range(num_attackers):
#             attacker_id = self.num_vehicles + i
#             pos_x = (i + 1) * highway_length / (num_attackers + 1)
#             position = np.array([pos_x, 7.5]) 
#             velocity = np.array([0.0, 0.0])

#             if self.attacker_type == 'RL':
#                 attacker = RLAttacker(attacker_id, position.copy(), velocity.copy(), self, self.action_num)
#             else: # Fixed Attacker
#                 attacker = FixAttacker(
#                     attacker_id,
#                     position.copy(),
#                     velocity.copy(),
#                     self,
#                     attack_cycle=self.fix_attacker_params.get('cycle', 20),
#                     num_subchannels=self.fix_attacker_params.get('num_subchannels', 2),
#                     attack_nearest=self.fix_attacker_params.get('attack_nearest', False),
#                     attack_most_and_nearest=self.fix_attacker_params.get('attack_most_and_nearest', False)
#                 )
#             self.attackers.append(attacker)
#             attacker_states.append((position.copy(), velocity.copy()))
#         self.initial_attacker_states = attacker_states

#     def close(self):
#         """关闭环境，清理资源"""
#         if hasattr(self, 'fig'):
#             plt.ioff()
#             plt.close()
#         if hasattr(self, 'sensing_fig'):
#             plt.close(self.sensing_fig)


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
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('V2X-RL-Environment-Enhanced-SINR')
np.random.seed(42)
# Global attack mode switch
TARGETED_ATTACK_MODE = True

class Packet:
    """Represents a V2X packet for transmission"""

    def __init__(self, sender_id, timestamp, position, packet_id, size=190, is_attack=False):
        self.sender_id = sender_id
        self.timestamp = timestamp
        self.position = position
        self.size = size
        self.is_attack = is_attack
        self.packet_id = packet_id
        self.expected_receivers = 0

class SensingData:
    """Represents sensing data"""
    def __init__(self, slot_id, subchannel, pRsvp, sender_id, timestamp, sender_position,expected_receivers):
        self.slot_id = slot_id
        self.subchannel = subchannel
        self.pRsvp = pRsvp
        self.sender_id = sender_id
        self.timestamp = timestamp
        self.sender_position = sender_position  # Added sender position information
        self.expected_receivers = expected_receivers  # Added number of expected receivers

class ResourceInfo:
    """Represents a resource block (slot + subchannel)"""
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
    """SINR Calculator - Encapsulates SINR calculation logic"""

    def __init__(self, tx_power=23.0, noise_power=-95.0, use_simple_interference=False,attacker_interference_boost=5.0,normal_interference_boost=4.0):
        self.tx_power = tx_power  # Transmit power (dBm)
        self.noise_power = noise_power  # Noise power (dBm)
        self.use_simple_interference = use_simple_interference  # Whether to use simplified interference calculation
        self.attacker_interference_boost = attacker_interference_boost  # Attacker interference enhancement
        self.normal_interference_boost = normal_interference_boost
    def calculate_path_loss(self, distance):
        """
        Optimized path loss model - ensures that a single sender at a long distance can also be received successfully
        """
        if distance < 1:
            distance = 1

        # Corrected path loss model to reduce long-distance loss
        if distance <= 50:
            # Use standard model for short distances
            return 32.45 + 20 * np.log10(distance) + 20 * np.log10(5.9)
        elif distance <= 200:
            # Medium distance optimization
            return 32.45 + 20 * np.log10(50) + 20 * np.log10(5.9) + 15 * np.log10(distance / 50)
        else:
            # Further optimization for long distances to ensure reception at 300m+
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
            # Regular vehicle interference is also weighted
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
        Simplified SINR calculation: based only on the number of senders and path loss
        """
        # Calculate target signal received power
        tx_power = tx_power if tx_power is not None else self.tx_power
        distance = np.linalg.norm(receiver_pos - sender_pos)
        path_loss = self.calculate_path_loss(distance)
        rx_power = self.tx_power - path_loss  # dBm

        # Simplified interference calculation: based on the number of interferers
        if num_interferers == 0:
            # No interference - only noise
            interference_power_db = self.noise_power
        else:
            # Interference - assume average interference power
            avg_interference_power = self.tx_power - 60  # Assume average 60dB path loss
            total_interference_mw = num_interferers * (10 ** (avg_interference_power / 10))
            noise_mw = 10 ** (self.noise_power / 10)
            interference_power_db = 10 * np.log10(total_interference_mw + noise_mw)

        # Calculate SINR
        signal_mw = 10 ** (rx_power / 10)
        interference_mw = 10 ** (interference_power_db / 10)
        sinr_linear = signal_mw / interference_mw
        sinr_db = 10 * np.log10(sinr_linear) if sinr_linear > 0 else -100

        return sinr_db

class Message:
    """Represents a V2X message from sender to receiver"""
    def __init__(self, sender_id, receiver_id, packet_id):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.packet_id = packet_id
        self.resources_received = 0  # Number of resource blocks received
        self.success = True  # Initially assume success, becomes False upon failure
        self.completed = False  # Whether it is completed (both resource blocks are processed)

    def record_reception(self, success):
        """Record resource block reception status"""
        self.resources_received += 1
        if not success:
            self.success = False  # Once failed, it is irreversible

    def is_completed(self):
        """Check if the message is complete (both resource blocks have been processed)"""
        return self.resources_received >= 2

class Vehicle:
    """Represents a vehicle with V2X capabilities"""

    def __init__(self, vehicle_id, initial_position, initial_velocity, sim, resource_selection_mode='Combine',enable_collision_triggered_reselection=False):
        self.id = vehicle_id
        self.position = initial_position
        self.velocity = initial_velocity
        self.sim = sim
        self.resource_selection_mode = resource_selection_mode

        # Get resource pool parameters
        self.num_slots = sim.resource_pool.num_slots
        self.num_subchannels = sim.resource_pool.num_subchannels

        # Resource selection parameters
        self.resel_counter = 0
        self.prob_resource_keep = random.uniform(0.2, 0.8)
        self.current_resources = None
        self.sensing_data = []
        self.next_transmission_time = 0
        self.sent_resources_count = 0
        self.current_packet_id = 0
        # Sensing window parameters
        self.sensing_window_duration = 1000
        self.has_transmitted = False
        self.total_faild = 0
        self.enable_collision_triggered_reselection = enable_collision_triggered_reselection
        self.current_resources_collision = False  # Record whether the current resource is in conflict
        # Initialize statistics
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_received_succeed = 0
        self.collisions = 0
        self.successful_transmissions = 0
        self.prr = 0.0
        self.expected_receptions = 0
        self.total_successful_receptions = 0
        self.total_expected_receptions = 0
        self.transmission_count = 0
        self.transmission_on_current_resource_counter = 0

    def reset(self):
        """Reset vehicle state"""
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
        self.total_successful_receptions = 0
        self.total_expected_receptions = 0
        self.total_faild = 0
        self.transmission_count = 0
        self.transmission_on_current_resource_counter = 0

    def move(self, delta_time):
        """Update vehicle position based on velocity and time delta"""
        self.position = self.position + self.velocity * delta_time

        # Handle boundary conditions (reflection)
        if self.position[0] >= 1000:
            self.position[0] = 1000 - (self.position[0] - 1000)
            self.velocity = -self.velocity
            self.position[1] = 10.0
        if self.position[0] <= 0:
            self.position[0] = -self.position[0]
            self.velocity = -self.velocity
            self.position[1] = 5.0

    def get_sensed_resource_occupancy(self):
        """Get the resource occupancy status matrix in the listening window"""
        occupancy = np.zeros((self.num_slots, self.num_subchannels), dtype=int)

        for data in self.sensing_data:
            slot = data.slot_id % self.num_slots
            if 0 <= slot < self.num_slots and 0 <= data.subchannel < self.num_subchannels:
                occupancy[slot, data.subchannel] = 1
            # print(occupancy)
        return occupancy

    def select_future_resource(self, current_time):
        """Select future resources - select resource blocks based on the mode"""
        self._update_sensing_window(current_time)
        selection_window = self._create_selection_window(current_time)

        # Create a set of occupied resources
        occupied_resources = set()
        for data in self.sensing_data:
            resource_key = (data.slot_id, data.subchannel)
            occupied_resources.add(resource_key)

        # Select resources according to the mode
        if self.resource_selection_mode == 'Combine':
            selected_resources = self._select_combined_resources(selection_window, occupied_resources)
        else:  # Separate mode
            selected_resources = self._select_separate_resources(selection_window, occupied_resources)

        self.resel_counter = random.randint(5, 15)
        # print(selected_resources[0].slot_id,selected_resources[0].subchannel)
        return selected_resources

    def _select_separate_resources(self, selection_window, occupied_resources):
        """Separate mode: select two independent resource blocks"""
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
        """Combine mode: select two adjacent subchannels in the same slot"""
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
        """Create a selection window (T1=4 to T2=100)"""
        selection_window = []
        current_slot = current_time % self.num_slots
        start_slot = (current_slot + 0) % self.num_slots
        end_slot = (current_slot + self.num_slots-1) % self.num_slots

        if start_slot < end_slot:
            slots = range(start_slot, end_slot)
        else:
            slots = list(range(start_slot, self.num_slots)) + list(range(0, end_slot))

        for slot in slots:
            for subchannel in range(self.num_subchannels):
                selection_window.append(ResourceInfo(slot, subchannel))

        return selection_window

    def _update_sensing_window(self, current_time):
        """Update the sensing window by removing old entries"""
        sensing_window_start = current_time - self.sensing_window_duration
        self.sensing_data = [data for data in self.sensing_data
                            if data.timestamp >= sensing_window_start]

    def add_sensing_data(self, slot_id, subchannel, pRsvp, sender_id, timestamp, sender_position,expected_receivers):
        """Add sensing data"""
        sensing_data = SensingData(
            slot_id=slot_id,
            subchannel=subchannel,
            pRsvp=pRsvp,
            sender_id=sender_id,
            timestamp=timestamp,
            sender_position=sender_position,  # Added sender position
            expected_receivers =expected_receivers
        )
        self.sensing_data.append(sensing_data)

    def handle_periodic_resource_reselection(self, current_time):
        """Maneja la nueva selección de recursos basándose en el modo del vehículo."""

        # Esta rama es para el modo donde las colisiones NO provocan una nueva selección de recursos.
        if not self.enable_collision_triggered_reselection:
            should_reselect = False

            # Condición 1: Se deben seleccionar recursos si no existen.
            if self.current_resources is None:
                should_reselect = True

            # Condición 2: Nueva selección forzada después de 15 mensajes en el mismo recurso.
            # Nota: self.transmission_on_current_resource_counter debe ser incrementado después de cada transmisión completa.
            elif self.transmission_on_current_resource_counter >= 15:
                should_reselect = True

            # Condición 3: Si el contador inicial ha expirado, se vuelve a seleccionar según la probabilidad.
            # Nota: self.resel_counter debe ser decrementado después de cada transmisión durante su período de gracia.
            elif self.resel_counter <= 0:
                # La probabilidad de volver a seleccionar está entre 0.2 y 0.8.
                prob_to_reselect = random.uniform(0.2, 0.8)
                if random.random() < prob_to_reselect:
                    should_reselect = True

            # Si se cumple alguna condición, se realiza la nueva selección y se reinician todos los contadores.
            if should_reselect:
                self.current_resources = self.select_future_resource(current_time)
                # Se establece una nueva cuenta atrás inicial (período de gracia) de 5 a 15.
                self.resel_counter = random.randint(5, 15)
                # Se reinicia el contador de mensajes enviados en el recurso actual.
                self.transmission_on_current_resource_counter = 0
                self.sent_resources_count = 0
                # self.current_packet_id += 1 # Se genera un nuevo paquete para los nuevos recursos

        else:
            # Esta rama se activa cuando las colisiones SÍ provocan una nueva selección de recursos.
            should_reselect = False

            # Condición 1: Se deben seleccionar recursos si no existen.
            if self.current_resources is None:
                should_reselect = True

            # Condición 2: Nueva selección forzada si se detectó una colisión en los recursos actuales.
            elif self.current_resources_collision:
                should_reselect = True

            # Condición 3: Nueva selección forzada después de 15 mensajes en el mismo recurso.
            elif self.transmission_on_current_resource_counter >= 15:
                should_reselect = True

            # Condición 4: Si el contador inicial ha expirado, se vuelve a seleccionar según la probabilidad.
            elif self.resel_counter <= 0:
                # La probabilidad de volver a seleccionar está entre 0.2 y 0.8.
                prob_to_reselect = random.uniform(0.2, 0.8)
                if random.random() < prob_to_reselect:
                    should_reselect = True

            # Si se cumple alguna condición, se realiza la nueva selección y se reinician todos los contadores.
            if should_reselect:
                self.current_resources = self.select_future_resource(current_time)
                # Se restablece el indicador de colisión ya que estamos eligiendo nuevos recursos.
                self.current_resources_collision = False
                # Se establece una nueva cuenta atrás inicial (período de gracia) de 5 a 15.
                self.resel_counter = random.randint(5, 15)
                # Se reinicia el contador de mensajes enviados en el recurso actual.
                self.transmission_on_current_resource_counter = 0
                self.sent_resources_count = 0
                # self.current_packet_id += 1


    def send_packet(self, current_time):
        """Send a packet using the selected resources (now using two resource blocks)"""
        if self.current_resources is None:
            return None

        current_slot = current_time % self.num_slots
        # print(current_slot)
        resources_to_send = []
        for resource in self.current_resources:
            if resource.slot_id == current_slot:
                print("fale")
                resources_to_send.append(resource)

        if not resources_to_send:
            return None

        packet = Packet(self.id, current_time, self.position, self.current_packet_id)
        packet.expected_receivers = self._calculate_expected_receivers()
        # self.expected_receivers = 
        self.expected_receivers = packet.expected_receivers
        # for resource in resources_to_send:
        #     print(self.id, "sending packet", packet.packet_id, "to resource", resource.slot_id," ",resource.subchannel)
        transmissions = []
        for resource in resources_to_send:
            transmissions.append((packet, resource))

        self.sent_resources_count += len(resources_to_send)
        self.packets_sent += len(resources_to_send)
        if self.sent_resources_count >= 2:
            self.has_transmitted = True
            
            # --- LÓGICA AÑADIDA/MODIFICADA ---
            # Si el período de gracia no ha terminado, decrémentalo
            if self.resel_counter > 0:
                self.resel_counter -= 1
            
            # Incrementa el contador de transmisiones en el recurso actual
            self.transmission_on_current_resource_counter += 1
            # --- FIN DE LA LÓGICA AÑADIDA/MODIFICADA ---
            self.current_packet_id += 1
            self.sent_resources_count = 0

            # self.sent_resources_count = 0
        # if self.sent_resources_count >= 2:
        #     self.has_transmitted = True
        #     self.resel_counter -= 1
            

        return transmissions

    def _calculate_expected_receivers(self):
        """Calculate the number of vehicles that can receive this packet at the current time (within communication range)"""
        count = 0
        for vehicle in self.sim.vehicles:
            if vehicle.id != self.id and vehicle.should_receive_packet(self.position):
                count += 1
        return count


    def calculate_prr(self):
        """Calculate personal PRR"""
        if self.total_expected_receptions > 0:
            return self.total_successful_receptions/self.total_expected_receptions
        return 0.0

    def receive_packet(self, packet, resource, success):
        """Process received packets - modified to always add sensing data"""
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
            packet.position,  # Added sender position
            packet.expected_receivers
        )
        if not packet.is_attack:
            self.packets_received += 1
        if not packet.is_attack and success:
            self.packets_received_succeed += 1
            return True
        return False

    def should_receive_packet(self, sender_position):
        """Determine whether this vehicle should receive a packet from the sender"""
        distance = np.linalg.norm(self.position - sender_position)
        return distance <= self.sim.communication_range

class RLVehicle:
    """RL-based sender"""

    def __init__(self, attacker_id, initial_position, initial_velocity, sim, action_num=2):
        self.id = attacker_id
        self.position = initial_position
        self.velocity = initial_velocity
        self.sim = sim
        self.last_collison = 0
        self.communication_range = 320
        self.num_slots = sim.resource_pool.num_slots
        self.num_subchannels = sim.resource_pool.num_subchannels
        self.action_num = action_num  # Number of resource blocks selected by each agent
        self.total_resources = self.num_slots * self.num_subchannels  # Total number of resource blocks
        
        self.next_transmission_time = 0
        self.transmission_cycle = self.num_slots
        self.current_resources = None
        self.normal_packets_sent = 0
        self.normal_success_count = 0
        self.collisions_caused = 0
        self.target_slot = -1
        self.sensing_data = []
        self.sensing_window_duration = self.num_slots
        self.last_action = None
        self.last_reward = 0
        self.resources_changed = False
        self.action_history = deque(maxlen=5)
        self.last_action_indices = np.zeros(5, dtype=np.int32)  # Store the indices of the last 5 actions
        self.send_success_this_step = False  # Added: flag whether this transmission caused a collision
        self.target_vehicle_id = 0
        self.target_vehicle_resources = []
        self.target_vehicle_tracking_time = 0
        self.targeted_resources = []
        self.last_prr = 1.0
        self.last_send_count = 0
        self.last_fail_counts = 0
        self.packets_received =0
        self.packets_received_succeed =0
        self.total_successful_receptions = 0
        self.total_expected_receptions = 0
        # Added: save the last selected resources
        self.last_selected_resources = set()
        self.last_action_index = -1  # Store the resource index of the previous action
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_received_succeed = 0
        self.collisions = 0
        self.successful_transmissions = 0
        self.prr = 0.0
        self.expected_receptions = 0
        self.successful_receptions = 0
        self.total_faild = 0
        self.successful_sends = 0
        self.total_sends = 0
        self.last_success_rate = 1.0
        self.current_packet_id = 0
        self.transmission_count = 0

    def reset(self):
        """Reset attacker state"""
        self.last_collison = 0
        self.next_transmission_time = 0
        self.current_resources = None
        self.normal_packets_sent = 0
        self.normal_success_count = 0
        self.collisions_caused = 0
        self.target_slot = -1
        self.resources_changed = False
        self.sensing_data = []
        self.last_action = None
        self.last_reward = 0

        self.last_prr = 1.0
        self.last_send_count = 0
        self.last_fail_counts = 0
        self.total_successful_receptions = 0
        self.total_expected_receptions = 0
        self.action_history.clear()
        self.send_success_this_step = False  # Added: flag whether this transmission caused a collision
        self.target_vehicle_id = 0
        self.target_vehicle_resources = []
        self.target_vehicle_tracking_time = 0
        self.targeted_resources = []
        self.last_action_index = -1
        self.packets_received =0
        self.packets_received_succeed =0
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_received_succeed = 0
        self.collisions = 0
        self.successful_transmissions = 0
        self.prr = 0.0
        self.expected_receptions = 0
        self.successful_receptions = 0
        self.last_selected_resources = set()
        # self.last_action_indices = np.zeros(self.action_num, dtype=np.int32)
        self.last_action_indices = np.zeros(5, dtype=np.int32)  # Store the indices of the last 5 actions
        self.total_faild = 0
        self.successful_sends = 0
        self.total_sends = 0
        self.last_success_rate = 1.0
        self.current_packet_id = 0
        self.transmission_count = 0

    def move(self, delta_time):
        """Update vehicle position based on velocity and time delta"""
        self.position = self.position + self.velocity * delta_time

        # Handle boundary conditions (reflection)
        if self.position[0] >= 1000:
            self.position[0] = 1000 - (self.position[0] - 1000)
            self.velocity = -self.velocity
            self.position[1] = 10.0
        if self.position[0] <= 0:
            self.position[0] = -self.position[0]
            self.velocity = -self.velocity
            self.position[1] = 5.0

    def calculate_prr(self):
        """Calculate personal PRR"""
        if self.total_expected_receptions > 0:
            return self.total_successful_receptions/self.total_expected_receptions
        return 0.0
    
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
        """
        MODIFICATION: Simplifies the state representation.
        A resource is marked as 1 if any transmission is sensed, 0 otherwise.
        The state is an absolute representation of the entire pool.
        """
        self._update_sensing_window(current_time)
        # State matrix covers all slots, initialized to 0 (free)
        resource_state = np.zeros((self.num_slots, self.num_subchannels))
        
        # Iterate through all sensed data points
        for data in self.sensing_data:
            slot_id = data.slot_id % self.num_slots
            subchannel_id = data.subchannel
            
            # Check for valid subchannel index
            if 0 <= subchannel_id < self.num_subchannels:
                # Mark the resource as occupied (1)
                resource_state[slot_id, subchannel_id] = 1
                
        return resource_state

    def get_state(self, current_time):
        """Get the current state of the RL agent - including channel state, previous action, and time step"""
       
        resource_state_flat = self.get_individual_resource_state(current_time).flatten()

        # 2. Add action history (last 5 actions) - MODIFIED
        action_history_state = np.zeros(5)  # 5 actions, each action is a single value

        all_last_actions_state = np.full(self.sim.num_vehicles, -1.0, dtype=np.float32)
        total_actions = self.sim.action_space[self.id].n
        for vehicle in self.sim.vehicles:
            # This part about other vehicles' actions remains the same as it's about observable outcomes
            if isinstance(vehicle, RLVehicle) and vehicle.action_history:
                last_action = vehicle.action_history[-1]
                action_norm = last_action/total_actions
                
                all_last_actions_state[vehicle.id] = action_norm

        # MODIFIED: New logic for self action history
        
        for i, action in enumerate(self.action_history):
            if action is not None:
                action_history_state[i] = action / total_actions

        # 3. Add time step (normalized)
        time_norm = np.array([current_time / self.sim.episode_duration])

        full_state = np.concatenate([
            resource_state_flat,         # Channel state
            action_history_state,      # Action history (5-dimensional)
            time_norm,                 # Time step (1-dimensional)
            all_last_actions_state     # Previous actions of other vehicles
        ])

        return full_state.astype(np.float32)

    def select_send_resources(self, action):
        """Select send resources based on the action - improved version (using 100ms window)"""
        # MODIFIED: Store the raw action in history
        self.action_history.append(action)

        # Action decoding: action index to resource index
        num_subchannel_choices = self.sim.num_subchannels - 1
        if num_subchannel_choices <= 0: num_subchannel_choices = 1

        slot_idx = action // num_subchannel_choices
        subchannel = action % num_subchannel_choices
        
        # Get the current slot
        current_slot = self.sim.current_time % self.num_slots
        # Calculate the absolute slot (current slot + action index)
        slot = (current_slot + slot_idx) % self.num_slots

        # Select two adjacent resources
        sc1 = subchannel
        sc2 = subchannel + 1

        # Create resource objects
        resource1 = ResourceInfo(slot, sc1)
        resource2 = ResourceInfo(slot, sc2)

        # Save the currently selected resources
        current_resources = [resource1, resource2]
        current_set = set(current_resources)
        # Check if the resources have changed
        resources_changed = current_set != self.last_selected_resources
        self.last_selected_resources = current_set

        return current_resources, resources_changed

    def record_send_success(self, success_send):
        """Record successful sending"""
        if success_send:
            self.normal_success_count += 1

    def should_receive_packet(self, sender_position):
        """The attacker can receive packets within the communication range"""
        distance = np.linalg.norm(self.position - sender_position)
        return distance <= self.communication_range
    
    def calculate_reward(self, was_successful):
        """Calculates the base reward based on transmission outcome."""
        current_success_rate = self.successful_sends / self.total_sends if self.total_sends > 0 else 0.0
        self.last_success_rate = current_success_rate
        success_rate_reward = (current_success_rate - self.last_success_rate) * 5.0 # Weighting factor
        if was_successful:
            return 2.0 + success_rate_reward  # Reward for successful transmission
        else:
            return -8.0 + success_rate_reward # Penalty for collision
        
    def add_sensing_data(self, slot_id, subchannel, pRsvp, sender_id, timestamp, sender_position,expected_receivers):
        """Add sensing data from received transmissions"""
        sensing_data = SensingData(
            slot_id=slot_id,
            subchannel=subchannel,
            pRsvp=pRsvp,
            sender_id=sender_id,
            timestamp=timestamp,
            sender_position=sender_position,  # Added sender position
            expected_receivers=expected_receivers
        )
        self.sensing_data.append(sensing_data)

    def receive_packet(self, packet, resource, success):
        """Process received packets - modified to always add sensing data"""
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
            packet.position,  # Added sender position
            packet.expected_receivers
        )
        if not packet.is_attack:
            self.packets_received += 1
        if not packet.is_attack and success:
            self.packets_received_succeed += 1
            return True
        return False
    
    def _update_sensing_window(self, current_time):
        """Update the listening window, remove expired data"""
        sensing_window_start = current_time - self.sensing_window_duration

        self.sensing_data = [data for data in self.sensing_data
                            if data.timestamp >= sensing_window_start]


class RLAttacker:
    """RL-based attacker"""

    def __init__(self, attacker_id, initial_position, initial_velocity, sim, action_num=2):
        self.id = attacker_id
        self.position = initial_position
        self.velocity = initial_velocity
        self.sim = sim
        self.last_collison = 0
        self.communication_range = 320
        self.num_slots = sim.resource_pool.num_slots
        self.num_subchannels = sim.resource_pool.num_subchannels
        self.action_num = action_num  # Number of resource blocks selected by each agent
        self.total_resources = self.num_slots * self.num_subchannels  # Total number of resource blocks
        self.last_attack_success_rate = 0.0  # Last attack success rate
        self.next_transmission_time = 0
        self.transmission_cycle = self.num_slots
        self.current_resources = None
        self.attack_packets_sent = 0
        self.attack_success_count = 0
        self.collisions_caused = 0
        self.target_slot = -1
        self.sensing_data = []
        self.sensing_window_duration = self.num_slots
        self.last_action = None
        self.last_reward = 0
        self.resources_changed = False
        self.action_history = deque(maxlen=5)
        self.last_action_indices = np.zeros(5, dtype=np.int32)  # Store the indices of the last 5 actions
        self.attack_success_this_step = False  # Added: flag whether this transmission caused a collision
        self.target_vehicle_id = 0
        self.target_vehicle_resources = []
        self.target_vehicle_tracking_time = 0
        self.targeted_resources = []
        self.last_prr = 1.0
        self.last_attack_count = 0
        self.last_fail_counts = 0
        self.current_packet_id = 0
        # Added: save the last selected resources
        self.last_selected_resources = set()
        self.last_action_index = -1  # Store the resource index of the previous action
        self.transmission_count = 0

    def reset(self):
        """Reset attacker state"""
        self.last_collison = 0
        self.next_transmission_time = 0
        self.current_resources = None
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
        self.current_packet_id = 0
        self.last_prr = 1.0
        self.last_attack_count = 0
        self.last_fail_counts = 0
        self.action_history.clear()
        self.attack_success_this_step = False  # Added: flag whether this transmission caused a collision
        self.target_vehicle_id = -1
        self.target_vehicle_resources = []
        self.target_vehicle_tracking_time = 0
        self.targeted_resources = []
        self.last_action_index = -1
        self.last_selected_resources = set()
        self.last_action_indices = np.zeros(5, dtype=np.int32)  # Store the indices of the last 5 actions
        self.last_attack_success_rate = 0.0
        self.transmission_count = 0

    def move(self, delta_time):
        """Update attacker position"""
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
        """Get the resource state matrix of a single attacker (20xN_subchannels)"""
        self._update_sensing_window(current_time)
        resource_state = np.zeros((20, self.num_subchannels))
        current_slot = current_time % self.num_slots
        window_start = current_slot
        window_end = (current_slot + 20) % self.num_slots

        for sensing_data in self.sensing_data:
            slot_id = sensing_data.slot_id % self.num_slots
            
            # Check if the slot is within the [current_slot, current_slot + 20) window
            if window_start < window_end: # Window does not wrap around
                 if not (window_start <= slot_id < window_end):
                    continue
            else: # Window wraps around
                if not (slot_id >= window_start or slot_id < window_end):
                    continue
            
            if 0 <= sensing_data.subchannel < self.num_subchannels:
                # Calculate the index relative to the window start
                slot_index = (slot_id - window_start + self.num_slots) % self.num_slots

                if 0 <= slot_index < 20:
                    sender_position = sensing_data.sender_position
                    distance = np.linalg.norm(self.position - sender_position)
                    normalized_distance = max(0, 1 - min(1, distance / self.sim.communication_range))
                    if normalized_distance > resource_state[slot_index, sensing_data.subchannel]:
                        resource_state[slot_index, sensing_data.subchannel] = normalized_distance
        return resource_state


    def get_state(self, current_time):
        """Get the current state of the RL agent - including channel state, previous action, and time step"""
        resource_state_flat = self.get_individual_resource_state(current_time).flatten()

        # 2. Add action history (last 5 actions) - MODIFIED
        action_history_state = np.zeros(5)  # 5 actions, each action is a single value

        # MODIFIED: New logic for self action history
        attacker_agent_index = self.id - self.sim.num_vehicles
        total_actions = self.sim.action_space[attacker_agent_index].n
        for i, action in enumerate(self.action_history):
            if action is not None:
                action_history_state[i] = action / total_actions

        # 3. Add time step (normalized)
        time_norm = np.array([current_time / self.sim.episode_duration])

        # Combine the full state
        full_state = np.concatenate([
            resource_state_flat,         # Channel state
            action_history_state,      # Action history (5-dimensional)
            time_norm                   # Time step (1-dimensional)
        ])

        return full_state.astype(np.float32)

    def select_attack_resources(self, action):
        """Select attack resources based on the action - improved version (using 20ms window)"""
        # MODIFIED: Store the raw action in history
        self.action_history.append(action)
        
        # Action decoding: action index to resource index
        num_subchannel_choices = self.sim.num_subchannels - 1
        if num_subchannel_choices <= 0: num_subchannel_choices = 1
    
        slot_idx = action // num_subchannel_choices
        subchannel = action % num_subchannel_choices

        # Get the current slot
        current_slot = self.sim.current_time % self.num_slots
        # Calculate the absolute slot (current slot + action index)
        slot = (current_slot + slot_idx) % self.num_slots
        # print(self.sim.current_time,slot,self.num_slots)
        # Select two adjacent resources
        sc1 = subchannel
        sc2 = subchannel + 1 

        # Create resource objects
        resource1 = ResourceInfo(slot, sc1)
        resource2 = ResourceInfo(slot, sc2)

        # Save the currently selected resources
        current_resources = [resource1, resource2]
        current_set = set(current_resources)

        # Check if the resources have changed
        resources_changed = current_set != self.last_selected_resources
        self.last_selected_resources = current_set

        return current_resources, resources_changed

    def calculate_reward(self, collision_count, affect_vehicles):
        reward = 0.0

        # 1. Core collision reward (weighted by the number of affected vehicles)
        if collision_count > 0:
            reward += 3.0 * collision_count

        # 2. Added: high-value target reward (attacking communication hubs)
        hub_bonus = 0
        if affect_vehicles: # Ensure the set is not empty
            median_expected_receptions = np.median([v.expected_receptions for v in self.sim.vehicles if v.expected_receptions > 0] or [0])
            for vid in affect_vehicles:
                # Check if vid is valid
                if 0 <= vid < len(self.sim.vehicles):
                    vehicle = self.sim.vehicles[vid]
                    if vehicle.expected_receptions > median_expected_receptions:
                        hub_bonus += 2.0
        reward += hub_bonus

        distance_reward = 0.0
        for affected_vehicle_id in affect_vehicles:
            if 0 <= affected_vehicle_id < len(self.sim.vehicles):
                # Calculate the distance between the attacker and the affected vehicle
                dist = np.linalg.norm(np.array(self.position) - np.array(self.sim.vehicles[affected_vehicle_id].position))

                # Give a reward based on the distance
                if dist < 50:
                    distance_reward += 1.0  # High reward for close distance
                elif dist < 320:
                    # Linearly decay between 50-300 meters
                    distance_reward += 1.0 * (1 - (dist - 50) / 320)
        reward += distance_reward

        # if len(affect_vehicles)>1:
        #     reward -= 2.0*(len(affect_vehicles)-1)
        # 3. Efficiency reward (simplified version)
        current_success_rate = self.attack_success_count / max(1, self.attack_packets_sent)
        reward += 10.0 * (current_success_rate - self.last_attack_success_rate)

        # 4. Penalty term
        if collision_count == 0:
            reward -= 5.0  # Penalty for not causing a collision

        self.last_attack_success_rate = current_success_rate
        return np.clip(reward, -10.0, 15.0)

    def record_attack_success(self, collision_occurred):
        """Record successful attacks"""
        if collision_occurred:
            self.collisions_caused += 1

    def should_receive_packet(self, sender_position):
        """The attacker can receive packets within the communication range"""
        distance = np.linalg.norm(self.position - sender_position)
        return distance <= self.communication_range

    def add_sensing_data(self, slot_id, subchannel, pRsvp, sender_id, timestamp, sender_position,expected_receivers):
        """Add sensing data from received transmissions"""
        sensing_data = SensingData(
            slot_id=slot_id,
            subchannel=subchannel,
            pRsvp=pRsvp,
            sender_id=sender_id,
            timestamp=timestamp,
            sender_position=sender_position,  # Added sender position
            expected_receivers=expected_receivers
        )
        self.sensing_data.append(sensing_data)

        if TARGETED_ATTACK_MODE and sender_id == self.target_vehicle_id:
            resource = ResourceInfo(slot_id, subchannel)
            self.target_vehicle_resources.append(resource)

    def _update_sensing_window(self, current_time):
        """Update the listening window, remove expired data"""
        sensing_window_start = current_time - self.sensing_window_duration

        self.sensing_data = [data for data in self.sensing_data
                            if data.timestamp >= sensing_window_start]

class FixAttacker(Vehicle):
    """Fixed strategy attacker"""

    def __init__(self, attacker_id, initial_position, initial_velocity, sim,
                 attack_cycle=20, num_subchannels=2, resource_selection_mode='Combine',
                 attack_nearest=False, attack_most_and_nearest=False,
                 attack_fixed_subchannel=False, attack_periodic_group=False, is_target_attacker=True):
        super().__init__(attacker_id, initial_position, initial_velocity, sim, resource_selection_mode)
        self.is_attack = True
        self.is_target_attacker = is_target_attacker
        self.attack_cycle = attack_cycle
        self.num_subchannels_param = num_subchannels 
        self.next_attack_time = 0
        self.attack_packets_sent = 0
        self.collisions_caused = 0
        self.has_transmitted = False
        self.attack_success_this_step = False
        self.num_slots = sim.resource_pool.num_slots
        self.sensing_window_duration = self.num_slots
        self.prob_resource_keep = 0.2
        self.communication_range = 320
        self.cycle_groups = self._calculate_cycle_groups() if not TARGETED_ATTACK_MODE else []
        self.current_packet_id = 0
        self.attack_success_count = 0
        self.target_vehicle_id = 1
        self.target_vehicle_resources = []
        self.target_vehicle_tracking_time = 0
        self.targeted_resources = []
        self.num_vehicles = sim.num_vehicles
        self.target_tracking_enabled = True
        self.last_target_update_time = 0
        self.attack_nearest = attack_nearest
        self.attack_most_and_nearest = attack_most_and_nearest
        self.nearest_vehicle_id = -1
        self.nearest_vehicle_resources = []
        self.best_target_vehicle_id = -1
        self.best_target_resources = []
        self.attack_fixed_subchannel = attack_fixed_subchannel
        self.attack_periodic_group = attack_periodic_group
        self.transmission_count = 0

    def reset(self):
        """Reset attacker state"""
        super().reset()
        self.next_attack_time = 0
        self.attack_packets_sent = 0
        self.collisions_caused = 0
        self.current_packet_id = 0
        self.attack_success_count = 0
        self.sensing_data = []
        self.cycle_groups = self._calculate_cycle_groups() if not TARGETED_ATTACK_MODE else []
        self.attack_success_this_step = False
        self.target_vehicle_id = 1
        self.target_vehicle_resources = []
        self.target_vehicle_tracking_time = 0
        self.targeted_resources = []
        self.last_target_update_time = 0
        self.nearest_vehicle_id = -1
        self.nearest_vehicle_resources = []
        self.transmission_count = 0

    def _calculate_cycle_groups(self):
        """Calculate slot groups based on the attack cycle"""
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
    
    def _send_fixed_subchannel_attack(self, current_time):
        """Send jamming messages on subchannels 0 and 4 in all slots"""
        transmissions = []
        current_slot = current_time % self.num_slots
        attack_subchannels = [0, 4]
        packet = Packet(self.id, current_time, self.position, self.current_packet_id, is_attack=True)
        for sc in attack_subchannels:
            if sc < self.sim.num_subchannels:
                resource = ResourceInfo(current_slot, sc)
                self.current_resources = [resource]
                return
    
    def _send_periodic_group_attack(self, current_time):
        """Send jamming messages on all subchannels in slots 0, 10, 20...90"""
        current_slot = current_time % self.num_slots
        if current_slot % 10 == 0:
            resources = []
            for sc in range(self.sim.num_subchannels):
                resources.append(ResourceInfo(current_slot, sc))
            self.current_resources = resources
            return

    def select_and_set_resources(self, current_time):
        """ New method for the step-based logic. Selects resources but doesn't send. """
        self.current_packet_id += 1
        self.current_resources = [] # Clear previous selection
        #attacker0
        # return 
        if self.is_target_attacker:
            self._update_target_resources() # This updates self.targeted_resources
            self.current_resources = self.targeted_resources
        elif self.attack_fixed_subchannel:
            # This attack happens every time slot, so we need to determine resources based on current time
            attack_subchannels = [0, self.sim.num_subchannels -1] # e.g. [0, 4]
            # In the new logic, we pre-select all resources for the entire step duration
            for t_offset in range(self.sim.step_duration):
                slot = (current_time + t_offset) % self.num_slots
                for sc in attack_subchannels:
                    if sc < self.sim.num_subchannels:
                        self.current_resources.append(ResourceInfo(slot, sc))
        elif self.attack_periodic_group:
            # Pre-select all resources for the entire step duration
            for t_offset in range(self.sim.step_duration):
                slot = (current_time + t_offset) % self.num_slots
                if slot % 10 == 0:
                    for sc in range(self.sim.num_subchannels):
                        self.current_resources.append(ResourceInfo(slot, sc))
        elif self.attack_most_and_nearest:
            self._update_best_target_vehicle()
            self.current_resources = self.best_target_resources
        elif self.attack_nearest:
            self._update_nearest_vehicle()
            self.current_resources = self.nearest_vehicle_resources
        else: # Fallback to cycle group
            # This logic needs adaptation for step-based selection
            # For simplicity, let's make it select one random resource for now
            slot = random.randint(0, self.num_slots-1)
            sc1 = random.randint(0, self.num_subchannels-2)
            self.current_resources = [ResourceInfo(slot, sc1), ResourceInfo(slot, sc1+1)]

    def _update_best_target_vehicle(self):
        """Update the best target vehicle (comprehensively considering the number of expected receivers and distance)"""
        if not self.sensing_data:
            self.best_target_vehicle_id = -1
            self.best_target_resources = []
            return

        vehicle_info = {}
        for data in self.sensing_data:
            if data.sender_id < self.num_vehicles:
                if data.sender_id not in vehicle_info:
                    vehicle_info[data.sender_id] = {
                        'position': data.sender_position,
                        'timestamp': data.timestamp,
                        'expected_receivers': data.expected_receivers,
                        'resources': []
                    }
                resource = ResourceInfo(data.slot_id, data.subchannel)
                vehicle_info[data.sender_id]['resources'].append(resource)

        if not vehicle_info:
            self.best_target_vehicle_id = -1
            self.best_target_resources = []
            return

        best_score = -float('inf')
        best_vehicle_id = -1
        for vehicle_id, info in vehicle_info.items():
            distance = np.linalg.norm(self.position - info['position'])
            expected_receivers = info.get('expected_receivers', 0)
            alpha = 2.0
            beta = 0.2
            score = alpha * expected_receivers - beta * distance
            if score > best_score:
                best_score = score
                best_vehicle_id = vehicle_id

        self.best_target_vehicle_id = best_vehicle_id
        if best_vehicle_id != -1:
            self.best_target_resources = list(set(vehicle_info[best_vehicle_id]['resources'])) # Use set to get unique resources
            if len(self.best_target_resources) > 2:
                self.best_target_resources = self.best_target_resources[-2:]

    def _update_nearest_vehicle(self):
        """Update the information and resource usage of the nearest vehicle"""
        if not self.sensing_data:
            self.nearest_vehicle_id = -1
            self.nearest_vehicle_resources = []
            return

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

        min_distance = float('inf')
        nearest_vehicle_id = -1
        for vehicle_id, info in vehicle_positions.items():
            distance = np.linalg.norm(self.position - info['position'])
            if distance < min_distance:
                min_distance = distance
                nearest_vehicle_id = vehicle_id
        
        self.nearest_vehicle_id = nearest_vehicle_id
        if nearest_vehicle_id != -1:
            self.nearest_vehicle_resources = list(set(vehicle_positions[nearest_vehicle_id]['resources']))
            if len(self.nearest_vehicle_resources) > 2:
                self.nearest_vehicle_resources = self.nearest_vehicle_resources[-2:]

    def _update_target_resources(self):
        """Enhanced target resource selection logic - random selection or use of other vehicles when no data is available"""
        target_resources = []
        for data in self.sensing_data:
            if data.sender_id == self.target_vehicle_id:
                target_resources.append((data.timestamp, ResourceInfo(data.slot_id, data.subchannel)))
        
        if target_resources:
            target_resources.sort(key=lambda x: x[0], reverse=True)
            # Get unique resources from the sorted list
            unique_latest_resources = []
            seen_resources = set()
            for _, res in target_resources:
                if res not in seen_resources:
                    unique_latest_resources.append(res)
                    seen_resources.add(res)
            
            # Take the 2 most recent unique resources
            self.targeted_resources = unique_latest_resources[:2]
        else:
            # MODIFIED: If target is not sensed, select a random resource pair to attack.
            # This keeps the attacker active instead of idle.
            random_slot = random.randint(0, self.sim.num_slots - 1)
            if self.sim.num_subchannels > 1:
                # Select two adjacent subchannels
                sc1 = random.randint(0, self.sim.num_subchannels - 2)
                sc2 = sc1 + 1
                self.targeted_resources = [ResourceInfo(random_slot, sc1), ResourceInfo(random_slot, sc2)]
            elif self.sim.num_subchannels == 1:
                # Handle the edge case of only one subchannel
                self.targeted_resources = [ResourceInfo(random_slot, 0)]
            else:
                self.targeted_resources = []


    def add_sensing_data(self, slot_id, subchannel, pRsvp, sender_id, timestamp, sender_position,expected_receivers):
        """Add sensing data - extended to support target attacks"""
        super().add_sensing_data(slot_id, subchannel, pRsvp, sender_id, timestamp, sender_position,expected_receivers)
        if self.is_target_attacker and sender_id == self.target_vehicle_id:
            logger.debug(f"Attacker {self.id} sensed target vehicle {self.target_vehicle_id} using resource (slot:{slot_id}, subchannel:{subchannel})")

    def should_receive_packet(self, sender_position):
        """The attacker can receive packets within the communication range"""
        distance = np.linalg.norm(self.position - sender_position)
        return distance <= self.communication_range

    def _update_sensing_window(self, current_time):
        """Update the listening window, remove expired data"""
        sensing_window_start = current_time - self.sensing_window_duration
        self.sensing_data = [data for data in self.sensing_data if data.timestamp >= sensing_window_start]


class ResourcePool:
    """Manages the sidelink resource pool for V2X communications"""

    def __init__(self, num_slots=100, num_subchannels=5, subchannel_size=10):
        self.num_slots = num_slots
        self.num_subchannels = num_subchannels
        self.subchannel_size = subchannel_size
        self.total_rbs = num_subchannels * num_slots


class V2XRLEnvironment(gym.Env):
    """V2X攻击优化的RL环境 - 支持优化的SINR碰撞检测"""

    def __init__(self, kwargs):
        super(V2XRLEnvironment, self).__init__()

        self.num_vehicles = kwargs.get('num_vehicles', 20)
        self.num_attackers = kwargs.get('num_attackers', 1)
        self.episode_duration = kwargs.get('episode_duration', 20000)
        self.communication_range = kwargs.get('communication_range', 320.0)
        self.vehicle_resource_mode = kwargs.get('vehicle_resource_mode', 'Separate')
        self.attacker_type = kwargs.get('attacker_type', 'RL')
        # MODIFICATION: Add is_target_attacker flag to fix_attacker_params
        self.fix_attacker_params = kwargs.get('fix_attacker_params', {'cycle': 20, 'num_subchannels': 2, 'attack_nearest': False, 'attack_most_and_nearest': False})
        self.num_slots = kwargs.get('num_slots', 100)
        self.num_subchannels = kwargs.get('num_subchannels', 5)
        self.tx_power = kwargs.get('tx_power', 23.0)
        self.attacker_tx_power = 30.0
        self.noise_power = kwargs.get('noise_power', -95.0)
        self.action_num = kwargs.get('action_num', 2)
        self.use_sinr = kwargs.get('use_sinr', True)
        self.sinr_threshold = kwargs.get('sinr_threshold', 5.0)
        self.use_simple_interference = kwargs.get('use_simple_interference', False)
        self.print_detail = False
        self.use_rl_vehicles = kwargs.get('use_rl_vehicles', False)
        
        if self.use_rl_vehicles:
            self.n_agents = self.num_vehicles
        else:
            self.n_agents = self.num_attackers
        
        self.multi_agent_coordination = True if self.n_agents > 1 else False
        self.num_agents = self.n_agents

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
        self.last_global_prr = 1.0

        self.resource_grid = [ [ [] for _ in range(self.num_subchannels)] for _ in range(self.num_slots) ]

        self.vehicles = []
        self.attackers = []
        self._initialize_vehicles()
        self._initialize_attackers()
        
        # <<< MODIFICATION START: Define state and action spaces based on fixed requirements >>>
        if self.num_subchannels <= 1:
            num_subchannel_choices = 1
        else:
            num_subchannel_choices = self.num_subchannels - 1

        if self.use_rl_vehicles:
            # --- Configuration for RLVehicle agents ---
            self.step_duration = self.num_slots  # ms
            
            # STATE SPACE for RLVehicle: num_slots * num_subchannels
            obs_dim = self.num_slots * self.num_subchannels
            
            # ACTION SPACE for RLVehicle: num_slots * (num_subchannels - 1)
            total_actions = self.num_slots * num_subchannel_choices
            
            # Additional state space dimensions (action history, time, other agent actions)
            action_history_dim = 5
            time_dim = 1
            other_actions_dim = self.num_vehicles
            final_obs_dim = obs_dim + action_history_dim + time_dim + other_actions_dim
        else:
            # --- Configuration for RLAttacker agents ---
            self.step_duration = 20  # ms

            # STATE SPACE for RLAttacker: 20 * num_subchannels
            obs_dim = 20 * self.num_subchannels

            # ACTION SPACE for RLAttacker: 20 * (num_subchannels - 1)
            total_actions = 20 * num_subchannel_choices

            # Additional state space dimensions (action history, time)
            action_history_dim = 5
            time_dim = 1
            final_obs_dim = obs_dim + action_history_dim + time_dim

        # Handle the case where there are no RL agents to avoid errors
        if (self.attacker_type != 'RL' and not self.use_rl_vehicles):
            final_obs_dim = 1
            total_actions = 1
            self.step_duration = self.num_slots

        self.action_space = [spaces.Discrete(total_actions) for _ in range(self.n_agents)]
        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(final_obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        # <<< MODIFICATION END >>>

        share_obs_dim = self.num_slots * self.num_subchannels
        self.share_observation_space = [
             spaces.Box(low=0.0, high=1.0, shape=(share_obs_dim,), dtype=np.float32)
             for _ in range(self.n_agents)]
        # print(len(self.attackers),len(self.vehicles),self.num_agents)
        # exit(0)
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
                        resource_grid[resource.slot_id, resource.subchannel] = 1.0 # 标记为1

        return resource_grid.flatten()
    
    def render(self, mode='human'):
        pass # Render logic can be added here if needed

    def reset(self):
        # print("\nSimulation finished.")
    
        # # ===============================================================
        # # ===== AÑADE ESTE BLOQUE PARA IMPRIMIR LAS ESTADÍSTICAS FINALES =====
        # # ===============================================================
        # print("\n" + "="*40)
        # print(" " * 8 + "Final Episode Statistics Dictionary")
        # print("="*40)

        # # Llama a la función para obtener el diccionario de estadísticas
        # final_stats = self.get_episode_stats()

        # # Imprime el contenido del diccionario clave por clave
        # print(f"  'step_count': {final_stats['step_count']}")
        # print(f"  'total_collisions': {final_stats['total_collisions']}")
        # print(f"  'message_failures': {final_stats['message_failures']}")
        # print(f"  'prr': {final_stats['prr']:.4f}")
        # print(f"  'attack_success_rate': {final_stats['attack_success_rate']:.4f}")

        # # Imprime el diccionario anidado 'vehicle_prrs' de forma legible
        # print("  'vehicle_prrs': {")
        # if final_stats['vehicle_prrs']:
        #     for vehicle_id, prr in final_stats['vehicle_prrs'].items():
        #         print(f"    Vehicle {vehicle_id}: {prr:.4f},")
        # else:
        #     print("    (No data)")
        # print("  }")
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
                vehicle.position, vehicle.velocity = self.initial_vehicle_states[i]
        if self.initial_attacker_states:
             for i, attacker in enumerate(self.attackers):
                attacker.reset()
                attacker.position, attacker.velocity = self.initial_attacker_states[i]


        obs = []
        agent_list = self.vehicles if self.use_rl_vehicles else self.attackers
        rl_agent_class = RLVehicle if self.use_rl_vehicles else RLAttacker

        for agent in agent_list:
            if isinstance(agent, rl_agent_class):
                obs.append(agent.get_state(self.current_time))
            else:
                obs.append(np.zeros(self.observation_space[0].shape, dtype=np.float32))

        global_state = self._get_global_state()
        state = [global_state for _ in range(self.n_agents)]

        total_actions = self.action_space[0].n
        avail_actions = [np.ones(total_actions, dtype=np.float32) for _ in range(self.n_agents)]
        return obs, state, avail_actions

    
    def step(self, actions):
        """
        Executes one time step in the environment using a step-based resolution.
        """
        self.step_count += 1
        
        # --- 1. Action and Resource Selection Phase ---
        attacker_action_resources = defaultdict(list)
        
        # RL agents select resources based on actions
        if self.use_rl_vehicles:
            for i, vehicle in enumerate(self.vehicles):
                if isinstance(vehicle, RLVehicle):
                    resources, _ = vehicle.select_send_resources(actions[i])
                    vehicle.current_resources = resources
        else: # RL Attackers
            for i, attacker in enumerate(self.attackers):
                if isinstance(attacker, RLAttacker):
                    resources, _ = attacker.select_attack_resources(actions[i])
                    attacker.current_resources = resources
                    attacker_action_resources[attacker.id] = resources
                    attacker.sent_resources_count = 0

        # Non-RL agents select their resources
        for vehicle in self.vehicles:
            if isinstance(vehicle, Vehicle):
                if self.current_time%self.num_slots==0:
                    vehicle.has_transmitted = False
                if not vehicle.has_transmitted:
                    vehicle.handle_periodic_resource_reselection(self.current_time)
                    vehicle.has_transmitted = True

                    if vehicle.resel_counter > 0:
                        vehicle.resel_counter -= 1


                    vehicle.transmission_on_current_resource_counter += 1
                    vehicle.sent_resources_count = 0
                # tx_results = vehicle.send_packet(self.current_time)
                # # print("fale",self.current_time)
                # if tx_results:
                #     print("sender:",tx_results[0][1].slot_id,tx_results[0][1].subchannel ,self.current_time)
        
        for attacker in self.attackers:
            if isinstance(attacker, FixAttacker):
                attacker.select_and_set_resources(self.current_time)
                # tx_results = attacker.send_packet(self.current_time)
                # if tx_results:
                #     print("**attacker:",tx_results[0][1].slot_id,tx_results[0][1].subchannel,self.current_time)

        # --- 2. Transmission Aggregation & Outcome Resolution Phase ---
        resource_usage_map = defaultdict(list)
        # print(f"\n--- [Step {self.step_count} at Time {self.current_time}ms] Sender Resource Selections ---")
        all_agents = self.vehicles + self.attackers
        # for agent in sorted(all_agents, key=lambda x: x.id):
        #     if agent.current_resources:
        #         # 将资源对象转换为更易读的字符串列表
        #         res_str_list = [str(r) for r in agent.current_resources]
        #         agent_type = "Attacker" if isinstance(agent, (RLAttacker, FixAttacker)) else "Vehicle"
        #         print(f"  - {agent_type} ID {agent.id}: {', '.join(res_str_list)}")
        # print("-" * (60 + len(str(self.step_count)) + len(str(self.current_time))))
        left = int((self.current_time % self.num_slots)/self.step_duration)*self.step_duration
        right = left + self.step_duration
        
        for agent in all_agents:
            if agent.current_resources and agent.current_resources[0].slot_id >= left and agent.current_resources[0].slot_id < right:
                # Create a single packet for this agent for the whole step
                is_attack = isinstance(agent, (RLAttacker, FixAttacker))
                
                packet = Packet(agent.id, self.current_time, agent.position, agent.current_packet_id, is_attack=is_attack)
                # if agent.id == 1 or agent.id==20:
                    # print("send:",agent.id,[(r.slot_id,r.subchannel) for r in agent.current_resources],self.current_time)
                for resource in agent.current_resources:
                     # Only consider resources within the current step's time window for resolution
                    resource_usage_map[resource].append((agent, packet))
            agent.transmission_count+=1
        # Resolve all outcomes for this step
        step_outcomes = self._resolve_step_transmissions(resource_usage_map)

        total_collision_caused_due_to_attack = 0
        if step_outcomes['attacker_impact']:
            affected_vehicles = set()
            for vehicle_set in step_outcomes['attacker_impact'].values():
                affected_vehicles.update(vehicle_set)
            total_collision_caused_due_to_attack = len(affected_vehicles)
            
        # --- 3. Reward Calculation Phase ---
        rewards = []
        if self.use_rl_vehicles:
            current_prr = self._calculate_current_prr()
            prr_reward = (current_prr - self.last_global_prr) * 20.0
            self.last_global_prr = current_prr
            
            successful_tx = sum(1 for success in step_outcomes['rl_vehicle_status'].values() if success)
            team_reward = 2 * (successful_tx - 16) 

            for vehicle in self.vehicles:
                if isinstance(vehicle, RLVehicle):
                    was_successful = step_outcomes['rl_vehicle_status'].get(vehicle.id, False)
                    vehicle.total_sends += 1
                    if was_successful: vehicle.successful_sends += 1
                    
                    individual_reward = vehicle.calculate_reward(was_successful)
                    final_reward = individual_reward + team_reward + prr_reward
                    rewards.append([final_reward])
                else:
                    rewards.append([0.0])
        elif self.attacker_type == 'RL':
            team_reward = 0
            if self.n_agents > 1 and self.multi_agent_coordination:
                # resource_list = [res for res_list in attacker_action_resources.values() for res in res_list]
                # num_overlaps = len(resource_list) - len(set(resource_list))
                # team_reward -= num_overlaps * 3.0
                if len(step_outcomes['attacker_impact']) >= 2:
                    team_reward += 3.0

            for attacker in self.attackers:
                if isinstance(attacker, RLAttacker):
                    affected_vehicles = step_outcomes['attacker_impact'].get(attacker.id, set())
                    collision_count = len(affected_vehicles)
                    attacker.attack_success_count += 1 if collision_count > 0 else 0
                    attacker.attack_packets_sent += len(attacker.current_resources)
                    
                    individual_reward = attacker.calculate_reward(collision_count, affected_vehicles)
                    rewards.append([individual_reward + team_reward])
        else:
            rewards = [[0.0] for _ in range(self.n_agents)]

        # --- 4. State Update Phase ---
        delta_time_sec = self.step_duration / 1000.0
        for agent in all_agents:
            agent.move(delta_time_sec)
        self.current_time += self.step_duration

        next_obs = []
        agent_list = self.vehicles if self.use_rl_vehicles else self.attackers
        rl_agent_class = RLVehicle if self.use_rl_vehicles else RLAttacker
        for agent in agent_list:
            if isinstance(agent, rl_agent_class):
                next_obs.append(agent.get_state(self.current_time))
            else:
                next_obs.append(np.zeros(self.observation_space[0].shape, dtype=np.float32))

        next_global_state = self._get_global_state()
        next_state = [next_global_state for _ in range(self.n_agents)]

        # --- 5. Return values ---
        done = self.current_time >= self.episode_duration
        dones = [done for _ in range(self.n_agents)]
        
        # # <<< MODIFICATION START: Format the info dictionary as requested >>>
        # info = {
        #     'collisions_caused_this_step': total_collision_caused_due_to_attack,
        #     'total_collisions': self.collision_count,
        #     'attack_success_rate': self.total_attack_success / max(1, self.attack_transmission_count),
        #     'prr': self._calculate_current_prr(),
        #     'message_failures': self.message_failures,
        #     'collision_rate': self.collision_count / max(1, self.transmission_count)
        # }
        # # <<< MODIFICATION END >>>
        # infos = [info for _ in range(self.n_agents)]
        infos = {}

        # 1. 添加当前时间
        infos['current_time'] = self.current_time

        # 2. 初始化用于存储每个步骤详细信息的列表
        infos['resource_usage'] = []
        infos['transmission_outcomes'] = []
        infos['vehicle_prr'] = {}
        
        for resource, occupants in resource_usage_map.items():
            # 1. 为当前 resource 初始化一个空的占用者列表
            occupants_list = []
            
            # 2. 遍历这个 resource 上的所有占用者
            for occupant in occupants:
                occupant_type = 'Unknown'
                # 判断使用者类型
                if isinstance(occupant[0], RLVehicle):
                    occupant_type = 'RL_Vehicle'
                elif isinstance(occupant[0], (RLAttacker, FixAttacker)):
                    occupant_type = 'Attacker'
                elif isinstance(occupant[0], Vehicle):
                    occupant_type = 'Normal_Vehicle'
                
                
                # 3. 将每个占用者的信息作为一个字典添加到列表中
                occupants_list.append({
                    'id': occupant[0].id,
                    'type': occupant_type
                })
            
            # 4. 在处理完所有占用者后，将 resource 的完整信息一次性添加到最终结果中
            infos['resource_usage'].append({
                'resource_id': (resource.slot_id, resource.subchannel),
                'occupants': occupants_list
            })
        # # # 记录RL车辆的发送状态
        if 'rl_vehicle_status' in step_outcomes:
            infos['transmission_outcomes'].append(step_outcomes['rl_vehicle_status'])
        
        # 记录攻击者的干扰影响
        if 'attacker_impact' in step_outcomes:
            # 创建一个 attacker_impact 的可序列化副本
            serializable_impact = {}
            for attacker_id, impacted_vehicle_set in step_outcomes['attacker_impact'].items():
                # 将每个 set 转换为 list
                serializable_impact[attacker_id] = list(impacted_vehicle_set)

            # 将可序列化的版本添加到 infos 中
            infos['transmission_outcomes'].append(serializable_impact)
        # infos['transmission_outcomes'].append(step_outcomes)
            
        # 5. 记录各个车辆的PRR
        for vehicle in self.vehicles:
            infos['vehicle_prr'][vehicle.id] = vehicle.calculate_prr()
        
        infos['total_collisions'] = self.collision_count
        infos['message_failures'] = self.message_failures
        infos['prr'] = self._calculate_current_prr()
        infos['attack_success_rate'] = self.total_attack_success / max(1, self.attack_transmission_count)
        infos['collisions_caused_this_step'] = step_outcomes['collisions_caused_this_step']
        infos['collision_rate'] = self.collision_count / max(1, self.transmission_count)
        # print(infos)
        # 6. 为每个 agent 创建一个 info 字典
        final_infos = [infos for _ in range(self.num_vehicles)]
        
        total_actions = self.action_space[0].n
        avail_actions = [np.ones(total_actions, dtype=np.float32) for _ in range(self.n_agents)]

        return next_obs, next_state, rewards, dones, final_infos, avail_actions

    
    def _resolve_step_transmissions(self, resource_usage_map):
        """
        Resolves outcomes for all transmissions in a step based on resource usage.
        This function correctly handles two-block messages for SINR-based PRR calculation
        and identifies collisions for rewards.
        """
        outcomes = {
            'attacker_impact': defaultdict(set),
            'rl_vehicle_status': {},
            'collisions_caused_this_step' : 0
        }
        
        # MODIFICATION: Use a dictionary to track the success of each block for each message-receiver pair.
        # Key: (sender_id, packet_id, receiver_id), Value: list of success booleans [True, False, ...]
        message_reception_status = defaultdict(list)

        # ADDED: Main title for the report section
        # print("\n--- [SINR Resolution Report] ---")
        for resource, users in resource_usage_map.items():
            senders = [user[0] for user in users]
            sender_ids = [s.id for s in senders]  # ADDED: Get sender IDs

            # ADDED: Print resource block and which vehicles are using it
            # print(f"\nResource Block: {str(resource)}")
            # print(f"  Users (Senders): {sender_ids}")
            outcomes['rl_vehicle_status'].update({v.id: True for v in senders if isinstance(v, (RLVehicle,Vehicle))})
            normal_vehicles = [s for s in senders if not isinstance(s, (RLAttacker, FixAttacker))]
            attackers = [s for s in senders if isinstance(s, (RLAttacker, FixAttacker))]
            
            # if attackers:
            #     print(f"  'slotid': {resource.slot_id} 'subchannel': {resource.subchannel}")
            #     for sender in senders:
            #         print(f"  'sender': {sender.id}")
            #     for attacker in attackers:
            #         print(f"  'sender': {attacker.id}")
            # --- Update outcomes for reward calculation ---
            if len(senders) > 1:
                for sender in senders:
                    if isinstance(sender, (RLVehicle, Vehicle)):
                        outcomes['rl_vehicle_status'][sender.id] = False
                    if isinstance(sender,Vehicle):
                        sender.current_resources_collision = True
            
            if attackers:
                if normal_vehicles:
                    self.total_attack_success += 1
                    for attacker in attackers:
                        for vehicle in normal_vehicles:
                            outcomes['attacker_impact'][attacker.id].add(vehicle.id)
                self.attack_transmission_count +=1

            # --- SINR Calculation: Gather success status for each resource block ---
            for sender, packet in users:
                interferers = [u[0] for u in users if u[0].id != sender.id]
                interferer_positions = [i.position for i in interferers]
                has_attack_in_interferers = any(isinstance(i, (RLAttacker, FixAttacker)) for i in interferers)
                
                # ADDED: Print a sub-header for the current sender being analyzed
                # print(f"  - Analyzing Sender ID: {sender.id}")
                
                for receiver in self.vehicles:
                    if receiver.id == sender.id:
                        continue
                    
                    if receiver.should_receive_packet(sender.position):
                        
                        # ADDED: Calculate sender-receiver distance
                        sender_dist = np.linalg.norm(receiver.position - sender.position)

                        # ADDED: Calculate interferer distances and format them for printing
                        interferer_dists_str = []
                        for interferer in interferers:
                            dist = np.linalg.norm(receiver.position - interferer.position)
                            interferer_dists_str.append(f"ID {interferer.id}: {dist:.1f}m")
                        interferer_info_str = f"Interferers: [{', '.join(interferer_dists_str)}]" if interferers else "Interferers: [None]"

                        sinr = self.sinr_calculator.calculate_sinr_optimized(
                            receiver.position, sender.position, interferer_positions, 
                            self.attacker_tx_power, has_attack_in_interferers
                        )
                        success = sinr >= self.sinr_threshold
                        
                        status_str = "Success" if success else "Fail"
                        
                        # # MODIFIED: Update the print statement to include distances
                        # print(f"    - Rx ID: {receiver.id} | Tx Dist: {sender_dist:.1f}m | SINR: {sinr:.2f} dB | Status: {status_str}")
                        # print(f"      L {interferer_info_str}")


                        # Update receiver's sensing data regardless of message type
                        receiver.receive_packet(packet, resource, success)

                        # Only track reception status for non-attack packets for PRR calculation
                        if not packet.is_attack:
                            msg_key = (sender.id, packet.packet_id, receiver.id)
                            message_reception_status[msg_key].append(success)
                        if not isinstance(sender, (RLAttacker, FixAttacker)):   
                            self.transmission_count +=1
                            # sender.total_expected_receptions+=1
                            if not success:
                                self.collision_count += 1
                                sender.collisions+=1
                                outcomes['collisions_caused_this_step'] += 1
                            # else:
                                # sender.total_successful_receptions +=1 
                for receiver in self.attackers:
                    if receiver.should_receive_packet(sender.position):
                                    pRsvp = 100 if not isinstance(sender, (RLAttacker, FixAttacker)) else 20
                                    receiver.add_sensing_data(
                                        resource.slot_id,
                                        resource.subchannel,
                                        pRsvp,
                                        sender.id,
                                        packet.timestamp,
                                        packet.position,
                                        packet.expected_receivers
                                    )
        
        # print("--- [End SINR Resolution Report] ---\n")  # ADDED: Footer for the report section

        # --- Final Tally: Process the tracked message statuses to update counters correctly ---
        # A sender's message is defined by a unique packet_id. We need to avoid double-counting expected packets.
        processed_expected_packets = set() # (sender_id, packet_id)

        for msg_key, success_list in message_reception_status.items():
            sender_id, packet_id, _ = msg_key
            sender = self.vehicles[sender_id] # Assuming vehicle IDs are 0 to num_vehicles-1

            # Increment expected packets only once per transmitted message
            # if (sender_id, packet_id) not in processed_expected_packets:
                # A message is expected to be received by all vehicles in range.
                # Here we are processing on a per-receiver basis, which is correct for PRR.
                # sender.expected_receptions += 1
                
                # Mark this sender's packet as processed for expected counts to avoid recounting
                # Note: This logic assumes one receiver is processed at a time. A better way is to count all receivers in range once.
                # However, this per-receiver counting is standard for PRR calculation.
            
            # A message is successful if both its blocks were received successfully.
            # It's a failure if it's incomplete (less than 2 blocks) or if any block failed.
            is_complete = len(success_list) == 2
            is_successful = all(success_list)

            if is_complete:
                sender.total_expected_receptions+=1
                self.total_expected_packets += 1
                if is_successful:
                    sender.total_successful_receptions += 1
                    self.total_received_packets += 1
                else:
                    self.message_failures += 1
                # We only count a message failure once, even if it's for the same sender-receiver pair
                # The current loop structure naturally handles this.
                # Failures are tallied at the end
            
        # <<< MODIFICATION START: Per-attacker success counting moved here >>>
        total_vehicles_impacted_by_attack = 0
        # Iterate through the outcomes to update each attacker's individual counter
        for attacker_id, impacted_vehicle_ids in outcomes['attacker_impact'].items():
            collision_count = len(impacted_vehicle_ids)
            if collision_count > 0:
                # Find the attacker object by its ID to update its counter
                # Attacker IDs start after vehicle IDs, so their index is id - num_vehicles
                attacker_index = attacker_id - self.num_vehicles
                if 0 <= attacker_index < len(self.attackers):
                    attacker = self.attackers[attacker_index]
                    attacker.attack_success_count += collision_count
            total_vehicles_impacted_by_attack += collision_count
        
        # Update the global counter for the info dict
        # self.total_attack_success += total_vehicles_impacted_by_attack
        # outcomes['collisions_caused_this_step'] = total_vehicles_impacted_by_attack
        # Now, calculate total message failures based on the difference
        # between expected receptions and successful receptions.
        # self.message_failures = self.total_expected_packets - self.total_received_packets
        # Also update individual sender collision stats

        # Remove the old, incorrect division-by-two logic.
        
        return outcomes
    
    def _distance(self, pos1, pos2):
        """计算两点之间的欧几里得距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _calculate_current_prr(self):
        """计算当前分组接收率(PRR)"""
        if self.total_expected_packets > 0:
            return self.total_received_packets / self.total_expected_packets
        # Return 1.0 if no packets were expected, to avoid penalizing inaction
        return 1.0

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
            'prr': self._calculate_current_prr(),
            'attack_success_rate': self.total_attack_success / max(1, self.attack_transmission_count),
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
        self.last_global_prr = 1.0

    def _initialize_vehicles(self):
        """初始化车辆位置和速度，并保存初始状态"""
        lane1_y = 5.0
        lane2_y = 10.0
        highway_length = 1000.0
        self.vehicles = []
        vehicle_states = []
        for i in range(self.num_vehicles):
            lane_y = lane1_y if i % 2 == 0 else lane2_y
            pos_x = random.uniform(0, highway_length)
            position = np.array([pos_x, lane_y])
            velocity = np.array([33.0 if lane_y == lane1_y else -33.0, 0.0])
            if self.use_rl_vehicles:
                vehicle = RLVehicle(i, position.copy(), velocity.copy(), self, self.action_num)
            else:
                vehicle = Vehicle(i, position.copy(), velocity.copy(), self, self.vehicle_resource_mode)
            self.vehicles.append(vehicle)
            vehicle_states.append((position.copy(), velocity.copy()))
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
                attacker = RLAttacker(attacker_id, position.copy(), velocity.copy(), self, self.action_num)
            else: # Fixed Attacker
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
