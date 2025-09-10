import torch
import copy
import random
from torch.optim import Adam
from .happo import HAPPO
from .ea_ha_agent import EA_HA_Agent_SR, EA_HA_Agent_W
from .ea_mine import MINE
from harl.models.value_function_models.v_net import VNet

class EA_HAPPO(HAPPO):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        # First, initialize the parent HAPPO class
        super(EA_HAPPO, self).__init__(args, obs_space, act_space, device)
        
        # --- Override Actor with EA-specific structure ---
        # We need to rebuild the actor to have SR and W parts
        self.actors = []
        self.agent_SR = [] # Shared Representation networks
        self.agent_W_populations = [] # Populations of W networks
        
        for i in range(self.num_agents):
            # Create SR part for each agent type
            sr_net = EA_HA_Agent_SR(args, self.obs_space[i].shape[0], self.act_space[i].n, device)
            self.agent_SR.append(sr_net)

            # Create W part population for each agent type
            w_population = []
            main_w_net = EA_HA_Agent_W(args, self.act_space[i].n, device)
            w_population.append(main_w_net) # Main actor's W is at index 0
            for _ in range(self.args.pop_size):
                w = EA_HA_Agent_W(args, self.act_space[i].n, device)
                w.load_state_dict(main_w_net.state_dict())
                w_population.append(w)
            self.agent_W_populations.append(nn.ModuleList(w_population))
        
        self.agent_SR = nn.ModuleList(self.agent_SR)
        self.agent_W_populations = nn.ModuleList(self.agent_W_populations)
        
        # Optimizer should only train the main agent (SR + W_main)
        actor_params = list(self.agent_SR.parameters())
        for pop in self.agent_W_populations:
            actor_params += list(pop[0].parameters())
        self.actor_optimizer = Adam(params=actor_params, lr=self.lr, eps=self.opti_eps)

        # --- Initialize EA-specific networks and optimizers ---
        # 1. PeVFA Critic (Parameter-evolving Value Function Approximator)
        # It evaluates the quality of population members
        self.pevfa_critic = VNet(self.args, self.state_dim, self.device)
        self.target_pevfa_critic = copy.deepcopy(self.pevfa_critic)
        self.pevfa_critic_optimizer = Adam(self.pevfa_critic.parameters(), lr=self.critic_lr, eps=self.opti_eps)
        
        # 2. MINE for mutual information estimation
        self.mine_nets = nn.ModuleList([
            MINE(self.hidden_size, self.state_dim) for _ in range(self.num_agents)
        ])
        self.mine_optimizer = Adam(self.mine_nets.parameters(), lr=self.critic_lr, eps=self.opti_eps)
    
    def get_actions(self, obs, state, rnn_states, masks, agent_id, pop_id=0, available_actions=None, deterministic=False):
        """
        Modified get_actions to select from population.
        pop_id = 0 -> main actor
        pop_id > 0 -> population member
        """
        # Get feature from SR network
        actor_features, rnn_states = self.agent_SR[agent_id](obs, rnn_states)
        # Get logits from W network of the selected population member
        action_logits = self.agent_W_populations[agent_id][pop_id](actor_features)
        
        actions, action_log_probs = self.action_dist_calculator.sample(action_logits, available_actions, deterministic)
        
        return actions, action_log_probs, rnn_states

    def train(self, actor_buffer, critic_buffer):
        """
        The core training function, combining HAPPO loss with EA guidance and MINE loss.
        """
        # --- 1. Train PeVFA Critic ---
        # This critic is trained to evaluate population members.
        # We randomly sample one population member to generate value targets.
        pop_id = random.randint(1, self.args.pop_size)
        
        # Get value predictions from the target PeVFA critic using a population policy
        # Note: This requires a full forward pass to get actions and values, which can be computationally expensive.
        # For simplicity, we use the main critic's returns as a proxy target for the PeVFA critic.
        # A more rigorous implementation would re-evaluate the buffer with the population policy.
        _, pevfa_values, _ = self.pevfa_critic.evaluate_actions(critic_buffer.get_state(), actor_buffer.get_obs(), actor_buffer.get_actions())
        pevfa_returns = critic_buffer.get_returns() # Using main returns as target
        pevfa_critic_loss = self.cal_value_loss(pevfa_values, pevfa_returns, critic_buffer.get_active_masks())
        
        self.pevfa_critic_optimizer.zero_grad()
        pevfa_critic_loss.backward()
        self.pevfa_critic_optimizer.step()

        # --- 2. Standard HAPPO Actor and Critic Training ---
        # This part remains mostly the same as in the original HAPPO class
        train_info = super().train(actor_buffer, critic_buffer)
        
        # --- 3. Compute EA Guidance and MINE Losses ---
        ea_pg_loss = 0
        mine_loss = 0
        
        # Get hidden states and global states from the buffer
        obs_batch = actor_buffer.get_obs() # Shape: (batch_size, num_agents, obs_dim)
        state_batch = critic_buffer.get_state() # Shape: (batch_size, state_dim)
        
        for i in range(self.num_agents):
            # Recalculate features and logits for the main actor
            features, _ = self.agent_SR[i](obs_batch[:, i, :], actor_buffer.rnn_states[i])
            logits = self.agent_W_populations[i][0](features)

            # a) EA Guidance Loss
            # The advantage is how much better the main actor's action is than the population's average.
            # We use the PeVFA critic to get the advantage signal.
            with torch.no_grad():
                pevfa_values_guidance, _ = self.pevfa_critic.get_values(state_batch)
            advantages = (critic_buffer.get_returns() - pevfa_values_guidance).detach()
            
            # Use action log probs from the buffer for the main actor
            action_log_probs = actor_buffer.action_log_probs[i]
            ea_pg_loss_agent = - (action_log_probs * advantages).mean()
            ea_pg_loss += ea_pg_loss_agent
            
            # b) MINE Loss
            # Maximize mutual information between agent's hidden state and global state
            # Flatten for MINE input
            state_flat = state_batch.view(-1, self.state_dim)
            features_flat = features.view(-1, self.hidden_size)
            # Ensure same batch size for both inputs
            if len(state_flat) > len(features_flat):
                state_flat = state_flat[:len(features_flat)]
            elif len(features_flat) > len(state_flat):
                features_flat = features_flat[:len(state_flat)]

            mine_loss_agent = self.mine_nets[i](features_flat, state_flat)
            mine_loss += mine_loss_agent
        
        # --- 4. Combine losses and optimize ---
        # The original HAPPO loss is already computed and backpropagated in super().train()
        # We now backpropagate the auxiliary losses.
        
        ea_total_loss = self.args.EA_alpha * ea_pg_loss + self.args.state_alpha * mine_loss
        
        # We need to zero_grad again for actor and mine optimizers
        self.actor_optimizer.zero_grad()
        self.mine_optimizer.zero_grad()
        
        ea_total_loss.backward()
        
        self.actor_optimizer.step()
        self.mine_optimizer.step()
        
        # --- 5. Update Target Network ---
        self.soft_update(self.target_pevfa_critic, self.pevfa_critic, self.tau)

        train_info['ea_pg_loss'] = ea_pg_loss.item()
        train_info['mine_loss'] = mine_loss.item()
        
        return train_info