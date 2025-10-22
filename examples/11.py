import argparse
import numpy as np
import json
# 确保从您保存环境的文件中导入
from harl.envs.v2v_env.env_copy_4 import V2XRLEnvironment 
import argparse
import numpy as np
import json
import time
import random
# Make sure to import the environment class from your final environment file
# from env_copy_4_1111 import V2XRLEnvironment

def main():
    """
    Main function to run a V2X simulation without any training.
    This script initializes the V2X environment in a specified mode,
    runs it for one full episode, and prints the final statistics.
    """
    parser = argparse.ArgumentParser(
        description="Run a V2X simulation to evaluate agents without training.",
        formatter_class=argparse.RawTextHelpFormatter  # Allows for better formatting in help text
    )

    # --- Core Simulation Arguments ---
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="""Defines the simulation scenario. Choose from:
    - 'eval_fix_attacker': Rule-based Vehicles vs. FixAttacker.
    - 'eval_pretrained_vehicle': Pre-trained RL Senders vs. FixAttacker.
    - 'eval_pretrained_attacker': Rule-based Vehicles vs. Pre-trained RL Attacker.
    - 'eval_pretrained_vehicle_vs_pretrained_attacker': Pre-trained RL Senders vs. Pre-trained RL Attacker.
"""
    )
    parser.add_argument("--num_vehicles", type=int, default=20, help="Number of normal vehicles.")
    parser.add_argument("--num_attackers", type=int, default=1, help="Number of attackers.")
    parser.add_argument("--episode_duration", type=int, default=20000, help="Total simulation time in milliseconds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # --- Model Path Arguments (only needed for modes with pre-trained models) ---
    parser.add_argument("--rl_vehicle_model_path", type=str, default="/home/xhl/mywork/harl/examples/results/v2v/my_v2v_env/happo/newTest/newTestRLVehcile/TEST_20RLVEHICLE_1FIX/seed-00042-2025-09-30-16-36-33/models", help="Path to pre-trained vehicle models.")
    parser.add_argument("--rl_vehicle_config_path", type=str, default="/home/xhl/mywork/harl/examples/results/v2v/my_v2v_env/happo/newTest/newTestRLVehcile/TEST_20RLVEHICLE_1FIX/seed-00042-2025-09-30-16-36-33/config.json", help="Path to pre-trained vehicle config.json.")
    parser.add_argument("--rl_attacker_model_path", type=str, default="/home/xhl/mywork/harl/examples/results/v2v/my_v2v_env/happo/newTest/compare_with_eval/test_pretrained/newTestPretrainedRLVehicleWith1Attacker/seed-00042-2025-10-02-20-20-56/models", help="Path to pre-trained attacker models.")
    parser.add_argument("--rl_attacker_config_path", type=str, default="/home/xhl/mywork/harl/examples/results/v2v/my_v2v_env/happo/newTest/compare_with_eval/test_pretrained/newTestPretrainedRLVehicleWith1Attacker/seed-00042-2025-10-02-20-20-56/config.json", help="Path to pre-trained attacker config.json.")

    args = parser.parse_args()
    
    # Set the seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Convert argparse Namespace to a dictionary to pass to the environment
    env_kwargs = vars(args)

    # --- Simulation Setup ---
    try:
        env = V2XRLEnvironment(kwargs=env_kwargs)
    except (ValueError, FileNotFoundError) as e:
        print(f"\n--- Error creating environment: {e} ---")
        print("Please ensure you have provided the correct --mode and any required model/config paths.")
        return

    # Reset the environment to get the initial state
    env.reset()
    done = False
    step_count = 0

    print("\n" + "="*50)
    print(f"--- Starting Simulation ---")
    print(f"Mode: '{args.mode}'")
    print(f"Configuration: {args.num_vehicles} Vehicles, {args.num_attackers} Attackers")
    print("="*50 + "\n")
    
    start_time = time.time()

    # --- Main Simulation Loop ---
    while not done:
        # Create a dummy action array. The content does not matter for evaluation modes,
        # as the environment will use its internal logic (rules or pre-trained models).
        # The size must match the number of agents the environment expects.
        dummy_actions = [0] * env.n_agents 

        # Take a step in the environment
        _, _, _, dones, infos, _ = env.step(dummy_actions)
        
        # 'dones' is a list of booleans. The simulation ends when the first one is True.
        done = dones[0]
        step_count += 1

        # Optional: Print progress every 100 steps
        if step_count % 100 == 0:
            info = infos[0]  # All info dicts in the list are identical
            print(f"Time: {info['current_time']:>5}ms | PRR: {info['prr']:.3f} | Total Collisions: {info['total_collisions']}")

    end_time = time.time()
    simulation_duration = end_time - start_time

    # --- Final Results ---
    print("\n" + "="*50)
    print(f"--- Simulation Finished in {simulation_duration:.2f} seconds ---")
    
    # Print final statistics from the last info dictionary
    final_info = infos[0]
    print("Final Statistics:")
    print(f"  - Simulation Duration: {final_info['current_time']}ms")
    print(f"  - Global PRR (Packet Reception Rate): {final_info['prr']:.4f}")
    print(f"  - Total Collisions (failed receptions): {final_info['total_collisions']}")
    print(f"  - Total Message Failures (incomplete): {final_info['message_failures']}")
    print(f"  - Attack Success Rate: {final_info['attack_success_rate']:.4f}")
    print("="*50 + "\n")

    env.close()

if __name__ == "__main__":
    main()
    
# def main():
#     parser = argparse.ArgumentParser(description="Run a V2X simulation without training.")
    
#     # --- Essential Arguments ---
#     parser.add_argument("--mode", type=str, required=True, 
#                         help="Defines the simulation scenario. E.g., 'eval_fix_attacker', 'eval_pretrained_vehicle', etc.")
#     parser.add_argument("--num_vehicles", type=int, default=20, help="Number of normal vehicles.")
#     parser.add_argument("--num_attackers", type=int, default=1, help="Number of attackers.")
#     parser.add_argument("--episode_duration", type=int, default=20000, help="Total simulation time in ms.")
    
#     # --- Model Path Arguments (only needed for modes with pre-trained models) ---
#     parser.add_argument("--rl_vehicle_model_path", type=str, default="/home/xhl/mywork/harl/examples/results/v2v/my_v2v_env/happo/newTest/newTestRLVehcile/TEST_20RLVEHICLE_1FIX/seed-00042-2025-09-30-16-36-33/models", help="Path to pre-trained vehicle models.")
#     parser.add_argument("--rl_vehicle_config_path", type=str, default="/home/xhl/mywork/harl/examples/results/v2v/my_v2v_env/happo/newTest/newTestRLVehcile/TEST_20RLVEHICLE_1FIX/seed-00042-2025-09-30-16-36-33/config.json", help="Path to pre-trained vehicle config.json.")
#     parser.add_argument("--rl_attacker_model_path", type=str, default="/home/xhl/mywork/harl/examples/results/v2v/my_v2v_env/happo/newTest/compare_with_eval/test_pretrained/newTestPretrainedRLVehicleWith1Attacker/seed-00042-2025-10-02-20-20-56/models", help="Path to pre-trained attacker models.")
#     parser.add_argument("--rl_attacker_config_path", type=str, default="/home/xhl/mywork/harl/examples/results/v2v/my_v2v_env/happo/newTest/compare_with_eval/test_pretrained/newTestPretrainedRLVehicleWith1Attacker/seed-00042-2025-10-02-20-20-56/config.json", help="Path to pre-trained attacker config.json.")

#     args = parser.parse_args()
    
#     # Convert argparse Namespace to a dictionary to pass to the environment
#     env_kwargs = vars(args)

#     # Create the environment using the specified mode and parameters
#     env = V2XRLEnvironment(kwargs=env_kwargs)
    
#     # Reset the environment to get the initial state
#     env.reset()
#     done = False
#     step_count = 0

#     print("\n" + "="*50)
#     print(f"--- Starting Simulation in Mode: '{args.mode}' ---")
#     print(f"--- {args.num_vehicles} Vehicles, {args.num_attackers} Attackers ---")
#     print("="*50 + "\n")
    
#     # Main simulation loop
#     while not done:
#         # Create a dummy action array. The content doesn't matter for evaluation/simulation modes,
#         # as the environment will use internal logic (rules or pre-trained models).
#         # The size must match the number of agents the environment expects for the framework.
#         dummy_actions = [0] * env.n_agents 

#         # Take a step in the environment
#         # The 'actions' parameter will be ignored by rule-based and pre-trained agents.
#         _, _, _, dones, infos, _ = env.step(dummy_actions)
        
#         # The 'dones' list contains a boolean for each agent. We check if the simulation is done.
#         done = dones[0]
#         step_count += 1

#         # Optional: Print progress every 100 steps
#         if step_count % 100 == 0:
#             info = infos[0] # All info dicts in the list are identical
#             print(f"Time: {info['current_time']:>5}ms | PRR: {info['prr']:.3f} | Total Collisions: {info['total_collisions']}")

#     print("\n" + "="*50)
#     print("--- Simulation Finished ---")
    
#     # Print final statistics from the last info dictionary
#     final_info = infos[0]
#     print("Final Statistics:")
#     print(f"  - Duration: {final_info['current_time']}ms")
#     print(f"  - Global PRR (Packet Reception Rate): {final_info['prr']:.4f}")
#     print(f"  - Total Collisions: {final_info['total_collisions']}")
#     print(f"  - Message Failures: {final_info['message_failures']}")
#     print(f"  - Attack Success Rate: {final_info['attack_success_rate']:.4f}")
#     print("="*50 + "\n")

#     env.close()

# if __name__ == "__main__":
#     main()