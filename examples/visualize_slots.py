import json
import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_slot_usage(json_file_path, output_dir='slot_visualizations'):
    """
    读取包含时隙使用数据的JSON文件，并为每个时间步生成可视化图片。

    参数:
    - json_file_path (str): 输入的JSON文件路径。
    - output_dir (str): 保存生成图片的目标目录。
    """
    # --- 1. 准备工作 ---
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    # 读取JSON数据
    try:
        with open(json_file_path, 'r') as f:
            all_steps_data = json.load(f)
        print(f"成功读取 {len(all_steps_data)} 条记录从 {json_file_path}")
    except FileNotFoundError:
        print(f"错误: JSON文件未找到 at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"错误: 无法解析JSON文件 {json_file_path}")
        return

    # --- 2. 遍历每个时间步并生成图片 ---
    for step_index, step_data in enumerate(all_steps_data):
        # 初始化60个时隙的状态
        num_slots = 60
        # 每个时隙的状态: [颜色, '显示的文本']
        slot_states = [['white', ''] for _ in range(num_slots)]

        # 解析当前时间步的数据
        for slot_str, occupants in step_data.items():
            slot_id = int(slot_str)
            if 0 <= slot_id < num_slots:
                vehicles = [p for p in occupants if p < 20]
                attackers = [p for p in occupants if p >= 20]
                
                # 根据规则确定颜色和文本
                if attackers and vehicles or len(vehicles) > 1:
                    # 碰撞: 红色
                    slot_states[slot_id][0] = 'red'
                elif attackers and not vehicles:
                    # 只有攻击者: 蓝色
                    slot_states[slot_id][0] = 'lightblue'
                elif vehicles and not attackers:
                    # 只有车辆: 绿色
                    slot_states[slot_id][0] = 'lightgreen'
                
                # 合并所有ID用于显示
                all_ids_str = ','.join(map(str, sorted(occupants)))
                slot_states[slot_id][1] = all_ids_str

        # --- 3. 使用Matplotlib绘图 ---
        fig, ax = plt.subplots(figsize=(20, 4))
        
        for i in range(num_slots):
            color, text = slot_states[i]
            # 绘制背景色块
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color, ec='black'))
            # 在色块中添加ID文本
            ax.text(i + 0.5, 0.5, text, ha='center', va='center', fontsize=10, color='black')

        # --- 4. 美化图表并保存 ---
        ax.set_xlim(0, num_slots)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, num_slots, 5))
        ax.set_yticks([])
        ax.set_xlabel('Slot ID')
        ax.set_title(f'step {step_index}: resource slot usage')
        
        # 创建图例
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color='lightgreen', label='only vehicles'),
            plt.Rectangle((0, 0), 1, 1, color='lightblue', label='only attackers'),
            plt.Rectangle((0, 0), 1, 1, color='red', label='collision (vehicles & attackers)')
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left')

        # 保存图片
        output_filename = os.path.join(output_dir, f'step_{step_index:04d}.jpg')
        plt.savefig(output_filename, bbox_inches='tight', dpi=150)
        plt.close(fig) # 关闭图形，释放内存

    print(f"可视化完成！共 {len(all_steps_data)} 张图片已保存到 '{output_dir}' 目录。")

# --- 主程序入口 ---
if __name__ == '__main__':
    # 将这里的JSON文件名替换为您自己的文件名
    json_file = '/home/xhl/mywork/harl/examples/results/v2v/my_v2v_env/happo/newTest/newTestWithEval/newTest/TESTWITH_INITIAL/TEST20RLVechile_with_1_pretrained_attacker_use_pretrained_vehicle/seed-00042-2025-10-16-16-13-08/infos/slot_usage_details/eval_episode_1_at_2025-10-17_08-07-40.json'
    visualize_slot_usage(json_file)