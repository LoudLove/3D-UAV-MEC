"""
任务参数随机初始化脚本
为PathPlanner中的每个点生成随机任务参数，计算本地和卸载的时延/能耗，生成甘特图
"""

import json
import numpy as np
import math
from typing import Dict, List, Any, Optional
import os

# =====================================================================
# 系统参数（来自local_vs_offload_compare.py）
# =====================================================================
B = 20e6                         # Hz 带宽
sigma2 = 2.53e-13                # W 噪声功率
h0_dB = -50.0
h0 = 10 ** (h0_dB / 10.0)        # 线性
H_hov = 10.0                     # m UAV悬停高度
f_c = 8e9                        # Hz 边缘计算频率
gamma_c = 1e-28                  # 芯片系数
O_k = 0.5                        # 卸载输出比例
P_k_ul = 0.05                    # W IMD上行发射功率
P_u_dl = 1.0                     # W UAV下行发射功率
P_u_ul = 0.5                     # W UAV上行接收电路功率
P_k_dl = 0.2                     # W IMD下行接收电路功率
C_per_bit = 1000.0               # cycles/bit
f_loc = 1.5e9                    # Hz IMD本地频率
gamma_loc = 1e-28                # 本地芯片系数

# 固定信道增益
FIXED_CHANNEL_GAIN_DB = -70.0    # dB

# 本地计算可靠性参数
# 本地计算可靠性参数
LOCAL_SUCCESS_PROB = 0.10       # 本地计算成功概率
LOCAL_FAIL_COMPLETION_LOW = 0.05  # 本地失败时完成率下界
LOCAL_FAIL_COMPLETION_HIGH = 0.10 # 本地失败时完成率上界
NON_TRIGGER_RATIO=0.4
# =====================================================================
# 通信和计算函数
# =====================================================================

def channel_gain_from_db(h_dB: float) -> float:
    """从dB转换为线性信道增益"""
    return 10 ** (h_dB / 10.0)


def rate_bps_per_hz(P_tx: float, h_lin: float) -> float:
    """计算速率 R = log2(1 + P*h/sigma^2) [bps/Hz]"""
    snr = P_tx * h_lin / sigma2
    return math.log2(1.0 + snr)


def compute_task_metrics(L_bits: float, h_dB: float = FIXED_CHANNEL_GAIN_DB) -> Dict[str, Any]:
    """
    计算任务的本地和卸载指标

    参数:
    - L_bits: 任务数据量（bits）
    - h_dB: 信道增益（dB）

    返回:
    - metrics: 包含所有计算指标的字典
    """
    # 信道增益
    h_lin = channel_gain_from_db(h_dB)

    # 上行和下行速率
    R_ul = rate_bps_per_hz(P_k_ul, h_lin)  # bps/Hz
    R_dl = rate_bps_per_hz(P_u_dl, h_lin)  # bps/Hz

    # === 卸载模式 ===
    # 传输时间
    t_ul = L_bits / (B * R_ul)  # 上行传输时间
    t_dl = (O_k * L_bits) / (B * R_dl)  # 下行传输时间

    # 计算时间
    t_ec = (C_per_bit * L_bits) / f_c  # 边缘计算时间
    t_mec = t_ul + t_ec + t_dl         # 卸载总时延

    # UAV能耗
    E_u_ul = P_u_ul * t_ul  # UAV上行接收电路
    E_u_dl = P_u_dl * t_dl  # UAV下行发射
    E_u_ec = gamma_c * C_per_bit * L_bits * (f_c ** 2)  # UAV边缘计算
    E_u_total = E_u_ul + E_u_dl + E_u_ec

    # IMD卸载模式能耗
    E_k_ul_tx = P_k_ul * t_ul  # IMD上行发射
    E_k_dl_rx = P_k_dl * t_dl  # IMD下行接收

    # === 本地模式 ===
    t_loc = (C_per_bit * L_bits) / f_loc  # 本地计算时间
    E_loc = gamma_loc * C_per_bit * L_bits * (f_loc ** 2)  # 本地计算能耗

    return {
        "channel_quality": {
            "h_dB": float(h_dB),
            "h_lin": float(h_lin),
            "R_ul_bpsHz": float(R_ul),
            "R_dl_bpsHz": float(R_dl),
            "throughput_ul_Mbps": float(B * R_ul / 1e6),
            "throughput_dl_Mbps": float(B * R_dl / 1e6),
        },
        "local_compute": {
            "t_loc_s": float(t_loc),
            "E_loc_J": float(E_loc),
        },
        "offload_compute": {
            "t_ul_s": float(t_ul),
            "t_ec_s": float(t_ec),
            "t_dl_s": float(t_dl),
            "t_mec_s": float(t_mec),
            "E_u_ul_J": float(E_u_ul),
            "E_u_dl_J": float(E_u_dl),
            "E_u_ec_J": float(E_u_ec),
            "E_u_total_J": float(E_u_total),
            "E_k_ul_tx_J": float(E_k_ul_tx),
            "E_k_dl_rx_J": float(E_k_dl_rx),
        }
    }


def generate_gantt_chart(
    period: float,
    t_loc: float,
    t_mec: float,
    T_max: float,
    total_time: float = 1800.0,
    non_trigger_ratio: float = 0.4,  # 新增参数：不触发比例
) -> List[Dict[str, Any]]:
    """
    生成任务的甘特图数据（精简版）

    新增功能：引入任务触发的稀疏性，40%的时间不会触发任务，且不触发的周期连续分布

    对于触发的周期，记录：
      - trigger_time
      - local_complete_time
      - local_success_or_not（考虑可靠性）
      - local_completion_rate（成功=1，失败在[0.05,0.10]）
      - deadline_time
      - latest_offload_arrival_time
      - earliest_offload_arrival_time
      - uav_offloaded, uav_id, uav_arrival_time, uav_complete_time（初始化为未卸载）

    对于不触发的周期：
      - trigger_time设置为9999
      - 其他字段设置为相应的默认值
    """
    # 首先按原来的逻辑生成完整的甘特图（假设都触发）
    gantt_chart: List[Dict[str, Any]] = []
    current_time = 0.0
    all_periods = []  # 存储所有周期的信息

    # 只看时延约束下，本地/卸载能否完成一次
    can_complete_local_time = (t_loc <= period) and (t_loc <= T_max)
    can_complete_offload_time = (t_mec <= period) and (t_mec <= T_max)

    # 若本地能完成且 t_mec < t_loc，则在 [trigger, trigger + (t_loc - t_mec)) 内
    # UAV 到达并卸载，完成时间会早于本地完成
    if can_complete_local_time and (t_mec < t_loc):
        offload_better_window_offset = t_loc - t_mec
    else:
        offload_better_window_offset = None

    # 若卸载能完成，则在 [trigger, trigger + (T_max - t_mec)] 内到达还能赶上 deadline
    if can_complete_offload_time:
        latest_offload_arrival_offset = T_max - t_mec
    else:
        latest_offload_arrival_offset = None

    # 第一步：生成所有可能的周期信息
    while current_time < total_time:
        trigger_time = current_time
        deadline_time = trigger_time + T_max
        next_trigger_time = trigger_time + period
        has_next_period = next_trigger_time < total_time

        # 本地完成时间（纯时间轴意义上）
        local_complete_time = trigger_time + t_loc

        # === 本地成功/失败 + completion_rate ===
        if can_complete_local_time:
            # 满足时延约束，再叠加可靠性
            success_flag = 1 if np.random.rand() < LOCAL_SUCCESS_PROB else 0
        else:
            # 时延本身无法满足，必然失败
            success_flag = 0

        if success_flag == 1:
            local_completion_rate = 1.0
        else:
            local_completion_rate = float(
                np.random.uniform(LOCAL_FAIL_COMPLETION_LOW, LOCAL_FAIL_COMPLETION_HIGH)
            )

        # === 卸载时间窗口 ===
        if offload_better_window_offset is not None:
            earliest_offload_arrival_time = float(trigger_time)
        else:
            earliest_offload_arrival_time = None

        if latest_offload_arrival_offset is not None:
            latest_offload_arrival_time = float(trigger_time + latest_offload_arrival_offset)
        else:
            latest_offload_arrival_time = None

        # 存储这个周期的完整信息
        period_info = {
            "trigger_time": float(trigger_time),
            "local_complete_time": float(local_complete_time),
            "local_success_or_not": int(success_flag),
            "local_completion_rate": float(local_completion_rate),
            "deadline_time": float(deadline_time),
            "latest_offload_arrival_time": latest_offload_arrival_time,
            "earliest_offload_arrival_time": earliest_offload_arrival_time,
            "uav_offloaded": 0,
            "uav_id": -1,
            "uav_arrival_time": -1.0,
            "uav_complete_time": -1.0,
        }

        all_periods.append(period_info)

        if not has_next_period:
            break

        current_time = next_trigger_time

    # 第二步：确定不触发的连续区间
    total_periods = len(all_periods)
    if total_periods == 0:
        return gantt_chart

    # 计算需要置为不触发的周期数
    non_trigger_count = int(round(total_periods * non_trigger_ratio))

    if non_trigger_count > 0:
        # 随机选择连续不触发区间的起始位置
        start_index = np.random.randint(0, total_periods)

        # 创建不触发周期的索引集合（考虑环形连续）
        non_trigger_indices = set()
        for i in range(non_trigger_count):
            index = (start_index + i) % total_periods
            non_trigger_indices.add(index)
    else:
        non_trigger_indices = set()

    # 第三步：生成最终的甘特图，对不触发的周期进行特殊处理
    for i, period_info in enumerate(all_periods):
        if i in non_trigger_indices:
            # 不触发的周期：trigger_time设置为9999，其他字段保持但含义不同
            non_trigger_entry = {
                "trigger_time": 9999.0,  # 标记为不触发
                "local_complete_time": period_info["local_complete_time"],  # 保持原计算值供参考
                "local_success_or_not": 0,  # 不触发就是0
                "local_completion_rate": 0.0,  # 不触发完成率为0
                "deadline_time": period_info["deadline_time"],  # 保持原计算值供参考
                "latest_offload_arrival_time": period_info["latest_offload_arrival_time"],
                "earliest_offload_arrival_time": period_info["earliest_offload_arrival_time"],
                "uav_offloaded": 0,
                "uav_id": -1,
                "uav_arrival_time": -1.0,
                "uav_complete_time": -1.0,
            }
            gantt_chart.append(non_trigger_entry)
        else:
            # 触发的周期：保持原有逻辑
            gantt_chart.append(period_info)

    return gantt_chart


def classify_point_mode(
    t_loc: float,
    t_mec: float,
    period: float,
    T_max: float,
) -> str:
    """
    按照全局关系对点进行分类，方便后续在策略中快速筛选：

    - "local_always_success":
        每个周期本地都能在周期内且不违反 T_max 完成任务（只考虑时延）
    - "local_always_timeout_offload_ok":
        本地总是超时，但通过卸载可以在周期和 T_max 内完成
    - "both_timeout":
        连卸载也满足不了（当前参数下一般比较少见）
    """
    can_local = (t_loc <= period) and (t_loc <= T_max)
    can_offload = (t_mec <= period) and (t_mec <= T_max)

    if can_local:
        return "local_always_success"
    if (not can_local) and can_offload:
        return "local_always_timeout_offload_ok"
    return "both_timeout"


# =====================================================================
# 任务参数随机初始化器
# =====================================================================

class TaskInitializer:
    """为PathPlanner中的所有点初始化随机任务参数"""

    def __init__(self, seed: int = 42):
        """
        参数:
        - seed: 随机种子，用于结果复现
        """
        self.seed = seed
        np.random.seed(seed)
        self.task_data: Dict[int, Dict[str, Any]] = {}

    def initialize_all_tasks(self, path_planner, non_trigger_ratio: float = NON_TRIGGER_RATIO) -> Dict[int, Dict[str, Any]]:
        """
        为PathPlanner中的所有点初始化任务

        参数:
        - path_planner: PathPlanner实例
        - non_trigger_ratio: 不触发任务的比例（默认0.4，即40%）

        返回:
        - task_data: 包含所有点任务数据的字典
        """
        print(f"\n{'='*80}")
        print(f"开始初始化任务参数（随机种子={self.seed}，不触发比例={non_trigger_ratio*100:.1f}%）")
        print(f"{'='*80}")

        # 获取所有点的ID
        all_point_ids = list(path_planner.id_to_point.keys())
        total_points = len(all_point_ids)

        print(f"总点数: {total_points}")
        print(f"正在生成任务参数（仅对 Green / Yellow / Red 点生成任务）...")

        # 统计一下各类点的数量
        color_counter = {
            "Green": 0,
            "Yellow": 0,
            "Red": 0,
            "other": 0,
        }

        for idx, point_id in enumerate(all_point_ids):
            if (idx + 1) % 1000 == 0 or idx == 0 or idx == total_points - 1:
                print(f"  处理进度: {idx+1}/{total_points} ({100*(idx+1)/total_points:.1f}%)")

            # 获取点的基本信息
            point_info = path_planner.id_to_point[point_id]
            coords = point_info["coords"]          # [x, y, z]
            area_name = point_info["area_name"]
            # 通过 PathPlanner 的映射拿颜色
            color_class = path_planner.id_to_classname.get(point_id, "unknown")

            # 只为 Green / Yellow / Red 生成任务，其它颜色直接跳过
            if color_class not in ("Green", "Yellow", "Red"):
                color_counter["other"] += 1
                continue
            else:
                color_counter[color_class] += 1

            # ================= 从这里开始是原来的任务生成逻辑 =================

            # 随机生成任务参数
            # L_bits = float(np.random.uniform(2.5e6, 50e6))  # 2.5-50 Mbit
            # L_bits = float(np.random.uniform(2.5e6, 10e6))  # 2.5-10 Mbit
            L_bits = float(np.random.uniform(40e6, 50e6))  # 40-50 Mbit
            # period = int(np.random.randint(32, 65))          # 【32，64】秒之间的整数
            # period = int(np.random.randint(12, 25))          # 【12，25】秒之间的整数
            period = int(np.random.randint(64, 121))          # 【64，120】秒之间的整数

            # 计算本地和卸载的指标
            metrics = compute_task_metrics(L_bits)

            t_loc = metrics["local_compute"]["t_loc_s"]
            t_mec = metrics["offload_compute"]["t_mec_s"]

            # 计算容许时延 T_max = min(t_loc / random(0.75, 0.8), period)
            random_factor = float(np.random.uniform(0.75, 0.8))
            T_max = float(min(t_loc / random_factor, period))

            # 生成甘特图（包含稀疏触发逻辑）
            gantt_chart = generate_gantt_chart(
                period=period,
                t_loc=t_loc,
                t_mec=t_mec,
                T_max=T_max,
                non_trigger_ratio=non_trigger_ratio,  # 传入不触发比例
            )

            # 统计信息（需要区分触发和不触发的任务）
            total_periods = len(gantt_chart)
            triggered_tasks = [g for g in gantt_chart if g["trigger_time"] != 9999.0]
            non_triggered_tasks = [g for g in gantt_chart if g["trigger_time"] == 9999.0]

            total_triggers = len(triggered_tasks)
            total_non_triggers = len(non_triggered_tasks)

            # 按实际模拟后的本地成功次数来统计（只统计触发的任务）
            local_completions = sum(g["local_success_or_not"] for g in triggered_tasks)

            # 卸载能否在时延上完成是确定的（只依赖 t_mec、T_max 和 period）
            can_offload_time = (t_mec <= period) and (t_mec <= T_max)
            offload_completions = total_triggers if can_offload_time else 0

            # 点的全局模式分类：本地总能完成 / 本地总是超时但卸载可行 / 都不行
            point_mode = classify_point_mode(t_loc, t_mec, period, T_max)

            # 保存该点的所有数据
            self.task_data[point_id] = {
                "basic_info": {
                    "point_id": int(point_id),
                    "coords": [float(coords[0]), float(coords[1]), float(coords[2])],
                    "area_name": str(area_name),
                    "color_class": str(color_class),
                },
                "task_params": {
                    "L_bits": float(L_bits),
                    "L_Mbits": float(L_bits / 1e6),
                    "period_s": int(period),
                    "T_max_s": float(T_max),
                    "random_factor_for_Tmax": float(random_factor),
                    "non_trigger_ratio": float(non_trigger_ratio),  # 新增：记录不触发比例
                },
                "channel_quality": metrics["channel_quality"],
                "local_compute": metrics["local_compute"],
                "offload_compute": metrics["offload_compute"],
                "gantt_chart": gantt_chart,
                "statistics": {
                    "total_periods": int(total_periods),  # 总周期数（包括触发和不触发）
                    "total_triggers": int(total_triggers),  # 实际触发的任务数
                    "total_non_triggers": int(total_non_triggers),  # 不触发的任务数
                    "local_completions": int(local_completions),
                    "offload_completions": int(offload_completions),
                    "actual_non_trigger_ratio": (
                        float(total_non_triggers / total_periods)
                        if total_periods > 0 else 0.0
                    ),  # 实际的不触发比例
                    "local_success_rate": (
                        float(local_completions / total_triggers)
                        if total_triggers > 0 else 0.0
                    ),
                    "offload_success_rate": (
                        float(offload_completions / total_triggers)
                        if total_triggers > 0 else 0.0
                    ),
                    # 新增：方便后续按类型快速筛点
                    "point_mode": point_mode,
                },
            }

        print("\n颜色统计（仅这三类会生成任务）:")
        print(f"  Green 点数:  {color_counter['Green']}")
        print(f"  Yellow 点数: {color_counter['Yellow']}")
        print(f"  Red 点数:    {color_counter['Red']}")
        print(f"  其它颜色点数(未生成任务): {color_counter['other']}")

        print(f"\n任务初始化完成！（总任务点数 = {len(self.task_data)}）")
        print(f"{'='*80}\n")

        return self.task_data

    def save_to_json(self, output_path: str) -> None:
        """
        保存任务数据到JSON文件

        参数:
        - output_path: 输出文件路径
        """
        print(f"正在保存任务数据到: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.task_data, f, indent=2, ensure_ascii=False)

        print(f"保存完成！文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    def print_summary_statistics(self) -> None:
        """打印任务数据的统计摘要"""
        if not self.task_data:
            print("没有任务数据可供统计")
            return

        print(f"\n{'='*80}")
        print(f"任务数据统计摘要")
        print(f"{'='*80}")

        # 基本统计
        total_points = len(self.task_data)
        print(f"\n总点数: {total_points}")

        # 任务参数统计
        L_bits_list = [data["task_params"]["L_Mbits"] for data in self.task_data.values()]
        period_list = [data["task_params"]["period_s"] for data in self.task_data.values()]
        T_max_list = [data["task_params"]["T_max_s"] for data in self.task_data.values()]

        print(f"\n任务数据量分布 (Mbit):")
        print(f"  最小值: {min(L_bits_list):.2f}")
        print(f"  最大值: {max(L_bits_list):.2f}")
        print(f"  平均值: {np.mean(L_bits_list):.2f}")
        print(f"  中位数: {np.median(L_bits_list):.2f}")

        print(f"\n任务周期分布 (秒):")
        print(f"  最小值: {min(period_list)}")
        print(f"  最大值: {max(period_list)}")
        print(f"  平均值: {np.mean(period_list):.2f}")
        print(f"  中位数: {np.median(period_list):.1f}")

        print(f"\n容许时延分布 (秒):")
        print(f"  最小值: {min(T_max_list):.2f}")
        print(f"  最大值: {max(T_max_list):.2f}")
        print(f"  平均值: {np.mean(T_max_list):.2f}")
        print(f"  中位数: {np.median(T_max_list):.2f}")

        # 时延统计
        t_loc_list = [data["local_compute"]["t_loc_s"] for data in self.task_data.values()]
        t_mec_list = [data["offload_compute"]["t_mec_s"] for data in self.task_data.values()]

        print(f"\n本地计算时延分布 (秒):")
        print(f"  最小值: {min(t_loc_list):.2f}")
        print(f"  最大值: {max(t_loc_list):.2f}")
        print(f"  平均值: {np.mean(t_loc_list):.2f}")
        print(f"  中位数: {np.median(t_loc_list):.2f}")

        print(f"\n卸载计算时延分布 (秒):")
        print(f"  最小值: {min(t_mec_list):.2f}")
        print(f"  最大值: {max(t_mec_list):.2f}")
        print(f"  平均值: {np.mean(t_mec_list):.2f}")
        print(f"  中位数: {np.median(t_mec_list):.2f}")

        # 触发统计（新增）
        total_periods_list = [data["statistics"]["total_periods"] for data in self.task_data.values()]
        total_triggers_list = [data["statistics"]["total_triggers"] for data in self.task_data.values()]
        actual_non_trigger_ratios = [data["statistics"]["actual_non_trigger_ratio"] for data in self.task_data.values()]

        print(f"\n任务触发统计:")
        print(f"  平均总周期数: {np.mean(total_periods_list):.1f}")
        print(f"  平均触发任务数: {np.mean(total_triggers_list):.1f}")
        print(f"  平均实际不触发比例: {np.mean(actual_non_trigger_ratios)*100:.1f}%")
        print(f"  不触发比例标准差: {np.std(actual_non_trigger_ratios)*100:.2f}%")

        # 完成率统计（基于模拟结果，只统计触发的任务）
        local_success_rates = [data["statistics"]["local_success_rate"]
                              for data in self.task_data.values()]
        offload_success_rates = [data["statistics"]["offload_success_rate"]
                                for data in self.task_data.values()]

        print(f"\n本地计算成功率（仅触发任务，含可靠性）:")
        print(f"  平均值: {np.mean(local_success_rates)*100:.2f}%")
        print(f"  始终成功的点数: {sum(1 for r in local_success_rates if r == 1.0)}")
        print(f"  始终失败的点数: {sum(1 for r in local_success_rates if r == 0.0)}")

        print(f"\n卸载计算成功率（仅触发任务，只看时延可行性）:")
        print(f"  平均值: {np.mean(offload_success_rates)*100:.2f}%")
        print(f"  始终成功的点数: {sum(1 for r in offload_success_rates if r == 1.0)}")
        print(f"  始终失败的点数: {sum(1 for r in offload_success_rates if r == 0.0)}")

        # 模式统计：本地总能完成 / 本地超时但卸载可行 / 都不行
        modes = [data["statistics"]["point_mode"] for data in self.task_data.values()]
        print(f"\n点模式统计:")
        print(f"  local_always_success: {modes.count('local_always_success')}")
        print(f"  local_always_timeout_offload_ok: {modes.count('local_always_timeout_offload_ok')}")
        print(f"  both_timeout: {modes.count('both_timeout')}")

        # 总触发次数（1800秒内）
        total_triggers = sum(data["statistics"]["total_triggers"]
                           for data in self.task_data.values())
        total_periods_all = sum(data["statistics"]["total_periods"]
                              for data in self.task_data.values())
        print(f"\n1800秒内总统计:")
        print(f"  总周期数: {total_periods_all}")
        print(f"  实际任务触发次数: {total_triggers}")
        print(f"  总体实际触发比例: {total_triggers/total_periods_all*100:.1f}%")

        print(f"\n{'='*80}\n")

    def print_sample_tasks(self, num_samples: int = 3) -> None:
        """
        打印几个示例任务的详细信息

        参数:
        - num_samples: 要打印的示例数量
        """
        if not self.task_data:
            print("没有任务数据可供展示")
            return

        print(f"\n{'='*80}")
        print(f"示例任务详情（随机选取{num_samples}个）")
        print(f"{'='*80}")

        sample_ids = np.random.choice(list(self.task_data.keys()),
                                     size=min(num_samples, len(self.task_data)),
                                     replace=False)

        for i, point_id in enumerate(sample_ids, 1):
            data = self.task_data[point_id]

            print(f"\n{'─'*80}")
            print(f"示例 {i}: 点ID = {point_id}")
            print(f"{'─'*80}")

            # 基本信息
            print(f"区域: {data['basic_info']['area_name']}")
            print(f"颜色: {data['basic_info']['color_class']}")
            print(f"坐标: {data['basic_info']['coords']}")

            # 任务参数
            print(f"\n任务参数:")
            print(f"  数据量: {data['task_params']['L_Mbits']:.2f} Mbit")
            print(f"  周期: {data['task_params']['period_s']} 秒")
            print(f"  容许时延: {data['task_params']['T_max_s']:.2f} 秒")
            print(f"  不触发比例: {data['task_params']['non_trigger_ratio']*100:.1f}%")

            # 信道质量
            print(f"\n信道质量:")
            print(f"  信道增益: {data['channel_quality']['h_dB']:.2f} dB")
            print(f"  上行速率: {data['channel_quality']['throughput_ul_Mbps']:.2f} Mbps")
            print(f"  下行速率: {data['channel_quality']['throughput_dl_Mbps']:.2f} Mbps")

            # 本地计算
            print(f"\n本地计算:")
            print(f"  时延: {data['local_compute']['t_loc_s']:.2f} 秒")
            print(f"  能耗: {data['local_compute']['E_loc_J']:.6f} J")

            # 卸载计算
            print(f"\n卸载计算:")
            print(f"  上行时间: {data['offload_compute']['t_ul_s']:.3f} 秒")
            print(f"  计算时间: {data['offload_compute']['t_ec_s']:.3f} 秒")
            print(f"  下行时间: {data['offload_compute']['t_dl_s']:.3f} 秒")
            print(f"  总时延: {data['offload_compute']['t_mec_s']:.3f} 秒")
            print(f"  UAV总能耗: {data['offload_compute']['E_u_total_J']:.6f} J")

            # 统计
            stats = data['statistics']
            print(f"\n1800秒内统计（含稀疏触发）:")
            print(f"  点模式: {stats['point_mode']}")
            print(f"  总周期数: {stats['total_periods']}")
            print(f"  实际触发次数: {stats['total_triggers']}")
            print(f"  不触发次数: {stats['total_non_triggers']}")
            print(f"  实际不触发比例: {stats['actual_non_trigger_ratio']*100:.1f}%")
            print(f"  本地完成次数: {stats['local_completions']} "
                  f"({stats['local_success_rate']*100:.1f}%)")
            print(f"  卸载可行次数(时延上): {stats['offload_completions']} "
                  f"({stats['offload_success_rate']*100:.1f}%)")

            # 甘特图示例（前5个周期，包括触发和不触发）
            print(f"\n甘特图（前5个周期，包括触发/不触发）:")
            for j, gantt in enumerate(data['gantt_chart'][:5], 1):
                trigger_time = gantt['trigger_time']
                if trigger_time == 9999.0:
                    print(f"  周期{j}: 【不触发】trigger_time=9999")
                else:
                    print(f"  周期{j}: 【触发】T={trigger_time:.1f}s", end="")
                    print(f" | 本地完成时间={gantt['local_complete_time']:.2f}s", end="")
                    print(f" | 本地成功={gantt['local_success_or_not']}", end="")
                    print(f" | completion_rate={gantt['local_completion_rate']:.2f}", end="")
                    print(f" | 截止@{gantt['deadline_time']:.1f}s", end="")
                    if gantt["earliest_offload_arrival_time"] is not None:
                        print(f" | earliest_offload={gantt['earliest_offload_arrival_time']:.2f}s", end="")
                    if gantt["latest_offload_arrival_time"] is not None:
                        print(f" | latest_offload={gantt['latest_offload_arrival_time']:.2f}s", end="")
                    print(f" | uav_offloaded={gantt['uav_offloaded']}")

        print(f"\n{'='*80}\n")


# =====================================================================
# 主函数示例
# =====================================================================

def main():
    """示例：如何使用TaskInitializer"""

    from read_map_and_get_path_for_benchMARL import PathPlanner

    # 2025.11.19 调整了任务本地成功率和任务周期，期待着100个点5个UAV 和 500个点2个UAV能有不错的表现
    # areaid_map_json = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_100points\UE_map_encode_for_env_design\areaid_mapping.json"
    # base_path = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_100points\UE_map_encode_for_env_design"
    # output_json = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_100points\task_initialization_data_v2.json"
    # output_json = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_100points\task_initialization_data_example.json"

    # areaid_map_json = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_200points\UE_map_encode_for_env_design\areaid_mapping.json"
    # base_path = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_200points\UE_map_encode_for_env_design"
    # output_json = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_200points\task_initialization_data_v2.json"
    # output_json = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_200points\task_initialization_data_smallData_shortPeriod.json"
    # output_json = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_200points\task_initialization_data_largeData_longPeriod.json"

    # areaid_map_json = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_300points\UE_map_encode_for_env_design\areaid_mapping.json"
    # base_path = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_300points\UE_map_encode_for_env_design"
    # output_json = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_300points\task_initialization_data_v2.json"

    # areaid_map_json = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_400points\UE_map_encode_for_env_design\areaid_mapping.json"
    # base_path = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_400points\UE_map_encode_for_env_design"
    # output_json = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_400points\task_initialization_data_v2.json"

    # areaid_map_json = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_all_points\UE_map_encode_for_env_design\areaid_mapping.json"
    # base_path = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_all_points\UE_map_encode_for_env_design"
    # output_json = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_all_points\task_initialization_data_v2.json"

    # areaid_map_json = r"L:\python_projects\AirSim-main\PythonClient\multirotor\reconst\10\UE_map_encode\areaid_mapping.json"
    # base_path = r"L:\python_projects\AirSim-main\PythonClient\multirotor\reconst\10\UE_map_encode"
    # output_json = r"L:\python_projects\AirSim-main\PythonClient\multirotor\reconst\10\task_initialization_data.json"
    # output_json = r"L:\python_projects\AirSim-main\PythonClient\multirotor\reconst\10\task_initialization_data_short.json"
    # output_json = r"L:\python_projects\AirSim-main\PythonClient\multirotor\reconst\10\task_initialization_data_long.json"

    # areaid_map_json = r"L:\python_projects\AirSim-main\PythonClient\multirotor\reconst\20\UE_map_encode\areaid_mapping.json"
    # base_path = r"L:\python_projects\AirSim-main\PythonClient\multirotor\reconst\20\UE_map_encode"
    # output_json = r"L:\python_projects\AirSim-main\PythonClient\multirotor\reconst\20\task_initialization_data.json"

    # areaid_map_json = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario1\UE_map_encode\areaid_mapping.json"
    # base_path = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario1\UE_map_encode"
    # output_json = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario1\task_initialization_data.json"  #默认按 long 来计算

    # areaid_map_json = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario2\UE_map_encode\areaid_mapping.json"
    # base_path = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario2\UE_map_encode"
    # output_json = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario2\task_initialization_data.json"  #默认按 long 来计算

    areaid_map_json = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario3\UE_map_encode\areaid_mapping.json"
    base_path = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario3\UE_map_encode"
    output_json = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario3\task_initialization_data.json"  #默认按 long 来计算

    # 创建PathPlanner
    print("正在初始化PathPlanner...")
    path_planner = PathPlanner(base_path=base_path, areaid_map_json=areaid_map_json)

    # 创建任务初始化器（可以设置不同的seed）
    task_initializer = TaskInitializer(seed=42)

    # 初始化所有任务，使用40%的不触发比例
    task_data = task_initializer.initialize_all_tasks(path_planner)

    # 打印统计摘要
    task_initializer.print_summary_statistics()

    # 打印示例任务
    task_initializer.print_sample_tasks(num_samples=3)

    # 保存到JSON文件
    task_initializer.save_to_json(output_json)

    print(f"\n完成！任务数据已保存到: {output_json}")
    print(f"可以在MARL环境中加载该文件使用")
    print(f"新增功能：40%的任务周期不会触发（trigger_time=9999）")

    return task_initializer, task_data


if __name__ == "__main__":
    task_initializer, task_data = main()