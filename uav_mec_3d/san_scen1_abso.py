from dataclasses import dataclass, MISSING
from torchrl.envs import EnvBase
from torchrl.data import Categorical, Unbounded, Composite, Binary
from tensordict import TensorDict
import torch
import json
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import math
import random
import threading
import copy
import numpy as np
from pathlib import Path
from .get_points_and_paths import PathPlanner, BidirectionalPathManager


@dataclass
class TaskConfig:
    placeholder: int = MISSING


class San_scen1_abso_Env(EnvBase):
    """
    基于邻近点的多UAV调度MARL环境（使用绝对坐标版本）

    核心特性：
    1. 每个UAV观测其邻近的5个点（从distances_mapping获取）
    2. 使用绝对坐标系统：观测包含实际的绝对坐标
    3. 观测包含：自身绝对坐标 + 其他UAV绝对坐标 + 5个邻近点的绝对坐标+路径奖励 + 原地停留奖励
    4. 动作空间：选择5个邻近点中的一个(0-4)，或原地停留(5)
    """

    # 类变量：所有实例共享的数据
    _shared_data_loaded = False
    _path_planner = None
    _path_manager = None
    _task_data = None
    _distances_mapping = None
    _data_lock = threading.Lock()

    @classmethod
    def _load_shared_data(cls):
        """加载所有环境实例共享的数据，只在第一次调用时执行"""
        if cls._shared_data_loaded:
            return

        with cls._data_lock:
            if cls._shared_data_loaded:
                return

            print("Loading shared environment data...")

            try:
                # 路径配置
                base_path = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario1\UE_map_encode"
                areaid_map_json = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario1\UE_map_encode\areaid_mapping.json"
                bidire_jsonl_path = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario1\bidirection_dictionary.jsonl"

                # 加载路径规划器
                cls._path_planner = PathPlanner(base_path=base_path, areaid_map_json=areaid_map_json)
                cls._path_manager = BidirectionalPathManager(bidire_jsonl_path)

                # 加载任务数据
                with open(
                        r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario1\task_initialization_data.json",
                        'r', encoding='utf-8') as f:
                    cls._task_data = json.load(f)

                with open(
                        r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario1\all_nearest_points.json",
                        'r', encoding='utf-8') as f:
                    cls._distances_mapping = json.load(f)

                # 收集所有区域名
                cls._all_areas = sorted({pdata["basic_info"]["area_name"]
                                         for pdata in cls._task_data.values()})

                cls._shared_data_loaded = True
                print("Shared environment data loaded successfully!")

            except Exception as e:
                print(f"Error loading shared data: {e}")
                raise

    def __init__(
            self,
            uav_starts: List[int] = [20001, 40003],
            uav_speed: float = 18.0,
            max_time: float = 1800.0,
            device: str = "cpu",
            placeholder: int = 0,
    ):
        super().__init__(device=device, batch_size=torch.Size([]))

        # 确保共享数据已加载
        self._load_shared_data()

        # 使用共享数据的引用
        self.path_planner = self._path_planner
        self.path_manager = self._path_manager
        self.task_data = copy.deepcopy(self._task_data)
        self.processed_tasks = set()
        self.distances_mapping = self._distances_mapping
        self.last_episode_metrics = None

        self.uav_starts = list(uav_starts)
        self.num_uavs = len(uav_starts)
        self.uav_speed = float(uav_speed)
        self.max_time = float(max_time)
        self.hover_window = 15.0  # 固定的15秒悬停窗口

        # UAV能耗参数（仅用于最终指标计算）
        self.P_fly = 101.608126  # 飞行功率(W)
        self.P_hover = 221.160000  # 悬停功率(W)

        # 观测和动作维度 - 使用绝对坐标版本
        # 观测：自身绝对坐标(3) + 其他UAV绝对坐标(3*num_uavs) + 5个邻近点信息(每个4维：3绝对坐标+1路径奖励) + 1个原地停留奖励
        self.obs_dim = 3 + 3 * self.num_uavs + 5 * 4 + 1
        self.act_dim = 6  # 5个邻近点 + 1个原地停留

        # agent group
        self.group_map = {"agents": [f"uav_{i}" for i in range(self.num_uavs)]}

        # ===== UAV状态 =====
        self.uav_points = torch.zeros(self.num_uavs, dtype=torch.long, device=self.device)
        self.uav_coords = torch.zeros(self.num_uavs, 3, dtype=torch.float32, device=self.device)
        self.uav_times = torch.zeros(self.num_uavs, dtype=torch.float32, device=self.device)
        self.uav_energies = torch.zeros(self.num_uavs, dtype=torch.float32, device=self.device)

        # ===== 邻近点相关 =====
        # 每个UAV的5个邻近点信息
        self.uav_nearby_points = torch.zeros((self.num_uavs, 5), dtype=torch.long, device=self.device)
        self.uav_nearby_coords = torch.zeros((self.num_uavs, 5, 3), dtype=torch.float32, device=self.device)
        self.uav_nearby_path_rewards = torch.zeros((self.num_uavs, 5), dtype=torch.float32, device=self.device)

        # ===== 原地停留奖励 =====
        self.uav_stay_rewards = torch.zeros(self.num_uavs, dtype=torch.float32, device=self.device)

        # ===== 等待时长统计 =====
        self.uav_total_waiting_times = torch.zeros(self.num_uavs, dtype=torch.float32, device=self.device)

        # ===== 奖励设计参数 =====
        self.w_task = 100.0  # 每完成一个任务的基础奖励（提高到100分）
        self.w_time_penalty = 0.001  # 轻微时间惩罚

        # 基准能耗
        self.baseline_energy_per_uav = 450000.0
        self.baseline_energy_total = self.baseline_energy_per_uav * self.num_uavs

        # 任务完成统计
        self.completed_tasks_count = 0
        self.total_episode_time = 0.0

        # 区域统计（仅用于最终指标）
        self.area_served_bits = defaultdict(float)
        self.total_served_bits = 0.0
        self.all_areas = self._all_areas

        self.max_steps = 10_000_000

        # ===== 轨迹追踪 =====
        self.trajectory_history = []  # 存储每个step的轨迹信息
        self.current_step = 0

        self._make_spec()

    def _make_spec(self):
        """定义环境的spec"""
        self.observation_spec = Composite(
            agents=Composite(
                observation=Unbounded(
                    shape=(self.num_uavs, self.obs_dim),
                    device=self.device,
                ),
                device=self.device,
                shape=torch.Size([self.num_uavs]),
            ),
            shape=torch.Size([]),
            device=self.device,
        )

        self.action_spec = Composite(
            agents=Composite(
                action=Categorical(
                    n=self.act_dim,
                    shape=(self.num_uavs,),
                    device=self.device,
                ),
                device=self.device,
                shape=torch.Size([self.num_uavs]),
            ),
            shape=torch.Size([]),
            device=self.device,
        )

        self.reward_spec = Composite(
            agents=Composite(
                reward=Unbounded(
                    shape=(self.num_uavs, 1),
                    device=self.device,
                ),
                device=self.device,
                shape=torch.Size([self.num_uavs]),
            ),
            shape=torch.Size([]),
            device=self.device,
        )

        self.done_spec = Composite(
            done=Binary(
                shape=(1,),
                dtype=torch.bool,
                device=self.device,
            ),
            terminated=Binary(
                shape=(1,),
                dtype=torch.bool,
                device=self.device,
            ),
            shape=torch.Size([]),
            device=self.device,
        )

    def _update_uav_coordinates(self):
        """更新UAV的坐标（绝对坐标）"""
        for i in range(self.num_uavs):
            point_id = int(self.uav_points[i].item())
            try:
                coords, _ = self.path_planner.coords_from_global_id(point_id)
                # 转换为米并归一化到合理范围
                normalized_coords = [coord / 100000.0 for coord in coords]  # 厘米转为合理单位
                self.uav_coords[i] = torch.tensor(normalized_coords, dtype=torch.float32, device=self.device)
            except:
                # 如果无法获取坐标，使用默认值
                self.uav_coords[i] = torch.zeros(3, dtype=torch.float32, device=self.device)

    def _update_nearby_points_info(self):
        """更新每个UAV的邻近点信息"""
        for i in range(self.num_uavs):
            current_point_id = int(self.uav_points[i].item())
            current_point_str = str(current_point_id)

            # 获取邻近点信息
            if current_point_str in self.distances_mapping:
                nearest_points = self.distances_mapping[current_point_str]["nearest_points"]
                # 取前5个邻近点
                nearby_5 = nearest_points[:5]

                for j, neighbor in enumerate(nearby_5):
                    neighbor_id = neighbor["id"]
                    self.uav_nearby_points[i, j] = neighbor_id

                    # 获取邻近点坐标
                    try:
                        coords, _ = self.path_planner.coords_from_global_id(neighbor_id)
                        normalized_coords = [coord / 100000.0 for coord in coords]
                        self.uav_nearby_coords[i, j] = torch.tensor(
                            normalized_coords, dtype=torch.float32, device=self.device
                        )
                    except:
                        self.uav_nearby_coords[i, j] = torch.zeros(3, dtype=torch.float32, device=self.device)

                # 如果邻近点不足5个，用当前点填充
                while len(nearby_5) < 5:
                    j = len(nearby_5)
                    self.uav_nearby_points[i, j] = current_point_id
                    self.uav_nearby_coords[i, j] = self.uav_coords[i].clone()
                    nearby_5.append({"id": current_point_id, "distance": 0.0})
            else:
                # 如果没有邻近点信息，全部设为当前点
                for j in range(5):
                    self.uav_nearby_points[i, j] = current_point_id
                    self.uav_nearby_coords[i, j] = self.uav_coords[i].clone()

    def _update_nearby_path_rewards(self):
        """更新每个UAV到其邻近点的路径奖励（包含时间折扣）"""
        for i in range(self.num_uavs):
            current_point = int(self.uav_points[i].item())

            for j in range(5):
                target_point = int(self.uav_nearby_points[i, j].item())
                reward = self._calculate_path_reward(i, current_point, target_point)
                self.uav_nearby_path_rewards[i, j] = reward

            # 计算原地停留奖励
            self.uav_stay_rewards[i] = self._calculate_stay_reward(i, current_point)

    def _calculate_path_reward(self, uav_idx: int, start_id: int, end_id: int) -> float:
        """
        计算从start_id到end_id的路径预期收益（带时间折扣）
        奖励 = 任务奖励 / 总用时
        """
        if start_id == end_id:
            return 0.0  # 如果起点终点相同，返回0，用原地停留动作处理

        # 获取路径信息
        path_data = self.path_manager.get_path(start_id, end_id)
        if path_data is None:
            return -1.0  # 无法找到路径的惩罚

        current_time = float(self.uav_times[uav_idx].item())

        # 估算到达时间
        total_distance_cm = path_data["path"][-1]["prefix_sum_path_length"]
        total_distance_m = total_distance_cm / 100.0
        estimated_travel_time = total_distance_m / self.uav_speed
        estimated_arrival_time = current_time + estimated_travel_time

        # 只在最终目的地处理任务
        window_start = estimated_arrival_time
        window_end = estimated_arrival_time + self.hover_window

        end_str = str(end_id)
        if end_str not in self.task_data:
            return 0.0

        valid_tasks = self._get_tasks_in_window(end_str, window_start, window_end)
        task_reward = len(valid_tasks) * self.w_task  # 任务基础奖励

        # 计算总用时（飞行时间 + 任务处理时间）
        total_time = estimated_travel_time + self.hover_window

        if total_time <= 0:
            return 0.0

        # 时间折扣奖励：任务奖励 / 用时
        time_discounted_reward = task_reward / total_time

        # 减去时间成本
        time_cost = total_time * self.w_time_penalty

        return max(0.0, time_discounted_reward - time_cost)

    def _calculate_stay_reward(self, uav_idx: int, point_id: int) -> float:
        """
        计算在当前点原地停留15秒的奖励（带时间折扣）
        奖励 = 任务奖励 / 停留时间(15秒)
        """
        current_time = float(self.uav_times[uav_idx].item())
        window_start = current_time
        window_end = current_time + self.hover_window

        point_str = str(point_id)
        if point_str not in self.task_data:
            return 0.0

        valid_tasks = self._get_tasks_in_window(point_str, window_start, window_end)
        task_reward = len(valid_tasks) * self.w_task

        # 时间折扣奖励：任务奖励 / 停留时间
        if self.hover_window <= 0:
            return 0.0

        time_discounted_reward = task_reward / self.hover_window

        # 减去时间成本
        time_cost = self.hover_window * self.w_time_penalty

        return max(0.0, time_discounted_reward - time_cost)

    def _get_tasks_in_window(self, point_id: str, window_start: float, window_end: float) -> List[Dict[str, Any]]:
        """获取在时间窗口内可以处理的任务"""
        if point_id not in self.task_data:
            return []

        gantt_chart = self.task_data[point_id]["gantt_chart"]
        valid_tasks = []

        for task in gantt_chart:
            trigger_time = float(task["trigger_time"])

            # 跳过不触发的任务
            if trigger_time >= 9999.0:
                continue

            deadline_time = float(task["deadline_time"])
            task_key = (int(point_id), trigger_time)

            # 检查是否已处理
            if task.get("uav_offloaded", False) or task_key in self.processed_tasks:
                continue

            # 计算最晚到达时间
            latest_arrival = self._calculate_latest_offload_arrival_time(point_id, task)

            # 判断任务是否在窗口内可处理
            if trigger_time <= window_end and latest_arrival >= window_start:
                valid_tasks.append({
                    "task": task,
                    "trigger_time": trigger_time,
                    "latest_arrival": latest_arrival,
                    "deadline_time": deadline_time
                })

        return valid_tasks

    def _calculate_latest_offload_arrival_time(self, point_id: str, task: Dict[str, Any]) -> float:
        """计算任务的最晚可接受卸载到达时间"""
        deadline_time = float(task["deadline_time"])
        offload_info = self.task_data[point_id]["offload_compute"]
        t_mec = float(offload_info["t_mec_s"])
        return deadline_time - t_mec

    def _build_obs(self) -> TensorDict:
        """
        构建观测（使用绝对坐标系统）：
        - 自身绝对坐标(3)
        - 所有UAV绝对坐标(3*num_uavs) - 包括自己
        - 5个邻近点信息：每个包含绝对坐标(3) + 路径奖励(1) = 4维
        - 1个原地停留奖励(1)
        总维度：3 + 3*num_uavs + 5*4 + 1 = 3 + 6 + 20 + 1 = 30 (对于2个UAV)
        """
        obs = torch.zeros((self.num_uavs, self.obs_dim), dtype=torch.float32, device=self.device)

        for i in range(self.num_uavs):
            obs_idx = 0

            # 1. 自身绝对坐标(3维)
            obs[i, obs_idx:obs_idx + 3] = self.uav_coords[i]
            obs_idx += 3

            # 2. 所有UAV的绝对坐标（包括自己）
            for j in range(self.num_uavs):
                obs[i, obs_idx:obs_idx + 3] = self.uav_coords[j]
                obs_idx += 3

            # 3. 5个邻近点信息 (绝对坐标 + 路径奖励)
            for j in range(5):
                # 邻近点的绝对坐标
                obs[i, obs_idx:obs_idx + 3] = self.uav_nearby_coords[i, j]
                obs_idx += 3

                # 路径奖励
                obs[i, obs_idx] = self.uav_nearby_path_rewards[i, j]
                obs_idx += 1

            # 4. 原地停留奖励
            obs[i, obs_idx] = self.uav_stay_rewards[i]

        tensordict = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": obs,
                    },
                    batch_size=torch.Size([self.num_uavs]),
                    device=self.device,
                ),
            },
            batch_size=torch.Size([]),
            device=self.device,
        )
        return tensordict

    def _compute_action_via_path(self, uav_idx: int, start_id: int, end_id: int, current_time: float) -> Tuple[
        float, float, float]:
        """
        通过实际路径计算动作的时间、能耗和奖励（修改：只在终点执行任务，奖励按时间折扣）
        返回: (duration, total_reward, total_energy)
        """
        if start_id == end_id:
            # 这种情况不应该发生，因为移动到相同点由原地停留动作处理
            return 1e-3, 0.0, 0.0

        # 获取路径
        path_data = self.path_manager.get_path(start_id, end_id)
        if path_data is None:
            return 1e-3, -1.0, 0.0

        path_points = path_data["path"]
        accumulated_time = current_time
        accumulated_energy = 0.0
        completed_tasks = 0

        # 计算飞行时间和能耗
        total_distance_cm = path_points[-1]["prefix_sum_path_length"]
        total_distance_m = total_distance_cm / 100.0
        fly_time = total_distance_m / self.uav_speed
        fly_energy = self.P_fly * fly_time

        accumulated_energy += fly_energy
        arrival_time = current_time + fly_time

        # 只在最终目的地处理任务
        final_point_id = path_points[-1]["point_id"]
        point_str = str(final_point_id)

        window_start = arrival_time
        window_end = arrival_time + self.hover_window

        if point_str in self.task_data:
            valid_tasks = self._get_tasks_in_window(point_str, window_start, window_end)

            if valid_tasks:
                valid_tasks.sort(key=lambda x: x["trigger_time"])
                point_energy = 0.0
                last_service_end = arrival_time

                for task_info in valid_tasks:
                    service_start, service_end, energy, success = self._process_task_at_point(
                        uav_idx, point_str, task_info, last_service_end
                    )

                    if success:
                        completed_tasks += 1
                        point_energy += energy
                        last_service_end = service_end

                        if service_end >= window_end:
                            break

                accumulated_energy += point_energy
                if last_service_end > arrival_time:
                    accumulated_time = last_service_end
                else:
                    # 如果没有任务或任务很快完成，至少悬停完整窗口
                    accumulated_time = arrival_time + self.hover_window
                    accumulated_energy += self.P_hover * self.hover_window
            else:
                # 没有任务，悬停完整窗口
                accumulated_time = arrival_time + self.hover_window
                accumulated_energy += self.P_hover * self.hover_window
        else:
            # 没有任务数据，悬停完整窗口
            accumulated_time = arrival_time + self.hover_window
            accumulated_energy += self.P_hover * self.hover_window

        duration = accumulated_time - current_time
        task_reward = completed_tasks * self.w_task

        # 应用时间折扣：奖励 / 用时
        if duration > 0:
            time_discounted_reward = task_reward / duration
        else:
            time_discounted_reward = 0.0

        time_penalty = duration * self.w_time_penalty
        total_reward = time_discounted_reward - time_penalty

        return duration, total_reward, accumulated_energy

    def _compute_stay_action(self, uav_idx: int, point_id: int, current_time: float) -> Tuple[float, float, float]:
        """
        计算原地停留动作（修改：奖励按时间折扣）
        返回: (duration, total_reward, total_energy)
        """
        duration = self.hover_window
        accumulated_energy = self.P_hover * duration
        completed_tasks = 0

        point_str = str(point_id)
        if point_str in self.task_data:
            window_start = current_time
            window_end = current_time + self.hover_window

            valid_tasks = self._get_tasks_in_window(point_str, window_start, window_end)

            if valid_tasks:
                valid_tasks.sort(key=lambda x: x["trigger_time"])
                last_service_end = current_time

                for task_info in valid_tasks:
                    service_start, service_end, energy, success = self._process_task_at_point(
                        uav_idx, point_str, task_info, last_service_end
                    )

                    if success:
                        completed_tasks += 1
                        accumulated_energy += energy
                        last_service_end = service_end

                        if service_end >= window_end:
                            break

        task_reward = completed_tasks * self.w_task

        # 应用时间折扣：奖励 / 用时
        if duration > 0:
            time_discounted_reward = task_reward / duration
        else:
            time_discounted_reward = 0.0

        time_penalty = duration * self.w_time_penalty
        total_reward = time_discounted_reward - time_penalty

        return duration, total_reward, accumulated_energy

    def _process_task_at_point(self, uav_idx: int, point_id: str, task_info: Dict[str, Any],
                               arrival_time: float) -> Tuple[Optional[float], Optional[float], float, bool]:
        """处理单个任务"""
        task = task_info["task"]
        trigger_time = task_info["trigger_time"]
        latest_arrival = task_info["latest_arrival"]

        # 任务参数
        point_meta = self.task_data[point_id]
        offload_info = point_meta["offload_compute"]
        task_params = point_meta["task_params"]
        area_name = point_meta["basic_info"]["area_name"]
        L_bits = float(task_params["L_bits"])

        t_mec = float(offload_info["t_mec_s"])

        # 实际服务开始时间
        actual_start = max(arrival_time, trigger_time)

        # 检查是否在最晚到达时间之前
        if actual_start > latest_arrival:
            return None, None, 0.0, False

        # 服务结束时间
        service_end = actual_start + t_mec

        # 检查是否在deadline之前完成
        if service_end > task_info["deadline_time"]:
            return None, None, 0.0, False

        # 任务成功完成
        hover_duration = service_end - arrival_time
        hover_energy = self.P_hover * hover_duration

        # 计算计算能耗
        E_u_ul = float(offload_info["E_u_ul_J"])
        E_u_dl = float(offload_info["E_u_dl_J"])
        E_u_ec = float(offload_info["E_u_ec_J"])
        E_compute = E_u_ul + E_u_dl + E_u_ec

        total_energy = hover_energy + E_compute

        # 更新任务处理标记
        task_key = (int(point_id), trigger_time)
        self.processed_tasks.add(task_key)

        # 更新任务数据
        for gt_entry in self.task_data[point_id]["gantt_chart"]:
            if float(gt_entry["trigger_time"]) == trigger_time:
                gt_entry["uav_offloaded"] = True
                gt_entry["uav_id"] = uav_idx
                gt_entry["uav_arrival_time"] = arrival_time
                gt_entry["offload_complete_time"] = service_end
                break

        # 更新区域统计
        self.area_served_bits[area_name] += L_bits
        self.total_served_bits += L_bits

        return actual_start, service_end, total_energy, True

    def _reset(self, tensordict=None, **kwargs):
        """重置环境"""
        for i in range(self.num_uavs):
            self.uav_points[i] = int(self.uav_starts[i])

        self.uav_times.zero_()
        self.uav_energies.zero_()
        self.uav_total_waiting_times.zero_()  # 重置等待时长

        # 更新坐标
        self._update_uav_coordinates()

        self.task_data = copy.deepcopy(self._task_data)
        self.area_served_bits.clear()
        self.total_served_bits = 0.0
        self.completed_tasks_count = 0
        self.total_episode_time = 0.0

        self.processed_tasks.clear()

        # 重置轨迹追踪
        self.trajectory_history.clear()
        self.current_step = 0

        # 更新邻近点信息和路径奖励
        self._update_nearby_points_info()
        self._update_nearby_path_rewards()

        td = self._build_obs()
        td.set("done", torch.zeros(1, dtype=torch.bool, device=self.device))
        td.set("terminated", torch.zeros(1, dtype=torch.bool, device=self.device))
        return td

    def _step(self, tensordict: TensorDict):
        """执行一步 - 所有UAV同步执行（支持原地停留动作）"""
        actions = tensordict["agents"]["action"].to(self.device).long()
        rewards = torch.zeros((self.num_uavs, 1), dtype=torch.float32, device=self.device)

        # 记录当前step开始时的状态
        step_start_info = {
            "step": self.current_step,
            "timestamp_before": [float(self.uav_times[i].item()) for i in range(self.num_uavs)],
            "positions_before": [int(self.uav_points[i].item()) for i in range(self.num_uavs)],
            "coordinates_before": [self.uav_coords[i].cpu().numpy().tolist() for i in range(self.num_uavs)],
            "energies_before": [float(self.uav_energies[i].item()) for i in range(self.num_uavs)],
            "actions": [int(actions[i].item()) for i in range(self.num_uavs)],
        }

        # 存储每个UAV的动作信息
        action_durations = torch.zeros(self.num_uavs, dtype=torch.float32, device=self.device)
        action_rewards = torch.zeros(self.num_uavs, dtype=torch.float32, device=self.device)
        action_energies = torch.zeros(self.num_uavs, dtype=torch.float32, device=self.device)
        target_points = torch.zeros(self.num_uavs, dtype=torch.long, device=self.device)

        # 处理所有UAV的动作
        for i in range(self.num_uavs):
            action = int(actions[i].item())
            current_point = int(self.uav_points[i].item())
            current_time = float(self.uav_times[i].item())

            # 检查动作是否有效
            if not (0 <= action < 6):
                action_durations[i] = 1e-3
                action_rewards[i] = -1.0
                action_energies[i] = 0.0
                target_points[i] = current_point
                continue

            if action == 5:
                # 原地停留动作
                duration, total_reward, total_energy = self._compute_stay_action(
                    i, current_point, current_time
                )
                target_points[i] = current_point  # 保持在原地
            else:
                # 移动到邻近点动作 (0-4)
                target_id = int(self.uav_nearby_points[i, action].item())

                if target_id == current_point:
                    # 如果目标点就是当前点，相当于原地停留
                    duration, total_reward, total_energy = self._compute_stay_action(
                        i, current_point, current_time
                    )
                    target_points[i] = current_point
                else:
                    # 正常移动
                    duration, total_reward, total_energy = self._compute_action_via_path(
                        i, current_point, target_id, current_time
                    )
                    target_points[i] = target_id

            action_durations[i] = duration
            action_rewards[i] = total_reward
            action_energies[i] = total_energy

        # 找到最长的动作时间
        max_duration = torch.max(action_durations).item()
        if max_duration <= 0:
            max_duration = 1e-3

        # 计算每个UAV的等待时长并累加
        for i in range(self.num_uavs):
            individual_duration = action_durations[i].item()
            waiting_time = max_duration - individual_duration
            if waiting_time > 0:
                self.uav_total_waiting_times[i] += waiting_time

        # 所有UAV推进相同的时间
        for i in range(self.num_uavs):
            # 推进时间
            self.uav_times[i] += max_duration

            # 更新位置
            self.uav_points[i] = target_points[i]

            # 更新奖励
            rewards[i, 0] = action_rewards[i]

            # 累加能耗
            self.uav_energies[i] += action_energies[i]

        # 更新坐标、邻近点信息和路径奖励
        self._update_uav_coordinates()
        self._update_nearby_points_info()
        self._update_nearby_path_rewards()

        # 记录总时间
        self.total_episode_time = torch.max(self.uav_times).item()

        # 记录step结束后的状态，完成轨迹信息
        step_start_info.update({
            "timestamp_after": [float(self.uav_times[i].item()) for i in range(self.num_uavs)],
            "positions_after": [int(self.uav_points[i].item()) for i in range(self.num_uavs)],
            "coordinates_after": [self.uav_coords[i].cpu().numpy().tolist() for i in range(self.num_uavs)],
            "energies_after": [float(self.uav_energies[i].item()) for i in range(self.num_uavs)],
            "rewards": [float(rewards[i, 0].item()) for i in range(self.num_uavs)],
            "action_durations": [float(action_durations[i].item()) for i in range(self.num_uavs)],
            "max_duration": max_duration,
            "waiting_times": [max_duration - float(action_durations[i].item()) for i in range(self.num_uavs)],
            "target_points": [int(target_points[i].item()) for i in range(self.num_uavs)],
        })

        # 计算本步骤新完成的任务数
        current_completed_tasks = len(self.processed_tasks)
        previous_completed_tasks = 0
        if self.current_step > 0 and len(self.trajectory_history) > 0:
            # 从上一个step获取任务完成数
            last_step = self.trajectory_history[-1]
            previous_completed_tasks = last_step.get("total_completed_tasks_after_step", 0)

        step_start_info["tasks_completed_this_step"] = current_completed_tasks - previous_completed_tasks
        step_start_info["total_completed_tasks_after_step"] = current_completed_tasks

        # 详细的UAV动作信息
        uav_action_details = []
        for i in range(self.num_uavs):
            action = int(actions[i].item())
            action_detail = {
                "uav_id": i,
                "action": action,
                "action_description": self._get_action_description(action, i),
                "start_point": int(step_start_info["positions_before"][i]),
                "end_point": int(target_points[i].item()),
                "moved": step_start_info["positions_before"][i] != target_points[i].item(),
                "duration": float(action_durations[i].item()),
                "waiting_time": max_duration - float(action_durations[i].item()),
                "energy_consumed": step_start_info["energies_after"][i] - step_start_info["energies_before"][i],
                "reward_received": float(rewards[i, 0].item()),
            }
            uav_action_details.append(action_detail)

        step_start_info["uav_action_details"] = uav_action_details

        # 保存到轨迹历史
        self.trajectory_history.append(step_start_info)
        self.current_step += 1

        # 构建下一个观测
        next_td = self._build_obs()
        next_td["agents"]["reward"] = rewards

        # 判断是否结束
        all_done = bool(torch.all(self.uav_times >= self.max_time).item())
        done = torch.tensor([all_done], dtype=torch.bool, device=self.device)
        next_td.set("done", done)
        next_td.set("terminated", done.clone())

        if all_done:
            metrics = self._compute_final_metrics_corrected()
            self.last_episode_metrics = metrics

        return next_td

    def _compute_final_metrics_corrected(self) -> Dict[str, Any]:
        """计算环境结束时的性能指标，移除kappa地理公平性指标"""
        baseline_energy_per_uav = 450000.0
        baseline_energy_total = baseline_energy_per_uav * self.num_uavs

        # 收集所有任务实例
        instances = []
        skipped_non_triggered = 0

        for point_id, pdata in self.task_data.items():
            area = pdata["basic_info"]["area_name"]
            L_bits = float(pdata["task_params"]["L_bits"])
            T_max = float(pdata["task_params"]["T_max_s"])

            for entry in pdata["gantt_chart"]:
                a_k = float(entry["trigger_time"])

                if a_k >= 9999.0:
                    skipped_non_triggered += 1
                    continue

                deadline_time = float(entry["deadline_time"])
                local_complete_time = entry.get("local_complete_time")
                offload_complete_time = entry.get("offload_complete_time")
                uav_offloaded = bool(entry.get("uav_offloaded", False))
                local_success_flag = int(entry.get("local_success_or_not", 0))
                local_completion_rate = float(entry.get("local_completion_rate", 0.0))

                bits_delivered = 0.0
                success_task = False
                d_k_success = None

                # UAV卸载成功
                if uav_offloaded and offload_complete_time is not None:
                    c_k_edge = float(offload_complete_time)
                    d_edge = c_k_edge - a_k

                    if c_k_edge <= deadline_time + 1e-9:
                        bits_delivered = L_bits
                        success_task = True
                        d_k_success = max(0.0, d_edge)
                    else:
                        bits_delivered = 0.0

                # 本地计算
                else:
                    if local_complete_time is not None:
                        c_k_local = float(local_complete_time)
                        d_local = c_k_local - a_k

                        if c_k_local <= deadline_time + 1e-9:
                            bits_delivered = L_bits * local_completion_rate

                            if local_success_flag == 1 and abs(local_completion_rate - 1.0) < 1e-6:
                                success_task = True
                                d_k_success = max(0.0, d_local)
                        else:
                            bits_delivered = 0.0
                    else:
                        bits_delivered = 0.0

                instances.append({
                    "point_id": int(point_id),
                    "area_name": area,
                    "L_bits": L_bits,
                    "T_max": T_max,
                    "a_k": a_k,
                    "d_k_success": d_k_success,
                    "success": success_task,
                    "bits_delivered": bits_delivered,
                    "uav_served": uav_offloaded,
                })

        # 计算指标
        K_total = len(instances)
        K_success = sum(1 for inst in instances if inst["success"])
        K_uav_served = sum(1 for inst in instances if inst["uav_served"])

        # ψ_task: 任务级完成率
        psi_task = K_success / K_total if K_total > 0 else 0.0

        # ψ: 比特级完成率
        total_L = sum(inst["L_bits"] for inst in instances)
        total_L_success = sum(inst["bits_delivered"] for inst in instances)
        psi = (total_L_success / total_L) if total_L > 0 else 0.0

        # T: deadline-normalized latency
        if K_success > 0:
            sum_norm = 0.0
            for inst in instances:
                if not inst["success"]:
                    continue
                d_k = inst["d_k_success"]
                T_max = inst["T_max"]
                if d_k is None or T_max <= 0:
                    continue
                sum_norm += d_k / T_max

            T_score = sum_norm / K_success if sum_norm > 0 else float("inf")
        else:
            T_score = float("inf")

        # ξ: 归一化UAV能耗
        total_uav_energy = float(torch.sum(self.uav_energies).item())
        xi = total_uav_energy / baseline_energy_total if baseline_energy_total > 0 else 1.0

        # λ: 综合效率（移除kappa，简化为 λ = ψ / (ξ * T)）
        if xi > 0 and T_score > 0 and math.isfinite(T_score):
            lam = psi / (xi * T_score)
        else:
            lam = 0.0

        # 等待时长统计
        total_waiting_time = float(torch.sum(self.uav_total_waiting_times).item())
        waiting_times_per_uav = [float(self.uav_total_waiting_times[i].item()) for i in range(self.num_uavs)]
        avg_waiting_time_per_uav = total_waiting_time / self.num_uavs if self.num_uavs > 0 else 0.0
        max_waiting_time = float(torch.max(self.uav_total_waiting_times).item())

        return {
            "psi_task": torch.tensor(psi_task, device=self.device, dtype=torch.float32),
            "psi": torch.tensor(psi, device=self.device, dtype=torch.float32),
            "T": torch.tensor(T_score, device=self.device, dtype=torch.float32),
            "lambda": torch.tensor(lam, device=self.device, dtype=torch.float32),
            "xi": torch.tensor(xi, device=self.device, dtype=torch.float32),
            "total_uav_energy_J": torch.tensor(total_uav_energy, device=self.device, dtype=torch.float32),
            "baseline_energy_total_J": torch.tensor(baseline_energy_total, device=self.device, dtype=torch.float32),
            "K_total": torch.tensor(K_total, device=self.device, dtype=torch.long),
            "K_success": torch.tensor(K_success, device=self.device, dtype=torch.long),
            "K_uav_served": torch.tensor(K_uav_served, device=self.device, dtype=torch.long),
            "skipped_non_triggered": torch.tensor(skipped_non_triggered, device=self.device, dtype=torch.long),
            "total_episode_time": torch.tensor(self.total_episode_time, device=self.device, dtype=torch.float32),
            # 等待时长相关指标
            "total_waiting_time": torch.tensor(total_waiting_time, device=self.device, dtype=torch.float32),
            "avg_waiting_time_per_uav": torch.tensor(avg_waiting_time_per_uav, device=self.device, dtype=torch.float32),
            "max_waiting_time": torch.tensor(max_waiting_time, device=self.device, dtype=torch.float32),
            "waiting_times_per_uav": torch.tensor(waiting_times_per_uav, device=self.device, dtype=torch.float32),
        }

    def save_metric_taskdata_trajectory(self, save_dir: str, total_frames: int):
        """
        保存metrics、taskdata和trajectory到指定目录

        Args:
            save_dir: 保存目录
            total_frames: 当前训练帧数
        """
        # 创建保存目录
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 1. 强制计算并保存最终metrics
        metrics = self._compute_final_metrics_corrected()
        if metrics:
            # 转换所有torch张量为Python原生类型
            processed_metrics = {}
            for k, v in metrics.items():
                if torch.is_tensor(v):
                    if v.numel() == 1:  # 检查是否为标量
                        processed_metrics[k] = v.item()
                    else:  # 多元素张量转列表
                        processed_metrics[k] = v.tolist()
                else:
                    processed_metrics[k] = v

            # 添加训练进度信息
            processed_metrics["total_frames"] = total_frames
            processed_metrics["n_iters"] = getattr(self, 'n_iters_performed', 0)

            # 保存单次metrics
            metrics_file = save_path / f"metrics_frame_{total_frames}.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(processed_metrics, f, indent=2, ensure_ascii=False)

            # 保存或更新历史记录
            history_file = save_path / "metrics_history.json"
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = []

            history.append(processed_metrics)
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

            print(f"✅ Metrics saved: {metrics_file}")
            print(f"✅ History updated: {history_file}")

        # 2. 保存taskdata
        try:
            taskdata = self.get_taskdata_for_saving()
            taskdata_file = save_path / f"taskdata_frame_{total_frames}.json"
            with open(taskdata_file, 'w', encoding='utf-8') as f:
                json.dump(taskdata, f, indent=2, ensure_ascii=False)

            print(f"✅ TaskData saved: {taskdata_file}")
            print(f"   - Points with stats: {len(taskdata['point_statistics'])}")
            print(f"   - Total tasks processed: {taskdata['env_state']['completed_tasks_count']}")
        except Exception as e:
            print(f"❌ Warning: Failed to save taskdata: {e}")

        # 3. 保存trajectory（轨迹数据）
        try:
            trajectory = self.get_trajectory_for_saving()
            trajectory_file = save_path / f"trajectory_frame_{total_frames}.json"
            with open(trajectory_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory, f, indent=2, ensure_ascii=False)

            print(f"✅ Trajectory saved: {trajectory_file}")
            print(f"   - UAV count: {trajectory['metadata']['num_uavs']}")
            print(f"   - Episode duration: {trajectory['metadata']['total_episode_time']:.2f}s")
            print(f"   - Total steps recorded: {trajectory['metadata']['total_steps']}")
        except Exception as e:
            print(f"❌ Warning: Failed to save trajectory: {e}")
            import traceback
            traceback.print_exc()

        # 输出保存总结
        print(f"\n{'=' * 60}")
        print(f"📊 Evaluation Results Saved (Frame {total_frames})")
        print(f"{'=' * 60}")
        print(f"📈 Metrics: {save_path / f'metrics_frame_{total_frames}.json'}")
        print(f"📋 TaskData: {save_path / f'taskdata_frame_{total_frames}.json'}")
        print(f"🔄 Trajectory: {save_path / f'trajectory_frame_{total_frames}.json'}")
        print(f"📚 History: {save_path / 'metrics_history.json'}")
        print(f"{'=' * 60}\n")

    def get_taskdata_for_saving(self) -> Dict[str, Any]:
        """
        获取用于保存的任务数据快照
        """
        # 收集所有点的统计信息
        point_statistics = {}

        for point_id, pdata in self.task_data.items():
            area_name = pdata["basic_info"]["area_name"]
            gantt_chart = pdata["gantt_chart"]

            # 统计该点的任务情况
            total_tasks = 0
            triggered_tasks = 0
            uav_completed = 0
            local_completed = 0

            for task in gantt_chart:
                trigger_time = float(task["trigger_time"])
                if trigger_time < 9999.0:  # 有效触发的任务
                    total_tasks += 1
                    triggered_tasks += 1

                    if task.get("uav_offloaded", False):
                        uav_completed += 1
                    elif task.get("local_success_or_not", 0) == 1:
                        local_completed += 1

            point_statistics[point_id] = {
                "area_name": area_name,
                "total_tasks": total_tasks,
                "triggered_tasks": triggered_tasks,
                "uav_completed": uav_completed,
                "local_completed": local_completed,
                "uav_completion_rate": uav_completed / triggered_tasks if triggered_tasks > 0 else 0.0,
            }

        return {
            "point_statistics": point_statistics,
            "env_state": {
                "completed_tasks_count": len(self.processed_tasks),
                "total_episode_time": self.total_episode_time,
                "total_served_bits": self.total_served_bits,
                "area_served_bits": dict(self.area_served_bits),
            },
            "uav_states": {
                "positions": [int(p.item()) for p in self.uav_points],
                "times": [float(t.item()) for t in self.uav_times],
                "energies": [float(e.item()) for e in self.uav_energies],
                "waiting_times": [float(w.item()) for w in self.uav_total_waiting_times],
            }
        }

    def get_trajectory_for_saving(self) -> Dict[str, Any]:
        """
        获取用于保存的完整轨迹数据
        包含每个step的详细信息：时间、位置、动作、奖励等
        """
        try:
            # 获取最终UAV状态
            final_uav_states = []
            for i in range(self.num_uavs):
                uav_state = {
                    "uav_id": i,
                    "start_point": self.uav_starts[i],
                    "final_point": int(self.uav_points[i].item()),
                    "final_coordinates": self.uav_coords[i].cpu().numpy().tolist(),
                    "final_time": float(self.uav_times[i].item()),
                    "total_energy_consumed": float(self.uav_energies[i].item()),
                    "total_waiting_time": float(self.uav_total_waiting_times[i].item()),
                }
                final_uav_states.append(uav_state)

            # 统计轨迹数据
            total_steps = len(self.trajectory_history)
            total_movements = 0
            total_stays = 0

            for step_data in self.trajectory_history:
                if "uav_action_details" in step_data:
                    for uav_detail in step_data["uav_action_details"]:
                        if uav_detail.get("moved", False):
                            total_movements += 1
                        else:
                            total_stays += 1

            # 统计每个UAV的轨迹摘要
            uav_trajectory_summaries = []
            for i in range(self.num_uavs):
                moves = 0
                stays = 0
                total_distance = 0.0
                visited_points = set([self.uav_starts[i]])

                for step_data in self.trajectory_history:
                    uav_action_details = step_data.get("uav_action_details", [])
                    if i < len(uav_action_details):
                        uav_detail = uav_action_details[i]
                        visited_points.add(uav_detail.get("end_point", self.uav_starts[i]))
                        if uav_detail.get("moved", False):
                            moves += 1
                        else:
                            stays += 1

                uav_trajectory_summaries.append({
                    "uav_id": i,
                    "total_moves": moves,
                    "total_stays": stays,
                    "unique_points_visited": len(visited_points),
                    "visited_points_list": sorted(list(visited_points)),
                })

            # 构建完整轨迹数据
            trajectory_data = {
                # 基本信息
                "metadata": {
                    "num_uavs": self.num_uavs,
                    "uav_speed": self.uav_speed,
                    "max_time": self.max_time,
                    "hover_window": self.hover_window,
                    "total_episode_time": self.total_episode_time,
                    "total_steps": total_steps,
                    "environment_config": {
                        "obs_dim": self.obs_dim,
                        "act_dim": self.act_dim,
                        "w_task": self.w_task,
                        "w_time_penalty": self.w_time_penalty,
                        "P_fly": self.P_fly,
                        "P_hover": self.P_hover,
                    }
                },

                # 最终状态
                "final_states": {
                    "uav_states": final_uav_states,
                    "total_tasks_processed": len(self.processed_tasks),
                    "total_served_bits": self.total_served_bits,
                    "area_served_bits": dict(self.area_served_bits),
                },

                # 轨迹统计摘要
                "trajectory_summary": {
                    "total_movements": total_movements,
                    "total_stays": total_stays,
                    "uav_summaries": uav_trajectory_summaries,
                },

                # 详细的每步轨迹历史（这是核心数据）
                "step_by_step_history": self.trajectory_history,

                # 处理过的任务列表
                "processed_tasks": [
                    {
                        "point_id": task_key[0],
                        "trigger_time": task_key[1],
                    } for task_key in self.processed_tasks
                ],
            }

            return trajectory_data

        except Exception as e:
            print(f"Error in get_trajectory_for_saving: {e}")
            import traceback
            traceback.print_exc()

            # 返回基本的轨迹数据作为fallback
            return {
                "metadata": {
                    "num_uavs": getattr(self, 'num_uavs', 0),
                    "total_episode_time": getattr(self, 'total_episode_time', 0.0),
                    "total_steps": len(getattr(self, 'trajectory_history', [])),
                    "error": str(e),
                },
                "step_by_step_history": getattr(self, 'trajectory_history', []),
                "processed_tasks": [
                    {
                        "point_id": task_key[0],
                        "trigger_time": task_key[1],
                    } for task_key in getattr(self, 'processed_tasks', set())
                ],
            }

    def _get_action_description(self, action: int, uav_idx: int) -> str:
        """获取动作的描述"""
        if action == 5:
            return "Stay (hover at current position)"
        elif 0 <= action <= 4:
            target_point = int(self.uav_nearby_points[uav_idx, action].item())
            return f"Move to nearby point {action} (point_id: {target_point})"
        else:
            return f"Invalid action ({action})"

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        return seed

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            return 999
        elif mode == "human":
            return
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def get_nearby_points_info(self) -> List[List[Dict[str, Any]]]:
        """获取每个UAV的邻近点信息，用于调试和可视化（使用绝对坐标）"""
        result = []
        for i in range(self.num_uavs):
            uav_nearby = []
            for j in range(5):
                point_id = int(self.uav_nearby_points[i, j].item())
                coords = self.uav_nearby_coords[i, j].cpu().numpy()
                reward = float(self.uav_nearby_path_rewards[i, j].item())

                uav_nearby.append({
                    "id": point_id,
                    "absolute_coords": coords.tolist(),
                    "path_reward": reward
                })
            result.append(uav_nearby)
        return result

    def print_current_state(self):
        """打印当前环境状态，用于调试（使用绝对坐标）"""
        print(f"\n=== Environment State ===")
        print(f"Total episode time: {self.total_episode_time:.2f}s")
        print(f"UAV positions: {[int(p.item()) for p in self.uav_points]}")
        print(f"UAV times: {[f'{t.item():.2f}' for t in self.uav_times]}")
        print(f"UAV waiting times: {[f'{t.item():.2f}' for t in self.uav_total_waiting_times]}")
        print(f"Completed tasks: {len(self.processed_tasks)}")

        print(f"\nUAV Absolute Coordinates:")
        for i in range(self.num_uavs):
            coords = self.uav_coords[i].cpu().numpy()
            print(f"UAV {i}: {coords}")

        print(f"\nNearby Points and Path Rewards:")
        for i in range(self.num_uavs):
            print(f"UAV {i} (at point {int(self.uav_points[i].item())}):")
            for j in range(5):
                point_id = int(self.uav_nearby_points[i, j].item())
                reward = self.uav_nearby_path_rewards[i, j].item()
                coords = self.uav_nearby_coords[i, j].cpu().numpy()
                print(f"  Nearby {j}: ID={point_id}, absolute_coords={coords}, reward={reward:.2f}")

            # 打印原地停留奖励
            stay_reward = self.uav_stay_rewards[i].item()
            print(f"  Stay reward: {stay_reward:.2f}")