from dataclasses import dataclass, MISSING
from torchrl.envs import EnvBase
from torchrl.data import Categorical, Unbounded, Composite, Binary, BoundedContinuous
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
from datetime import datetime
from .get_points_and_paths import PathPlanner, BidirectionalPathManager


@dataclass
class TaskConfig:
    placeholder: int = MISSING


class San_scen1_continu_Env(EnvBase):
    """
    基于连续动作空间的多UAV调度MARL环境

    核心特性：
    1. 连续动作空间：[azimuth, elevation, distance_ratio]
       - azimuth: 方位角（0-360度），在XY平面内从X轴正方向的旋转角度
       - elevation: 仰角（-90到90度），从XY平面向上或向下的偏离角度
       - distance_ratio: 沿该方向移动的距离比例（0-1），实际距离 = uav_speed * distance_ratio * time_step
    2. 固定时间步长：1秒
    3. 任务卸载：UAV在20米范围内的点可以进行任务卸载
    4. 边界检查：UAV飞出所有cube限定的连通域时环境终止
    5. 增强观测：包含5个最近导航点的坐标和潜在奖励
    6. 轨迹跟踪：记录每个UAV在每个时间步的位置和行为
    """

    # 类变量：所有实例共享的数据
    _shared_data_loaded = False
    _path_planner = None
    _path_manager = None
    _task_data = None
    _navigable_points = None  # 新增：存储所有导航点
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

                # 收集所有区域名
                cls._all_areas = sorted({pdata["basic_info"]["area_name"]
                                         for pdata in cls._task_data.values()})

                # 新增：加载所有可导航点（green, yellow, red点）
                cls._load_navigable_points()

                cls._shared_data_loaded = True
                print("Shared environment data loaded successfully!")

            except Exception as e:
                print(f"Error loading shared data: {e}")
                raise

    @classmethod
    def _load_navigable_points(cls):
        """从PathPlanner加载所有可导航点（green, yellow, red点）"""
        cls._navigable_points = []

        # 获取所有可导航点（不包含出入口点）
        for area_name, area_data in cls._path_planner.areas.items():
            # 跳过outside区域（用于跨区域连接）
            if area_name == "outside":
                continue

            cubes = area_data["cubes"]
            for point in area_data["points"]:
                color_class = point.get("color_class", "unknown")

                # 只包含green, yellow, red点（导航点），排除blue（出入口）
                if color_class.lower() in ["green", "yellow", "red"]:
                    coords = point["coords"]

                    # 确保点在可飞行区域内
                    if cls._path_planner._is_point_in_flyable_area(coords, cubes):
                        cls._navigable_points.append({
                            "id": int(point["id"]),
                            "coords": coords,  # [x, y, z] in cm
                            "area_name": area_name,
                            "color_class": color_class
                        })

        print(f"Loaded {len(cls._navigable_points)} navigable points (green, yellow, red)")

    def __init__(
            self,
            uav_starts: List[int] = [20001, 40003],
            uav_speed: float = 18.0,
            max_time: float = 1800.0,
            time_step: float = 1.0,
            task_offload_range: float = 20.0,
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
        self.navigable_points = self._navigable_points  # 新增：可导航点引用
        self.processed_tasks = set()
        self.last_episode_metrics = None

        self.uav_starts = list(uav_starts)
        self.num_uavs = len(uav_starts)
        self.uav_speed = float(uav_speed)  # m/s
        self.max_time = float(max_time)
        self.time_step = float(time_step)  # 固定时间步长（秒）
        self.task_offload_range = float(task_offload_range)  # 任务卸载范围（米）

        # UAV能耗参数（仅用于最终指标计算）
        self.P_fly = 101.608126  # 飞行功率(W)
        self.P_hover = 221.160000  # 悬停功率(W)

        # 观测维度更新：
        # - 自身绝对坐标(3)
        # - 所有UAV绝对坐标(3*num_uavs) - 包括自己
        # - 5个邻近点信息：每个包含绝对坐标(3) + 路径奖励(1) = 4维
        # - 1个原地停留奖励(1)
        # 总维度：3 + 3*num_uavs + 5*4 + 1 = 3 + 6 + 20 + 1 = 30 (对于2个UAV)
        self.obs_dim = 3 + 3 * self.num_uavs + 5 * 4 + 1

        # 动作：连续动作 [azimuth, elevation, distance_ratio] - 3维
        self.act_dim = 3

        # agent group
        self.group_map = {"agents": [f"uav_{i}" for i in range(self.num_uavs)]}

        # ===== UAV状态 =====
        self.uav_points = torch.zeros(self.num_uavs, dtype=torch.long, device=self.device)
        self.uav_coords = torch.zeros(self.num_uavs, 3, dtype=torch.float32, device=self.device)  # 实际坐标（厘米）
        self.uav_normalized_coords = torch.zeros(self.num_uavs, 3, dtype=torch.float32, device=self.device)  # 归一化坐标
        self.uav_times = torch.zeros(self.num_uavs, dtype=torch.float32, device=self.device)
        self.uav_energies = torch.zeros(self.num_uavs, dtype=torch.float32, device=self.device)

        # ===== 等待时长统计 =====
        self.uav_total_waiting_times = torch.zeros(self.num_uavs, dtype=torch.float32, device=self.device)

        # ===== 奖励设计参数 =====
        self.w_task = 100.0  # 修改：每完成一个任务的基础奖励
        self.w_time_penalty = 0.001  # 轻微时间惩罚
        self.w_boundary_penalty = -10.0  # 飞出边界的惩罚

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

        # ===== 轨迹跟踪 =====
        self.trajectory_data = []  # 存储每个step的轨迹信息
        self.current_step = 0

        # 预处理所有区域的cube数据，用于边界检查
        self._prepare_boundary_data()

        self._make_spec()

    def _prepare_boundary_data(self):
        """预处理边界数据，用于高效的边界检查"""
        self.all_cubes = []
        for area_name, area_data in self.path_planner.areas.items():
            self.all_cubes.extend(area_data["cubes"])

        print(f"Loaded {len(self.all_cubes)} cubes for boundary checking")

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

        # 连续动作空间：[azimuth, elevation, distance_ratio]
        # azimuth: [0, 360]度, elevation: [-90, 90]度, distance_ratio: [0, 1]
        low = torch.tensor([0.0, -90.0, 0.0], device=self.device, dtype=torch.float32) \
            .expand(self.num_uavs, self.act_dim)
        high = torch.tensor([360.0, 90.0, 1.0], device=self.device, dtype=torch.float32) \
            .expand(self.num_uavs, self.act_dim)

        self.action_spec = Composite(
            agents=Composite(
                action=BoundedContinuous(
                    shape=(self.num_uavs, self.act_dim),
                    device=self.device,
                    dtype=torch.float32,
                    low=low,
                    high=high,
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

    def _record_trajectory_step(self, actions: torch.Tensor, rewards: torch.Tensor,
                                destinations: List[Optional[Dict[str, Any]]]):
        """
        记录当前step的轨迹信息

        Args:
            actions: UAV动作 [num_uavs, 3] - [azimuth, elevation, distance_ratio]
            rewards: UAV奖励 [num_uavs, 1]
            destinations: 每个UAV的目标点信息 (如果有的话)
        """
        current_time = float(torch.max(self.uav_times).item())

        step_data = {
            "step": self.current_step,
            "timestamp": current_time,
            "uavs": []
        }

        for i in range(self.num_uavs):
            # 获取UAV当前状态
            current_coords = self.uav_coords[i].cpu().numpy().tolist()  # [x, y, z] in cm
            action = actions[i].cpu().numpy().tolist()  # [azimuth, elevation, distance_ratio]
            reward = float(rewards[i, 0].item())

            # 计算目标坐标（基于动作预测）
            azimuth, elevation, distance_ratio = action
            direction = self._spherical_to_velocity(azimuth, elevation)
            move_distance_cm = self.uav_speed * distance_ratio * self.time_step * 100.0  # 转换为厘米
            target_coords = (np.array(current_coords) + direction * move_distance_cm).tolist()

            uav_data = {
                "uav_id": i,
                "current_position": {
                    "x_cm": current_coords[0],
                    "y_cm": current_coords[1],
                    "z_cm": current_coords[2]
                },
                "action": {
                    "azimuth_deg": azimuth,
                    "elevation_deg": elevation,
                    "distance_ratio": distance_ratio
                },
                "target_position": {
                    "x_cm": target_coords[0],
                    "y_cm": target_coords[1],
                    "z_cm": target_coords[2]
                },
                "reward": reward,
                "energy_consumed_J": float(self.uav_energies[i].item()),
                "waiting_time_s": float(self.uav_total_waiting_times[i].item())
            }

            # 如果有特定目标点，添加目标点信息
            if destinations and i < len(destinations) and destinations[i] is not None:
                dest = destinations[i]
                uav_data["destination_info"] = {
                    "target_point_id": dest.get("id"),
                    "target_area": dest.get("area_name"),
                    "target_coords": dest.get("coords"),
                    "distance_to_target_cm": dest.get("distance")
                }

            step_data["uavs"].append(uav_data)

        self.trajectory_data.append(step_data)
        self.current_step += 1

    def _update_uav_coordinates(self):
        """更新UAV的坐标（用于初始化和重置）"""
        for i in range(self.num_uavs):
            point_id = int(self.uav_points[i].item())
            try:
                coords, _ = self.path_planner.coords_from_global_id(point_id)
                # 保存实际坐标（厘米）
                self.uav_coords[i] = torch.tensor(coords, dtype=torch.float32, device=self.device)
                # 计算归一化坐标（用于观测）
                normalized_coords = [coord / 100000.0 for coord in coords]  # 厘米转为合理单位
                self.uav_normalized_coords[i] = torch.tensor(normalized_coords, dtype=torch.float32, device=self.device)
            except:
                # 如果无法获取坐标，使用默认值
                self.uav_coords[i] = torch.zeros(3, dtype=torch.float32, device=self.device)
                self.uav_normalized_coords[i] = torch.zeros(3, dtype=torch.float32, device=self.device)

    def _spherical_to_velocity(self, azimuth: float, elevation: float) -> np.ndarray:
        """
        将球面坐标转换为单位速度向量

        参数:
        - azimuth: 方位角（度），范围[0, 360]，在XY平面内从X轴正方向的旋转角度
        - elevation: 仰角（度），范围[-90, 90]，从XY平面向上或向下的偏离角度

        返回:
        - 3D单位向量 (x, y, z)
        """
        # 转换为弧度
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)

        # 球面坐标到直角坐标的转换
        # X轴：东方，Y轴：北方，Z轴：向上
        x = np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = np.sin(elevation_rad)

        # 确保是单位向量（理论上应该已经是）
        direction = np.array([x, y, z])
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm

        return direction

    def _is_point_in_flyable_area(self, point: np.ndarray) -> bool:
        """判断点是否在可飞行区域内（任何cube内）"""
        for cube in self.all_cubes:
            if self._is_point_in_cube(point, cube):
                return True
        return False

    def _is_point_in_cube(self, point: np.ndarray, cube: dict) -> bool:
        """判断点是否在立方体内"""
        center = np.array(cube["center"])
        size = np.array(cube["size"])
        R = cube["rotation_matrix"]

        # 转换到局部坐标系
        local = R.T @ (point - center)
        half = size / 2.0

        return np.all(np.abs(local) <= half)

    def _get_nearest_navigable_points(self, uav_coords: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        获取离UAV最近的k个可导航点

        返回格式：
        [
            {
                "coords": [x, y, z],  # 绝对坐标（厘米）
                "distance": float,     # 距离（厘米）
                "id": int,            # 点ID
                "area_name": str      # 区域名
            },
            ...
        ]
        """
        if not self.navigable_points:
            return []

        uav_pos = np.array(uav_coords, dtype=float)
        distances = []

        for point in self.navigable_points:
            point_coords = np.array(point["coords"], dtype=float)
            dist = np.linalg.norm(uav_pos - point_coords)
            distances.append({
                "coords": point["coords"],
                "distance": dist,
                "id": point["id"],
                "area_name": point["area_name"]
            })

        # 按距离排序并返回前k个
        distances.sort(key=lambda x: x["distance"])
        return distances[:k]

    def _calculate_potential_reward(self, uav_idx: int, target_coords: List[float],
                                    flight_duration: float, current_time: float) -> float:
        """
        计算UAV飞到目标点的潜在奖励G

        参数:
        - uav_idx: UAV索引
        - target_coords: 目标点坐标
        - flight_duration: 飞行时间（秒）
        - current_time: 当前时间

        返回:
        - 潜在奖励值
        """
        # 模拟到达时间
        arrival_time = current_time + flight_duration

        # 检查该点是否有任务可以执行
        target_coords_str = None
        for point_id_str, point_data in self.task_data.items():
            try:
                point_id = int(point_id_str)
                coords, _ = self.path_planner.coords_from_global_id(point_id)

                # 检查坐标是否匹配（允许小误差）
                if np.linalg.norm(np.array(coords) - np.array(target_coords)) < 100:  # 1米误差
                    target_coords_str = point_id_str
                    break
            except:
                continue

        if not target_coords_str:
            return 0.0

        # 获取该点在到达时间可处理的任务
        valid_tasks = self._get_tasks_at_point_and_time(target_coords_str, arrival_time)

        if not valid_tasks:
            return 0.0

        # 计算任务奖励
        task_count = len(valid_tasks)
        task_reward = task_count * self.w_task

        # 时间折扣奖励：任务奖励 / 用时
        if flight_duration > 0:
            time_discounted_reward = task_reward / flight_duration
        else:
            time_discounted_reward = task_reward

        # 减去时间成本
        time_cost = flight_duration * self.w_time_penalty

        return max(0.0, time_discounted_reward - time_cost)

    def _calculate_stay_reward(self, uav_idx: int, current_time: float) -> float:
        """
        计算UAV原地停留的奖励

        参数:
        - uav_idx: UAV索引
        - current_time: 当前时间

        返回:
        - 停留奖励值
        """
        uav_coords = self.uav_coords[uav_idx].cpu().numpy()

        # 检查当前位置是否有任务可以执行
        nearby_points = self._find_nearby_task_points(uav_coords, self.task_offload_range)

        total_reward = 0.0
        for point_id_str in nearby_points:
            valid_tasks = self._get_tasks_at_point_and_time(point_id_str, current_time)
            task_count = len(valid_tasks)
            total_reward += task_count * self.w_task

        # 减去时间成本
        time_cost = self.time_step * self.w_time_penalty

        return max(0.0, total_reward - time_cost)

    def _find_nearby_task_points(self, uav_coords: np.ndarray, range_m: float) -> List[str]:
        """找到UAV范围内的任务点"""
        nearby_points = []
        range_cm = range_m * 100.0  # 转换为厘米

        for point_id_str, point_data in self.task_data.items():
            try:
                point_id = int(point_id_str)
                coords, _ = self.path_planner.coords_from_global_id(point_id)
                coords = np.array(coords)

                # 计算距离
                distance = np.linalg.norm(uav_coords - coords)
                if distance <= range_cm:
                    nearby_points.append(point_id_str)
            except:
                continue

        return nearby_points

    def _process_tasks_at_nearby_points(self, uav_idx: int, current_time: float) -> Tuple[int, float, float]:
        """处理UAV周围的任务"""
        uav_coords = self.uav_coords[uav_idx].cpu().numpy()
        nearby_points = self._find_nearby_task_points(uav_coords, self.task_offload_range)

        total_tasks_completed = 0
        total_energy = 0.0
        total_service_time = 0.0

        for point_id_str in nearby_points:
            # 获取该点在当前时间可处理的任务
            valid_tasks = self._get_tasks_at_point_and_time(point_id_str, current_time)

            if not valid_tasks:
                continue

            # 按触发时间排序
            valid_tasks.sort(key=lambda x: x["trigger_time"])

            service_start_time = current_time

            for task_info in valid_tasks:
                # 尝试处理任务
                success, service_end, energy = self._process_single_task(
                    uav_idx, point_id_str, task_info, service_start_time
                )

                if success:
                    total_tasks_completed += 1
                    total_energy += energy
                    service_start_time = service_end  # 下一个任务的开始时间

                    # 如果服务时间超过了时间步长，停止处理更多任务
                    service_duration = service_end - current_time
                    if service_duration >= self.time_step:
                        total_service_time = service_duration
                        break

            # 更新服务时间
            if service_start_time > current_time:
                total_service_time = max(total_service_time, service_start_time - current_time)

        # 限制服务时间不超过时间步长
        total_service_time = min(total_service_time, self.time_step)

        return total_tasks_completed, total_energy, total_service_time

    def _get_tasks_at_point_and_time(self, point_id: str, current_time: float) -> List[Dict[str, Any]]:
        """获取指定点在指定时间可以处理的任务"""
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

            # 判断任务是否可处理：任务已触发且未过期
            if trigger_time <= current_time + self.time_step and latest_arrival >= current_time:
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

    def _process_single_task(self, uav_idx: int, point_id: str, task_info: Dict[str, Any],
                             service_start: float) -> Tuple[bool, Optional[float], float]:
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
        actual_start = max(service_start, trigger_time)

        # 检查是否在最晚到达时间之前
        if actual_start > latest_arrival:
            return False, None, 0.0

        # 服务结束时间
        service_end = actual_start + t_mec

        # 检查是否在deadline之前完成
        if service_end > task_info["deadline_time"]:
            return False, None, 0.0

        # 计算能耗
        E_u_ul = float(offload_info["E_u_ul_J"])
        E_u_dl = float(offload_info["E_u_dl_J"])
        E_u_ec = float(offload_info["E_u_ec_J"])
        E_compute = E_u_ul + E_u_dl + E_u_ec

        # 更新任务处理标记
        task_key = (int(point_id), trigger_time)
        self.processed_tasks.add(task_key)

        # 更新任务数据
        for gt_entry in self.task_data[point_id]["gantt_chart"]:
            if float(gt_entry["trigger_time"]) == trigger_time:
                gt_entry["uav_offloaded"] = True
                gt_entry["uav_id"] = uav_idx
                gt_entry["uav_arrival_time"] = service_start
                gt_entry["offload_complete_time"] = service_end
                break

        # 更新区域统计
        self.area_served_bits[area_name] += L_bits
        self.total_served_bits += L_bits

        return True, service_end, E_compute

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
        current_time = float(torch.max(self.uav_times).item())

        for i in range(self.num_uavs):
            obs_idx = 0
            current_coords = self.uav_normalized_coords[i]  # 当前UAV的归一化坐标
            current_coords_cm = self.uav_coords[i].cpu().numpy()  # 当前UAV的实际坐标（厘米）

            # 1. 自身绝对坐标(3维) - 使用归一化坐标
            obs[i, obs_idx:obs_idx + 3] = current_coords
            obs_idx += 3

            # 2. 所有UAV绝对坐标(3*num_uavs) - 包括自己
            for j in range(self.num_uavs):
                obs[i, obs_idx:obs_idx + 3] = self.uav_normalized_coords[j]
                obs_idx += 3

            # 3. 5个邻近点信息：每个包含绝对坐标(3) + 路径奖励(1) = 4维
            nearest_points = self._get_nearest_navigable_points(current_coords_cm, k=5)

            for k in range(5):
                if k < len(nearest_points):
                    point_info = nearest_points[k]
                    # 点的归一化坐标
                    point_coords_normalized = [coord / 100000.0 for coord in point_info["coords"]]
                    obs[i, obs_idx:obs_idx + 3] = torch.tensor(point_coords_normalized, dtype=torch.float32,
                                                               device=self.device)

                    # 计算飞到该点的时间
                    distance_m = point_info["distance"] / 100.0  # 厘米转米
                    flight_duration = distance_m / self.uav_speed if self.uav_speed > 0 else 0.0

                    # 计算潜在奖励
                    potential_reward = self._calculate_potential_reward(
                        i, point_info["coords"], flight_duration, current_time
                    )
                    obs[i, obs_idx + 3] = potential_reward
                else:
                    # 填充零值
                    obs[i, obs_idx:obs_idx + 4] = torch.zeros(4, dtype=torch.float32, device=self.device)
                obs_idx += 4

            # 4. 原地停留奖励(1维)
            stay_reward = self._calculate_stay_reward(i, current_time)
            obs[i, obs_idx] = stay_reward
            obs_idx += 1

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

    def _reset(self, tensordict=None, **kwargs):
        """重置环境"""
        # 重置UAV位置到起始点
        for i in range(self.num_uavs):
            self.uav_points[i] = int(self.uav_starts[i])

        # 重置状态
        self.uav_times.zero_()
        self.uav_energies.zero_()
        self.uav_total_waiting_times.zero_()

        # 重置轨迹跟踪
        self.trajectory_data = []
        self.current_step = 0

        # 更新坐标
        self._update_uav_coordinates()

        # 重置任务数据
        self.task_data = copy.deepcopy(self._task_data)
        self.area_served_bits.clear()
        self.total_served_bits = 0.0
        self.completed_tasks_count = 0
        self.total_episode_time = 0.0
        self.processed_tasks.clear()

        td = self._build_obs()
        td.set("done", torch.zeros(1, dtype=torch.bool, device=self.device))
        td.set("terminated", torch.zeros(1, dtype=torch.bool, device=self.device))
        return td

    def _step(self, tensordict: TensorDict):
        """执行一步 - 所有UAV同步执行连续动作（球面坐标系）"""
        actions = tensordict["agents"]["action"].to(self.device).float()  # [num_uavs, 3]
        rewards = torch.zeros((self.num_uavs, 1), dtype=torch.float32, device=self.device)

        current_time = float(torch.max(self.uav_times).item())

        # 准备轨迹记录的目标信息
        destinations = []

        # 处理每个UAV的连续动作
        boundary_violation = False

        for i in range(self.num_uavs):
            action = actions[i].cpu().numpy()  # [azimuth, elevation, distance_ratio]

            # 解析动作
            azimuth, elevation, distance_ratio = action

            # 限制参数范围（以防万一）
            azimuth = np.clip(azimuth, 0.0, 360.0)
            elevation = np.clip(elevation, -90.0, 90.0)
            distance_ratio = np.clip(distance_ratio, 0.0, 1.0)

            # 计算移动方向和距离
            direction = self._spherical_to_velocity(azimuth, elevation)
            move_distance = self.uav_speed * distance_ratio * self.time_step  # 米
            move_distance_cm = move_distance * 100.0  # 转换为厘米

            # 计算新位置
            current_pos = self.uav_coords[i].cpu().numpy()
            new_pos = current_pos + direction * move_distance_cm

            # 找到最近的导航点（用于轨迹记录）
            nearest_points = self._get_nearest_navigable_points(current_pos, k=1)
            destination_info = nearest_points[0] if nearest_points else None
            destinations.append(destination_info)

            # 边界检查
            if not self._is_point_in_flyable_area(new_pos):
                # 飞出边界，给予惩罚但不移动
                rewards[i, 0] += self.w_boundary_penalty
                boundary_violation = True
                # 保持原位置不变
                new_pos = current_pos
            else:
                # 更新位置
                self.uav_coords[i] = torch.tensor(new_pos, dtype=torch.float32, device=self.device)
                # 更新归一化坐标
                normalized_coords = [coord / 100000.0 for coord in new_pos]
                self.uav_normalized_coords[i] = torch.tensor(normalized_coords, dtype=torch.float32, device=self.device)

            # 计算移动能耗
            actual_distance = np.linalg.norm(new_pos - current_pos) / 100.0  # 米
            if actual_distance > 0:
                move_time = actual_distance / self.uav_speed
                move_energy = self.P_fly * move_time
            else:
                move_time = 0.0
                move_energy = 0.0

            # 计算悬停能耗（时间步长减去移动时间）
            hover_time = max(0.0, self.time_step - move_time)
            hover_energy = self.P_hover * hover_time

            total_energy = move_energy + hover_energy

            # 处理附近的任务
            tasks_completed, task_energy, service_time = self._process_tasks_at_nearby_points(i, current_time)

            # 累加能耗
            self.uav_energies[i] += total_energy + task_energy

            # 计算奖励
            task_reward = tasks_completed * self.w_task
            time_penalty = self.time_step * self.w_time_penalty
            total_reward = task_reward - time_penalty

            rewards[i, 0] += total_reward

        # 记录轨迹数据
        self._record_trajectory_step(actions, rewards, destinations)

        # 所有UAV推进相同的时间
        self.uav_times += self.time_step

        # 记录总时间
        self.total_episode_time = float(torch.max(self.uav_times).item())

        # 构建下一个观测
        next_td = self._build_obs()
        next_td["agents"]["reward"] = rewards

        # 判断是否结束
        time_up = bool(torch.all(self.uav_times >= self.max_time).item())
        done = torch.tensor([time_up or boundary_violation], dtype=torch.bool, device=self.device)
        next_td.set("done", done)
        next_td.set("terminated", done.clone())

        if done.item():
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
        保存metrics、taskdata和trajectory数据到指定目录

        Args:
            save_dir: 保存目录路径
            total_frames: 当前训练的总帧数
        """
        try:
            # 创建保存目录
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 1. 保存最终指标 (metrics)
            if self.last_episode_metrics is not None:
                metrics_file = save_path / f"metrics_frames_{total_frames}_{timestamp}.json"

                # 转换tensor为可序列化的格式
                metrics_dict = {}
                for key, value in self.last_episode_metrics.items():
                    if torch.is_tensor(value):
                        if value.numel() == 1:  # 标量tensor
                            metrics_dict[key] = float(value.item())
                        else:  # 多维tensor
                            metrics_dict[key] = value.cpu().numpy().tolist()
                    else:
                        metrics_dict[key] = value

                # 添加训练信息
                metrics_dict["training_info"] = {
                    "total_frames": total_frames,
                    "save_timestamp": timestamp,
                    "num_uavs": self.num_uavs,
                    "uav_starts": self.uav_starts,
                    "max_time": self.max_time,
                    "time_step": self.time_step,
                    "action_space": "spherical_coordinates",  # 新增动作空间标识
                    "action_dims": ["azimuth_deg", "elevation_deg", "distance_ratio"],
                    "action_bounds": [[0.0, 360.0], [-90.0, 90.0], [0.0, 1.0]]
                }

                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

                print(f"✅ Metrics saved to: {metrics_file}")

            # 2. 保存任务数据 (taskdata)
            taskdata_file = save_path / f"taskdata_frames_{total_frames}_{timestamp}.json"

            with open(taskdata_file, 'w', encoding='utf-8') as f:
                json.dump(self.task_data, f, indent=2, ensure_ascii=False)

            print(f"✅ Task data saved to: {taskdata_file}")

            # 3. 保存轨迹数据 (trajectory)
            trajectory_file = save_path / f"trajectory_frames_{total_frames}_{timestamp}.json"

            trajectory_summary = {
                "episode_info": {
                    "total_frames": total_frames,
                    "save_timestamp": timestamp,
                    "num_uavs": self.num_uavs,
                    "total_steps": len(self.trajectory_data),
                    "episode_duration_s": self.total_episode_time,
                    "uav_start_points": self.uav_starts,
                    "action_space": "spherical_coordinates",
                    "action_description": {
                        "azimuth_deg": "方位角，范围[0, 360]度，从X轴正方向在XY平面内的旋转角度",
                        "elevation_deg": "仰角，范围[-90, 90]度，从XY平面向上或向下的偏离角度",
                        "distance_ratio": "距离比例，范围[0, 1]，实际移动距离 = uav_speed * distance_ratio * time_step"
                    }
                },
                "trajectory_data": self.trajectory_data
            }

            with open(trajectory_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory_summary, f, indent=2, ensure_ascii=False)

            print(f"✅ Trajectory saved to: {trajectory_file}")

            # 4. 创建简要摘要文件
            summary_file = save_path / f"summary_frames_{total_frames}_{timestamp}.txt"

            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Episode Summary - Frame {total_frames}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"=" * 50 + "\n\n")

                f.write("Action Space: Spherical Coordinates\n")
                f.write("  - azimuth: [0, 360] degrees\n")
                f.write("  - elevation: [-90, 90] degrees\n")
                f.write("  - distance_ratio: [0, 1]\n\n")

                if self.last_episode_metrics is not None:
                    f.write("Performance Metrics:\n")
                    f.write(f"  Task Completion Rate (ψ_task): {metrics_dict.get('psi_task', 'N/A'):.4f}\n")
                    f.write(f"  Bit Completion Rate (ψ): {metrics_dict.get('psi', 'N/A'):.4f}\n")
                    f.write(f"  Normalized Latency (T): {metrics_dict.get('T', 'N/A'):.4f}\n")
                    f.write(f"  Normalized Energy (ξ): {metrics_dict.get('xi', 'N/A'):.4f}\n")
                    f.write(f"  Overall Efficiency (λ): {metrics_dict.get('lambda', 'N/A'):.4f}\n\n")

                f.write("Episode Statistics:\n")
                f.write(f"  Number of UAVs: {self.num_uavs}\n")
                f.write(f"  Episode Duration: {self.total_episode_time:.2f} seconds\n")
                f.write(f"  Total Steps: {len(self.trajectory_data)}\n")
                f.write(f"  Tasks Completed: {len(self.processed_tasks)}\n\n")

                f.write("Files Generated:\n")
                f.write(f"  Metrics: {metrics_file.name}\n")
                f.write(f"  Task Data: {taskdata_file.name}\n")
                f.write(f"  Trajectory: {trajectory_file.name}\n")

            print(f"✅ Summary saved to: {summary_file}")
            print(f"📊 Episode data successfully saved to {save_path}")

        except Exception as e:
            print(f"❌ Error saving episode data: {e}")
            import traceback
            traceback.print_exc()

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

    def print_current_state(self):
        """打印当前环境状态，用于调试"""
        print(f"\n=== Environment State ===")
        print(f"Action space: Spherical coordinates (azimuth, elevation, distance_ratio)")
        print(f"Total episode time: {self.total_episode_time:.2f}s")
        print(f"Current step: {self.current_step}")
        print(f"Trajectory records: {len(self.trajectory_data)}")
        print(f"UAV positions (normalized): {[coord.cpu().numpy().tolist() for coord in self.uav_normalized_coords]}")
        print(f"UAV positions (actual cm): {[coord.cpu().numpy().tolist() for coord in self.uav_coords]}")
        print(f"UAV times: {[f'{t.item():.2f}' for t in self.uav_times]}")
        print(f"UAV waiting times: {[f'{t.item():.2f}' for t in self.uav_total_waiting_times]}")
        print(f"Completed tasks: {len(self.processed_tasks)}")
        print(f"Loaded navigable points: {len(self.navigable_points)}")

    def get_uav_positions(self) -> List[List[float]]:
        """获取所有UAV的当前位置（实际坐标，厘米）"""
        return [self.uav_coords[i].cpu().numpy().tolist() for i in range(self.num_uavs)]

    def get_uav_normalized_positions(self) -> List[List[float]]:
        """获取所有UAV的当前位置（归一化坐标）"""
        return [self.uav_normalized_coords[i].cpu().numpy().tolist() for i in range(self.num_uavs)]

    def test_boundary_check(self, test_point: List[float]) -> bool:
        """测试边界检查功能"""
        return self._is_point_in_flyable_area(np.array(test_point))

    def get_nearest_points_info(self, uav_idx: int, k: int = 5) -> List[Dict[str, Any]]:
        """获取指定UAV附近最近的k个导航点信息（用于调试）"""
        uav_coords = self.uav_coords[uav_idx].cpu().numpy()
        return self._get_nearest_navigable_points(uav_coords, k)

    def get_trajectory_summary(self) -> Dict[str, Any]:
        """获取轨迹数据摘要（用于调试）"""
        if not self.trajectory_data:
            return {"message": "No trajectory data available"}

        return {
            "total_steps": len(self.trajectory_data),
            "episode_duration_s": self.total_episode_time,
            "first_step_time": self.trajectory_data[0]["timestamp"] if self.trajectory_data else 0,
            "last_step_time": self.trajectory_data[-1]["timestamp"] if self.trajectory_data else 0,
            "num_uavs": self.num_uavs,
            "action_space": "spherical_coordinates"
        }