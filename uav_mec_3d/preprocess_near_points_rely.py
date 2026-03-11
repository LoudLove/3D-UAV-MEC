# path_planner_encoded.py
# -*- coding: utf-8 -*-
import os
import glob
import json
import heapq
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.spatial import KDTree

from tqdm import tqdm
from typing import Optional
from tqdm import tqdm
import os, json, threading
from concurrent.futures import ThreadPoolExecutor
import math

# ============== 配置 ==============
# BASE_PATH = r"L:\python_projects\AirSim-main\PythonClient\multirotor\UE_output\UE_map_encode"
# BASE_PATH = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_100points\UE_map_encode_for_env_design"
# BASE_PATH = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_200points\UE_map_encode_for_env_design"
# BASE_PATH = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_300points\UE_map_encode_for_env_design"
# BASE_PATH = r"L:\python_projects\BenchMARL-main\two_basic_algorithms\env_400points\UE_map_encode_for_env_design"


#重新设计环境
# BASE_PATH = r"L:\python_projects\AirSim-main\PythonClient\multirotor\reconst\10\UE_map_encode"
# BASE_PATH = r"L:\python_projects\AirSim-main\PythonClient\multirotor\reconst\20\UE_map_encode"
# BASE_PATH = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario1\UE_map_encode"
# BASE_PATH = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario2\UE_map_encode"
BASE_PATH = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario3\UE_map_encode"
AREAID_MAP_JSON = "areaid_mapping.json"  # 位于 BASE_PATH 下

# =================================

# 颜色与区域风格（与你原版一致）
COLOR_MAP = {
    'yellow': '#FFD700', 'red': '#FF0000', 'green': '#00FF00',
    'brown': '#8B4513', 'blue': '#0000FF',
    'Yellow': '#FFD700', 'Red': '#FF0000', 'Green': '#00FF00',
    'Brown': '#8B4513', 'Blue': '#0000FF'
}
AREA_STYLES = {
    'warehouse': {'alpha': 0.15, 'edgecolor': 'red', 'linewidth': 1.0},
    'production_area': {'alpha': 0.20, 'edgecolor': 'blue', 'linewidth': 1.0},
    'outdoor': {'alpha': 0.10, 'edgecolor': 'green', 'linewidth': 1.0},
    'outside': {'alpha': 0.05, 'edgecolor': 'gray', 'linewidth': 0.5},
    'monitor': {'alpha': 0.25, 'edgecolor': 'orange', 'linewidth': 1.0},
    'command': {'alpha': 0.25, 'edgecolor': 'purple', 'linewidth': 1.0},
    'equipment': {'alpha': 0.20, 'edgecolor': 'brown', 'linewidth': 1.0},
    'default': {'alpha': 0.15, 'edgecolor': 'black', 'linewidth': 0.5}
}

def euler_to_rotation_matrix(pitch, yaw, roll):
    p = np.radians(pitch); y = np.radians(yaw); r = np.radians(roll)
    cos_p, sin_p = np.cos(p), np.sin(p); cos_y, sin_y = np.cos(y), np.sin(y)
    cos_r, sin_r = np.cos(r), np.sin(r)
    R_roll = np.array([[cos_r, 0, sin_r],[0, 1, 0],[-sin_r, 0, cos_r]])
    R_pitch = np.array([[1, 0, 0],[0, cos_p, -sin_p],[0, sin_p, cos_p]])
    R_yaw = np.array([[cos_y, -sin_y, 0],[sin_y, cos_y, 0],[0, 0, 1]])
    R = R_yaw @ R_pitch @ R_roll
    # Y翻转适配（左手系→右手系）
    flip_y = np.diag([1, -1, 1])
    return flip_y @ R @ flip_y

def get_rotated_cube_corners(center, size, rotation_matrix):
    x,y,z = center; dx,dy,dz = size[0]/2, size[1]/2, size[2]/2
    local_corners = np.array([
        [-dx,-dy,-dz],[dx,-dy,-dz],[dx,dy,-dz],[-dx,dy,-dz],
        [-dx,-dy,dz],[dx,-dy,dz],[dx,dy,dz],[-dx,dy,dz]
    ])
    rotated = (rotation_matrix @ local_corners.T).T
    rotated += np.array(center)
    return rotated

class PathPlanner:
    """
    迁移点：
    - 从 UE_map_encode 读取：*_points.jsonl 与 *_cubes.jsonl（每行一个 JSON 对象）
    - 字段已为字典：Location/Rotation/Scale，并自带 id/area_id/area_name
    - 起点/终点通过全局 id 指定；内部用 id->坐标 的索引直接查坐标
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.areas: Dict[str, dict] = {}
        self.area_boundaries: Dict[str, dict] = {}
        self.exit_points: Dict[str, List[List[float]]] = {}
        self.outside_points: List[List[float]] = []

        # 关键索引：由全局 id 直接查到坐标与区域
        self.id_to_point: Dict[int, dict] = {}  # {id: {'coords': [x,y,z], 'area_name': str}}

        # 新增：坐标到点信息的反向索引（用于路径点信息查找）
        self.coords_to_point: Dict[tuple, dict] = {}  # {(x,y,z): {'id': int, 'area_name': str, ...}}

        self.areaid_to_name: Dict[int, str] = self._load_areaid_mapping()

        self._load_all_areas()
        self._identify_exits_and_connections()

    def _read_points(self, file_path: str) -> List[dict]:
        pts = []
        for obj in self._read_jsonl_objects(file_path):
            loc = obj.get("Location", {})
            rot = obj.get("Rotation", {"P": 0, "Y": 0, "R": 0})
            scl = obj.get("Scale", {"X": 1, "Y": 1, "Z": 1})
            classname = obj.get("Classname", "unknown")
            gid = int(obj.get("id"))
            area_name = obj.get("area_name") or self.areaid_to_name.get(int(obj.get("area_id", 0)), "unknown")
            coords = [float(loc["X"]), -float(loc["Y"]), float(loc["Z"])]

            point_info = {
                "id": gid,
                "coords": coords,
                "color_class": classname,
                "location": {"X": float(loc["X"]), "Y": float(loc["Y"]), "Z": float(loc["Z"])},
                "rotation": {"P": float(rot["P"]), "Y": float(rot["Y"]), "R": float(rot["R"])},
                "scale": {"X": float(scl["X"]), "Y": float(scl["Y"]), "Z": float(scl["Z"])},
                "area_name": area_name
            }
            pts.append(point_info)

            # 建立双向索引
            self.id_to_point[gid] = {"coords": coords, "area_name": area_name}
            # 坐标索引（用元组作为key，便于精确匹配）
            coord_key = (float(coords[0]), float(coords[1]), float(coords[2]))
            self.coords_to_point[coord_key] = {
                "id": gid,
                "area_name": area_name,
                "color_class": classname
            }

        return pts

    # ---------- 数据装载 ----------
    def _load_areaid_mapping(self) -> Dict[int, str]:
        path = os.path.join(self.base_path, AREAID_MAP_JSON)
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)  # { "1": "command_room", ... }
        # 统一转 int key
        return {int(k): v for k, v in d.items()}

    def _read_jsonl_objects(self, file_path: str) -> List[dict]:
        """严格 JSONL：一行一个 JSON 对象；也兼容偶发连写的 }{"""
        objs: List[dict] = []
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        # 兼容连写
        text = text.replace('}\n{', '}\n{').replace('}{', '}\n{')
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except json.JSONDecodeError:
                # 再兜底一次
                line2 = line.replace(', }', '}').replace(',}', '}')
                objs.append(json.loads(line2))
        return objs

    def _read_cubes(self, file_path: str) -> List[dict]:
        cubes = []
        for obj in self._read_jsonl_objects(file_path):
            loc = obj.get("Location", {})
            rot = obj.get("Rotation", {"P":0,"Y":0,"R":0})
            scl = obj.get("Scale", {"X":1,"Y":1,"Z":1})
            classname = obj.get("Classname", "unknown")

            center = [float(loc["X"]), -float(loc["Y"]), float(loc["Z"])]
            # 与原逻辑一致：Scale 代表尺寸（米）→ 乘 100 转成厘米
            size = [float(scl["X"])*100.0, float(scl["Y"])*100.0, float(scl["Z"])*100.0]
            R = euler_to_rotation_matrix(float(rot["P"]), float(rot["Y"]), float(rot["R"]))
            corners = get_rotated_cube_corners(center, size, R)

            cubes.append({
                "center": center,
                "size": size,
                "rotation": {"P": float(rot["P"]), "Y": float(rot["Y"]), "R": float(rot["R"])},
                "rotation_matrix": R,
                "corners": corners,
                "classname": classname
            })
        return cubes

    def _load_all_areas(self):
        points_files = glob.glob(os.path.join(self.base_path, "*_points.jsonl"))
        area_names = sorted([os.path.basename(p).replace("_points.jsonl","") for p in points_files])
        print(f"发现区域: {area_names}")

        for area_name in area_names:
            pfile = os.path.join(self.base_path, f"{area_name}_points.jsonl")
            cfile = os.path.join(self.base_path, f"{area_name}_cubes.jsonl")

            print(f"正在加载区域: {area_name}")
            area_points = self._read_points(pfile) if os.path.exists(pfile) else []
            area_cubes  = self._read_cubes(cfile)  if os.path.exists(cfile)  else []

            # 分类（导航点与出入口点）
            navigable_points, exit_points = [], []
            for point in area_points:
                coords = point["coords"]
                if self._is_point_in_flyable_area(coords, area_cubes):
                    if point["color_class"] == "Blue":
                        exit_points.append(coords)
                    else:
                        navigable_points.append(coords)

            self.areas[area_name] = {
                "points": area_points,
                "cubes": area_cubes,
                "navigable_points": navigable_points,
                "exit_points": exit_points
            }
            self._calculate_area_boundary(area_name)
            print(f"  - 加载了 {len(area_points)} 点（{len(navigable_points)} 导航, {len(exit_points)} 出入口），{len(area_cubes)} 个飞行区域")

    def _identify_exits_and_connections(self):
        for area_name, area_data in self.areas.items():
            if area_name != "outside":
                self.exit_points[area_name] = area_data["exit_points"]
                print(f"区域 {area_name} 有 {len(area_data['exit_points'])} 个出入口点")
        if "outside" in self.areas:
            outside_points = self.areas["outside"]["points"]
            self.outside_points = [p["coords"] for p in outside_points if p["color_class"] == "Brown"]
            print(f"Outside 区域有 {len(self.outside_points)} 个连接点")
        print(f"总计识别出 {sum(len(v) for v in self.exit_points.values())} 个出入口点")

    # ---------- 空间几何 ----------
    def _is_point_in_flyable_area(self, point, cubes) -> bool:
        return any(self._is_point_in_cube(point, c) for c in cubes)

    def _is_point_in_cube(self, point, cube) -> bool:
        center = np.array(cube["center"])
        size = np.array(cube["size"])
        R = cube["rotation_matrix"]
        local = R.T @ (np.array(point) - center)
        half = size / 2.0
        return np.all(np.abs(local) <= half)

    def _calculate_area_boundary(self, area_name):
        area = self.areas[area_name]
        all_coords = []
        for pt in area["points"]:
            all_coords.append(pt["coords"])
        for c in area["cubes"]:
            all_coords.extend(c["corners"])
        if all_coords:
            all_coords = np.array(all_coords)
            mn = np.min(all_coords, axis=0); mx = np.max(all_coords, axis=0)
            self.area_boundaries[area_name] = {"min": mn, "max": mx, "center": (mn+mx)/2.0}

    def _get_point_area(self, point):
        for area_name, boundary in self.area_boundaries.items():
            if area_name == "outside":
                continue
            if np.all(point >= boundary["min"] - 200) and np.all(point <= boundary["max"] + 200):
                if self._is_point_in_flyable_area(point, self.areas[area_name]["cubes"]):
                    return area_name
        if "outside" in self.area_boundaries:
            b = self.area_boundaries["outside"]
            if np.all(point >= b["min"] - 200) and np.all(point <= b["max"] + 200):
                if self._is_point_in_flyable_area(point, self.areas["outside"]["cubes"]):
                    return "outside"
        return None

    # ---------- A* 相关 ----------
    @staticmethod
    def _heuristic(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _is_path_clear(self, p1, p2, cubes, step_size=50):
        p1 = np.array(p1); p2 = np.array(p2)
        dist = np.linalg.norm(p2 - p1)
        if dist < step_size: return True
        steps = int(dist / step_size)
        for i in range(1, steps):
            t = p1 + (p2 - p1) * i / steps
            if not self._is_point_in_flyable_area(t, cubes):
                return False
        return True

    def _a_star_in_area(self, start, goal, area_name, max_neighbors=20):

        # print("start: ",start," ,goal: ",goal," area_name: ",area_name)

        if area_name not in self.areas: return []
        if area_name == "outside":
            navigable = self.outside_points.copy()
        else:
            navigable = self.areas[area_name]["navigable_points"] + self.areas[area_name]["exit_points"]
        cubes = self.areas[area_name]["cubes"]
        if not navigable:
            print(f"区域 {area_name} 没有可用导航点"); return []
        if not self._is_point_in_flyable_area(start, cubes):
            print(f"起点 {start} 不在 {area_name} 可飞行区域内"); return []
        if not self._is_point_in_flyable_area(goal, cubes):
            print(f"终点 {goal} 不在 {area_name} 可飞行区域内"); return []

        all_points = navigable + [start, goal]
        tree = KDTree(all_points)
        si, gi = len(all_points)-2, len(all_points)-1

        open_set = [(0, si, [si])]
        closed = set()
        gscore = {si: 0.0}

        # print(f"在区域 {area_name} 中规划路径，使用 {len(navigable)} 个导航点")
        while open_set:
            f, idx, path = heapq.heappop(open_set)
            if idx in closed: continue
            closed.add(idx)
            cur = all_points[idx]
            if idx == gi:
                return [all_points[i] for i in path]
            k = min(max_neighbors, len(all_points))
            dists, neighs = tree.query(cur, k=k)
            # 当 k=1 时返回标量，统一成可迭代
            if np.isscalar(dists): dists = [dists]; neighs = [neighs]
            for dist, nb in zip(dists, neighs):
                if nb in closed or dist > 20000:  # 邻居距离阈值
                    continue
                nxt = all_points[nb]
                if not self._is_path_clear(cur, nxt, cubes):
                    continue
                tentative = gscore[idx] + float(dist)
                if nb not in gscore or tentative < gscore[nb]:
                    gscore[nb] = tentative
                    fscore = tentative + self._heuristic(nxt, goal)
                    heapq.heappush(open_set, (fscore, nb, path + [nb]))
        print(f"在区域 {area_name} 中未找到路径")
        return []

    def _a_star_in_area_debug(self, start, goal, area_name, max_neighbors=20,
                        debug=True,  # 打开/关闭调试输出
                        debug_step_limit=200,  # 最多打印多少次节点扩展
                        sample_neigh_print=10,  # 每次扩展最多打印多少个邻居决策
                        include_exits=True,  # 是否把 exit_points 也塞进本区域 KDTree
                        dist_threshold=2000  # 邻居距离阈值（原逻辑）
                        ):
        import heapq
        import numpy as np
        from scipy.spatial import KDTree

        def _fmt(p):
            return f"[{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}]"

        if area_name not in self.areas:
            if debug: print(f"[A*] 区域 {area_name} 不存在")
            return []

        # 1) 组装点集（保留你原来的行为；可用 include_exits=False 只用导航点）
        if area_name == "outside":
            navigable_raw = list(self.outside_points) if hasattr(self, "outside_points") else []
        else:
            base_nav = list(self.areas[area_name]["navigable_points"])
            exits = list(self.areas[area_name]["exit_points"]) if include_exits else []
            navigable_raw = base_nav + exits

        cubes = self.areas[area_name]["cubes"]

        # 2) 起终点域校验
        if not navigable_raw:
            print(f"区域 {area_name} 没有可用导航点");
            return []
        if not self._is_point_in_flyable_area(start, cubes):
            print(f"起点 {start} 不在 {area_name} 可飞行区域内");
            return []
        if not self._is_point_in_flyable_area(goal, cubes):
            print(f"终点 {goal} 不在 {area_name} 可飞行区域内");
            return []

        # 3) 去重（完全相同坐标的点只保留一个）
        uniq = []
        seen = set()
        for p in navigable_raw:
            key = (float(p[0]), float(p[1]), float(p[2]))
            if key not in seen:
                uniq.append(p)
                seen.add(key)
        dup_count = len(navigable_raw) - len(uniq)
        navigable = uniq

        # 4) 组装 KDTree 数据
        all_points = list(navigable) + [start, goal]
        tree = KDTree(all_points)
        si, gi = len(all_points) - 2, len(all_points) - 1

        open_set = [(0.0, si, [si])]
        gscore = {si: 0.0}
        closed = set()

        # 统计/诊断数据
        expanded = 0
        max_open = 1
        skip_stats = {"closed": 0, "self": 0, "far": 0, "not_flyable": 0, "blocked": 0}
        nearest = {"idx": si, "h": self._heuristic(start, goal)}  # 离目标最近的已达节点
        first_blocked = None  # 记录第一条被 _is_path_clear 拦下的线段

        if debug:
            print(f"在区域 {area_name} 中规划路径，使用 {len(navigable)} 个导航点"
                  f"{' + 出入口点' if (area_name != 'outside' and include_exits) else ''}")
            if dup_count > 0:
                print(f"[A*|告警] 输入点集中检测到 {dup_count} 个重复点，已去重。")
            print(f"[A*] start={_fmt(start)}, goal={_fmt(goal)}, k(max_neighbors)={max_neighbors}, "
                  f"dist阈值={dist_threshold}")

        # 5) 主循环
        while open_set:
            if len(open_set) > max_open: max_open = len(open_set)
            f, idx, path = heapq.heappop(open_set)
            if idx in closed:
                skip_stats["closed"] += 1
                continue

            closed.add(idx)
            cur = all_points[idx]
            g = gscore[idx]
            h = self._heuristic(cur, goal)
            if h < nearest["h"]:
                nearest.update({"idx": idx, "h": h})

            # 命中目标
            if idx == gi:
                if debug:
                    print(
                        f"[A*] 命中目标，步数={len(path) - 1}, 展开节点={expanded}, 访问={len(closed)}, max_open={max_open}")
                return [all_points[i] for i in path]

            # 限流的调试打印
            if debug and expanded < debug_step_limit:
                print(f"\n[A*] 扩展#{expanded + 1} idx={idx} pos={_fmt(cur)} g={g:.1f} h={h:.1f} f={f:.1f}")

            # KDTree 查询邻居
            k = min(max_neighbors, len(all_points))
            dists, neighs = tree.query(cur, k=k)
            if np.isscalar(dists):  # k=1 时为标量，铺平为列表
                dists, neighs = [float(dists)], [int(neighs)]
            else:
                dists = list(map(float, dists))
                neighs = list(map(int, neighs))

            # 遍历邻居
            printed = 0
            for dist, nb in zip(dists, neighs):
                if nb == idx:
                    skip_stats["self"] += 1
                    continue
                if nb in closed:
                    skip_stats["closed"] += 1
                    continue
                if dist > dist_threshold:
                    skip_stats["far"] += 1
                    continue

                nxt = all_points[nb]

                # 可选：只扩展本域内的点（避免“幽灵点”误混）
                if not self._is_point_in_flyable_area(nxt, cubes):
                    skip_stats["not_flyable"] += 1
                    if debug and expanded < debug_step_limit and printed < sample_neigh_print:
                        print(f"  - 跳过 nb={nb:>4} pos={_fmt(nxt)} dist={dist:.1f} 原因:not_flyable")
                        printed += 1
                    continue

                if not self._is_path_clear(cur, nxt, cubes):
                    skip_stats["blocked"] += 1
                    if first_blocked is None:
                        first_blocked = (cur, nxt)
                    if debug and expanded < debug_step_limit and printed < sample_neigh_print:
                        print(f"  - 跳过 nb={nb:>4} pos={_fmt(nxt)} dist={dist:.1f} 原因:blocked")
                        printed += 1
                    continue

                tentative = g + dist
                if nb not in gscore or tentative < gscore[nb]:
                    gscore[nb] = tentative
                    fscore = tentative + self._heuristic(nxt, goal)
                    heapq.heappush(open_set, (fscore, nb, path + [nb]))
                    if debug and expanded < debug_step_limit and printed < sample_neigh_print:
                        print(f"  + 接受 nb={nb:>4} pos={_fmt(nxt)} dist={dist:.1f} g'={tentative:.1f} f'={fscore:.1f}")
                        printed += 1

            expanded += 1

        # 6) 无路径 → 打印总结诊断
        print(f"在区域 {area_name} 中未找到路径")
        if debug:
            print("\n[A*] 失败诊断：")
            print(f"  - 展开节点数: {expanded}")
            print(f"  - 访问集合大小: {len(closed)}")
            print(f"  - open_set 最大长度: {max_open}")
            print(f"  - 跳过统计: {skip_stats}")
            # 离目标最近的可达点
            nb_idx = nearest["idx"];
            nb_p = all_points[nb_idx]
            print(f"  - 离目标最近的已达点: idx={nb_idx}, pos={_fmt(nb_p)}, h={nearest['h']:.1f}")
            # 第一条被阻挡的线段（如有）
            if first_blocked:
                a, b = first_blocked
                print(f"  - 第一条被 _is_path_clear 拦下的段: {_fmt(a)} -> {_fmt(b)}")
            # 进一步提示
            if skip_stats["blocked"] > 0 and skip_stats["not_flyable"] == 0:
                print("  * 主要原因似乎是障碍阻挡（blocked），可检查 _is_path_clear 判定或放宽阈值。")
            elif skip_stats["not_flyable"] > 0:
                print("  * 有邻居点不在本区域可飞域内（not_flyable），检查点归属或 'exit_points' 误混。")
            elif skip_stats["far"] > 0 and expanded == 0:
                print("  * 最近邻都超过距离阈值，尝试增大 dist_threshold 或 max_neighbors。")
            elif len(closed) <= 1:
                print("  * 图可能不连通，或 start 附近没有可扩展邻居。")

        return []

    def _best_exit_pair(self, start_area, goal_area):
        if start_area not in self.exit_points or goal_area not in self.exit_points:
            return None, None
        start_exits = self.exit_points[start_area]; goal_ent = self.exit_points[goal_area]
        if not start_exits or not goal_ent: return None, None
        best = (float("inf"), None, None)
        for a in start_exits:
            for b in goal_ent:
                d = self._heuristic(a, b)
                if d < best[0]: best = (d, a, b)
        return best[1], best[2]

    # ---------- 对外 API ----------
    def coords_from_global_id(self, global_id: int) -> Tuple[List[float], str]:
        """
        由全局 id 反查坐标与区域名。
        注意：直接使用编码文件中每条记录的 id 做索引，避免“再排序、再取 idx”的脆弱性。
        """
        # print(self.id_to_point)
        hit = self.id_to_point.get(int(global_id))
        if not hit:
            raise KeyError(f"未在 points 中找到 id={global_id} 的点")
        return hit["coords"], hit["area_name"]

    def _nearest_outside_point(self, ref_point):
        """
        在 self.outside_points 中找到距离 ref_point 最近的一个点。
        返回该点坐标 list[float]；若 outside 为空，返回 None。
        """
        if not self.outside_points:
            return None
        ref = np.array(ref_point, dtype=float)
        cand = np.array(self.outside_points, dtype=float)
        dists = np.linalg.norm(cand - ref, axis=1)
        idx = int(np.argmin(dists))
        return self.outside_points[idx]

    def _concat_paths(self, *segments):
        """
        将若干段路径顺序拼接，自动去除相邻段首尾重复点。
        例如：concat([A,B,C], [C,D], [D,E]) => [A,B,C,D,E]
        """
        out = []
        for seg in segments:
            if not seg:
                continue
            if not out:
                out.extend(seg)
            else:
                # 避免重复衔接点
                if np.allclose(out[-1], seg[0]):
                    out.extend(seg[1:])
                else:
                    out.extend(seg)
        return out

    def _as_segment(self, p1, p2):
        """
        生成一段由 p1->p2 的简单“直线段”（仅包含两个端点，留待上层可视化为折线）。
        注意：按你的保证，不做碰撞检测。
        """
        return [list(map(float, p1)), list(map(float, p2))]

    def plan_full_path(self, start_coords, goal_coords):
        """
        跨区时，若出入口不在 outside 内：
        - 先在起点区域内：start -> s_exit
        - 再直连：s_exit -> m_s（outside 最近点）
        - 再在 outside 内：m_s -> m_g
        - 再直连：m_g -> g_ent
        - 最后在目标区域内：g_ent -> goal
        """
        sa = self._get_point_area(np.array(start_coords))
        ga = self._get_point_area(np.array(goal_coords))
        # print(f"起点区域: {sa}, 终点区域: {ga}")
        if not sa or not ga:
            print("无法确定起点或终点所在区域")
            return []

        # 同区：直接在本区 A*。
        if sa == ga:
            # print(f"在区域 {sa} 内规划路径")
            return self._a_star_in_area(start_coords, goal_coords, sa)

        # 跨区：需要通过 outside 桥接。
        # print(f"规划从 {sa} 到 {ga} 的跨区域路径")
        s_exit, g_ent = self._best_exit_pair(sa, ga)
        if s_exit is None or g_ent is None:
            print(f"未找到 {sa} 与 {ga} 的连接出入口")
            return []

        # 1) 起点区域内：start -> s_exit
        # print("规划从起点到出口的路径...")
        p_start_to_exit = self._a_star_in_area(start_coords, s_exit, sa)
        if not p_start_to_exit:
            print("无法到达起始区域出口")
            return []

        # 2) 找 outside 的桥接端点：m_s 与 m_g
        if not self.outside_points:
            print("outside 区域没有可用连接点，无法桥接")
            return []

        m_s = self._nearest_outside_point(s_exit)
        m_g = self._nearest_outside_point(g_ent)
        # print(f"选择 outside 桥接点: m_s={m_s}, m_g={m_g}")

        # 3) 直连：s_exit -> m_s（不做碰撞检测，按你的保证）
        seg_exit_to_ms = self._as_segment(s_exit, m_s)

        # 4) outside 内：m_s -> m_g
        # print("规划 outside 桥接段路径（m_s -> m_g）...")
        p_ms_to_mg = self._a_star_in_area(m_s, m_g, "outside")
        if not p_ms_to_mg:
            print("无法在 outside 区域中找到从 m_s 到 m_g 的路径")
            return []

        # 5) 直连：m_g -> g_ent（不做碰撞检测）
        seg_mg_to_gent = self._as_segment(m_g, g_ent)

        # 6) 目标区内：g_ent -> goal
        # print("规划从入口到终点的路径...")
        p_gent_to_goal = self._a_star_in_area(g_ent, goal_coords, ga)
        if not p_gent_to_goal:
            print("无法从目标区域入口到达终点")
            return []

        # 最终拼接（自动去重衔接点）
        full = self._concat_paths(
            p_start_to_exit,
            seg_exit_to_ms,
            p_ms_to_mg,
            seg_mg_to_gent,
            p_gent_to_goal
        )
        return full

    def plan_full_path_old(self, start_coords, goal_coords):
        sa = self._get_point_area(np.array(start_coords))
        ga = self._get_point_area(np.array(goal_coords))
        print(f"起点区域: {sa}, 终点区域: {ga}")
        if not sa or not ga: print("无法确定起点或终点所在区域"); return []
        if sa == ga:
            print(f"在区域 {sa} 内规划路径")
            return self._a_star_in_area(start_coords, goal_coords, sa)
        print(f"规划从 {sa} 到 {ga} 的跨区域路径")
        s_exit, g_ent = self._best_exit_pair(sa, ga)
        if s_exit is None or g_ent is None:
            print(f"未找到 {sa} 与 {ga} 的连接出入口"); return []
        full = []
        print("规划从起点到出口..."); p1 = self._a_star_in_area(start_coords, s_exit, sa)
        if not p1: print("无法到达起始区域出口"); return []
        full.extend(p1)
        print("规划跨区域连接..."); p2 = self._a_star_in_area(s_exit, g_ent, "outside")
        print(s_exit," ",g_ent)
        if not p2: print("无法在 outside 找到连接路径"); return []
        full.extend(p2[1:])
        print("规划从入口到终点..."); p3 = self._a_star_in_area(g_ent, goal_coords, ga)
        if not p3: print("无法从目标区域入口到达终点"); return []
        full.extend(p3[1:])
        return full

    # ---------- 可视化 ----------
    def visualize_path(self, path, start, goal):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # 解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        if not path:
            print("没有找到有效路径"); return
        fig = plt.figure(figsize=(20,16))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制所有区域的飞行区域（cube）
        for area_name, area_data in self.areas.items():
            style = AREA_STYLES.get(area_name, None)
            if style is None:
                style = next((AREA_STYLES[k] for k in AREA_STYLES if k in area_name.lower()), AREA_STYLES['default'])
            for cube in area_data['cubes']:
                corners = cube['corners']
                faces = [
                    [corners[0], corners[1], corners[2], corners[3]],
                    [corners[4], corners[5], corners[6], corners[7]],
                    [corners[0], corners[1], corners[5], corners[4]],
                    [corners[2], corners[3], corners[7], corners[6]],
                    [corners[1], corners[2], corners[6], corners[5]],
                    [corners[0], corners[3], corners[7], corners[4]]
                ]
                poly = Poly3DCollection(faces,
                                        facecolors=style['edgecolor'],
                                        linewidths=style['linewidth'],
                                        edgecolors=style['edgecolor'],
                                        alpha=style['alpha'])
                ax.add_collection3d(poly)

        # 出入口点（Blue）
        for i, (area_name, exits) in enumerate(self.exit_points.items()):
            if exits:
                arr = np.array(exits)
                ax.scatter(arr[:,0], arr[:,1], arr[:,2], c='blue', s=60, marker='D', alpha=0.8,
                           label=f'{area_name} 出入口' if i == 0 else "")

        # outside 连接（Brown）
        if self.outside_points:
            arr = np.array(self.outside_points)
            ax.scatter(arr[:,0], arr[:,1], arr[:,2], c='brown', s=30, marker='o', alpha=0.6, label='Outside连接点')

        # 路径、起终点
        if len(path) > 1:
            arr = np.array(path)
            ax.plot(arr[:,0], arr[:,1], arr[:,2], 'r-', linewidth=4, label='规划路径', alpha=0.9)
            ax.scatter(arr[:,0], arr[:,1], arr[:,2], c='red', s=50, alpha=0.8, edgecolors='white', linewidth=1)
        ax.scatter(*start, c='green', s=300, marker='o', label='起点', edgecolors='white', linewidth=2)
        ax.scatter(*goal, c='purple', s=300, marker='s', label='终点', edgecolors='white', linewidth=2)

        # 视域
        all_coords = []
        for ad in self.areas.values():
            for p in ad['points']:
                all_coords.append(p['coords'])
        if all_coords:
            all_coords = np.array(all_coords)
            mn = np.min(all_coords, axis=0); mx = np.max(all_coords, axis=0)
            center = (mn+mx)/2; ranges = mx-mn; mr = np.max(ranges)
            if mr > 0:
                margin = mr * 0.05; half = mr/2 + margin
                ax.set_xlim([center[0]-half, center[0]+half])
                ax.set_ylim([center[1]-half, center[1]+half])
                ax.set_zlim([center[2]-half, center[2]+half])

        ax.set_xlabel('X (cm)'); ax.set_ylabel('Y (cm)'); ax.set_zlabel('Z (cm)')
        ax.set_title('3D UAV路径规划结果 (数据源: UE_map_encode)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(); plt.show()

        # 文本总结
        total_dist = sum(self._heuristic(path[i], path[i+1]) for i in range(len(path)-1)) if len(path)>1 else 0
        print(f"\n路径规划成功！")
        print(f"路径长度: {len(path)} 个点")
        print(f"总距离: {total_dist/100:.2f} 米")
        seq = []
        for pt in path:
            a = self._get_point_area(np.array(pt))
            if a and (not seq or seq[-1] != a):
                seq.append(a)
        if seq:
            print(f"路径经过区域: {' -> '.join(seq)}")

    def _iter_navigable_points(self, include_exits: bool = False):
        """
        返回 [(id, coords_dict, area_name, color_class), ...]
        - 默认排除出入口点(Blue)；include_exits=True 时包含
        - 仅返回在可飞行区域中的点
        - 排除 outside 区域的点（outside区域仅用于跨区域桥接）
        """
        out = []
        for area_name, area in self.areas.items():
            # 跳过 outside 区域
            if area_name == "outside":
                continue

            cubes = area["cubes"]
            for p in area["points"]:
                cc = str(p.get("color_class", "unknown"))
                if (not include_exits) and cc == "Blue":
                    continue
                coords = p["coords"]
                if self._is_point_in_flyable_area(coords, cubes):
                    out.append((
                        int(p["id"]),
                        {"X": float(coords[0]), "Y": float(coords[1]), "Z": float(coords[2])},
                        str(area_name),
                        cc
                    ))
        # 去重（按 id）
        uniq, seen = [], set()
        for item in out:
            if item[0] not in seen:
                uniq.append(item)
                seen.add(item[0])
        return uniq

    @staticmethod
    def _to_xyz_dict_list(path):
        return [{"X": float(p[0]), "Y": float(p[1]), "Z": float(p[2])} for p in path]

    def _path_length(self, path):
        if not path or len(path) < 2:
            return 0.0
        return float(sum(self._heuristic(path[i], path[i + 1]) for i in range(len(path) - 1)))

    def _find_point_info_by_coords(self, coords, tolerance=1e-3):
        """
        根据坐标查找点信息，支持容差匹配

        Args:
            coords: [x, y, z] 坐标
            tolerance: 容差范围

        Returns:
            dict: {'id': int, 'area_name': str, 'color_class': str} 或 None
        """
        coord_key = (float(coords[0]), float(coords[1]), float(coords[2]))

        # 精确匹配
        if coord_key in self.coords_to_point:
            return self.coords_to_point[coord_key]

        # 容差匹配
        target = np.array(coords, dtype=float)
        for stored_coords, info in self.coords_to_point.items():
            stored = np.array(stored_coords, dtype=float)
            if np.linalg.norm(target - stored) <= tolerance:
                return info

        return None

    def _generate_enhanced_path_info(self, path, start_coords, goal_coords):
        """
        为路径生成增强信息，包括点ID、区域、累积长度等

        Args:
            path: 路径坐标列表 [[x,y,z], ...]
            start_coords: 起点坐标
            goal_coords: 终点坐标

        Returns:
            list: 增强路径信息列表
        """
        if not path:
            return []

        enhanced_path = []
        cumulative_length = 0.0

        for i, point_coords in enumerate(path):
            # 查找点信息
            point_info = self._find_point_info_by_coords(point_coords)

            # 如果找不到对应的点信息，生成默认信息
            if point_info is None:
                # 判断是否是起点或终点
                if np.allclose(point_coords, start_coords, atol=1e-3):
                    area_name = self._get_point_area(np.array(start_coords)) or "unknown"
                    point_info = {
                        "id": -1,  # 特殊ID表示起点
                        "area_name": area_name,
                        "color_class": "start_point"
                    }
                elif np.allclose(point_coords, goal_coords, atol=1e-3):
                    area_name = self._get_point_area(np.array(goal_coords)) or "unknown"
                    point_info = {
                        "id": -2,  # 特殊ID表示终点
                        "area_name": area_name,
                        "color_class": "goal_point"
                    }
                else:
                    # 可能是中间生成的路径点
                    area_name = self._get_point_area(np.array(point_coords)) or "unknown"
                    point_info = {
                        "id": -(i + 100),  # 负数ID表示临时生成的点
                        "area_name": area_name,
                        "color_class": "intermediate_point"
                    }

            # 计算到当前点的累积距离
            if i > 0:
                segment_length = self._heuristic(path[i - 1], point_coords)
                cumulative_length += segment_length

            # 构建增强点信息
            enhanced_point = {
                "point_id": point_info["id"],
                "point_pos": {
                    "X": float(point_coords[0]),
                    "Y": float(point_coords[1]),
                    "Z": float(point_coords[2])
                },
                "point_area": point_info["area_name"],
                "prefix_sum_path_length": float(cumulative_length)
            }

            enhanced_path.append(enhanced_point)

        return enhanced_path

    @staticmethod
    def _to_enhanced_path_dict_list(path, start_coords, goal_coords, planner_instance):
        """
        将路径转换为增强格式的字典列表

        Args:
            path: 路径坐标列表
            start_coords: 起点坐标
            goal_coords: 终点坐标
            planner_instance: PathPlanner实例

        Returns:
            list: 增强路径信息列表
        """
        return planner_instance._generate_enhanced_path_info(path, start_coords, goal_coords)

    def run_exhaustive_tests(
            self,
            output_jsonl_path: str,
            include_exits: bool = False,
            ordered: bool = False,
            max_pairs: Optional[int] = None,
            flush_every: int = 500
    ):
        """
        全量两两配对测试并保存为 JSONL：
        - include_exits: 是否把 Blue 出入口点也纳入“导航点”集合
        - ordered: True 时跑有序对 (i->j, j->i)，False 时跑组合（无重复）
        - max_pairs: 仅跑前 N 对，便于抽样/预跑；None 为全部
        - flush_every: 每写入多少条强制 flush

        JSONL 字段：
        {
          "start_point_id": int,
          "start_point_pos": {"X":..., "Y":..., "Z":...},
          "end_point_id": int,
          "end_point_pos": {"X":..., "Y":..., "Z":...},
          "path": [{"X":..., "Y":..., "Z":...}, ...],
          "length": float(厘米；未找到路径为 -1)
        }
        """
        nav = self._iter_navigable_points(include_exits=include_exits)
        n = len(nav)
        if n < 2:
            print("可导航点不足 2 个，无法测试。")
            return

        if ordered:
            total = n * (n - 1)

            def pair_iter():
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        yield i, j
        else:
            total = n * (n - 1) // 2

            def pair_iter():
                for i in range(n):
                    for j in range(i + 1, n):
                        yield i, j

        if max_pairs is not None:
            total = min(total, max_pairs)

        print(f"可导航点数: {n}；将测试对数: {total}（{'有序' if ordered else '组合'}）")
        ok_cnt = 0
        fail_cnt = 0

        # 为了更快的区域判定，提前缓存 id->coords
        id2coords = {pid: (pos["X"], pos["Y"], pos["Z"]) for (pid, pos, *_rest) in nav}

        wrote = 0
        with open(output_jsonl_path, "w", encoding="utf-8") as f, tqdm(total=total, desc="路径测试",
                                                                       unit="pair") as pbar:
            for k, (i, j) in enumerate(pair_iter()):
                if (max_pairs is not None) and (k >= max_pairs):
                    break

                sid, spos, *_ = nav[i]
                gid, gpos, *_ = nav[j]
                start = id2coords[sid]
                goal = id2coords[gid]

                try:
                    path = self.plan_full_path(start, goal)
                    if path:
                        length = self._path_length(path)
                        ok_cnt += 1
                    else:
                        length = -1.0
                        fail_cnt += 1

                    rec = {
                        "start_point_id": sid,
                        "start_point_pos": spos,
                        "end_point_id": gid,
                        "end_point_pos": gpos,
                        "path": self._to_xyz_dict_list(path),
                        "length": length
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    wrote += 1
                    if wrote % flush_every == 0:
                        f.flush()
                except Exception as e:
                    # 记录异常为失败
                    rec = {
                        "start_point_id": sid,
                        "start_point_pos": spos,
                        "end_point_id": gid,
                        "end_point_pos": gpos,
                        "path": [],
                        "length": -1.0,
                        "error": repr(e)
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    wrote += 1
                    fail_cnt += 1

                pbar.set_postfix({"ok": ok_cnt, "fail": fail_cnt})
                pbar.update(1)

        print(f"完成：成功 {ok_cnt} 对，失败 {fail_cnt} 对。结果已写入：{output_jsonl_path}")

    def run_exhaustive_tests_save_jsonl_in_process(
        self,
        output_jsonl_path: str,
        include_exits: bool = False,
        ordered: bool = False,
        max_pairs: Optional[int] = None,
        flush_every: int = 500,
        checkpoint_path: Optional[str] = None,
    ):
        """
        全量两两配对测试并保存为 JSONL（边算边写，支持断点续跑）。

        - include_exits: 是否把 Blue 出入口点也纳入“导航点”集合
        - ordered: True 时跑有序对 (i->j, j->i)，False 时跑组合（无重复）
        - max_pairs: 仅跑前 N 对；None 为全部
        - flush_every: 每写入多少条执行 flush()+fsync() 强制落盘
        - checkpoint_path: 断点文件路径；默认 output_jsonl_path + ".ckpt"

        JSONL 每行：
        {
          "start_point_id": int,
          "start_point_pos": {"X":..., "Y":..., "Z":...},
          "end_point_id": int,
          "end_point_pos": {"X":..., "Y":..., "Z":...},
          "path": [{"X":..., "Y":..., "Z":...}, ...],
          "length": float  # 厘米；未找到路径为 -1
        }
        """
        # --- 导航点集合 ---
        nav = self._iter_navigable_points(include_exits=include_exits)
        n = len(nav)
        if n < 2:
            print("可导航点不足 2 个，无法测试。")
            return

        # --- 成对生成器（确定性顺序，便于断点续跑） ---
        if ordered:
            total_all = n * (n - 1)

            def pair_at(index: int):
                # 映射线性 index -> (i, j)
                i, j = divmod(index, n - 1)
                # 把 j 映射到 [0..n) \ {i}
                if j >= i:
                    j += 1
                return i, j

            def total_pairs():
                return total_all
        else:
            total_all = n * (n - 1) // 2

            def pair_at(index: int):
                # 组合序编号到(i,j)（字典序，i<j）
                # 反推：找到 i 使得 index < 累加(n-1-i)
                # 累加和 S(i) = (n-1) + (n-2) + ... + (n-1-i) = i*(2n-i-1)/2
                # 这里用线性推进更稳妥（n 不太夸张时足够快）
                rem = index
                i = 0
                span = n - 1
                while rem >= span:
                    rem -= span
                    i += 1
                    span -= 1
                j = i + 1 + rem
                return i, j

            def total_pairs():
                return total_all

        if max_pairs is not None:
            total_target = min(total_all, max_pairs)
        else:
            total_target = total_all

        # --- 断点文件 ---
        if checkpoint_path is None:
            checkpoint_path = output_jsonl_path + ".ckpt"

        start_index = 0
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as cf:
                    start_index = int(cf.read().strip() or "0")
                print(f"发现断点：从第 {start_index} 对继续。")
            except Exception:
                print("断点文件损坏，忽略并从头开始。")
                start_index = 0

        # 若输出文件存在且没有断点，从头覆盖写（避免混淆）
        # 若想“无断点也续写”，把模式改为 'a' 并自行承担重复风险。
        file_mode = "a" if start_index > 0 else "w"

        print(f"可导航点数: {n}；计划测试对数: {total_target}（{'有序' if ordered else '组合'}）")
        ok_cnt = 0
        fail_cnt = 0

        # 快速索引
        id2coords = {pid: (pos["X"], pos["Y"], pos["Z"]) for (pid, pos, *_rest) in nav}

        # 工具函数
        def save_ckpt(k: int):
            try:
                with open(checkpoint_path, "w", encoding="utf-8") as cf:
                    cf.write(str(k))
            except Exception:
                pass  # 断点保存失败不应中断主流程

        wrote_since_flush = 0

        # --- 主循环（边算边写 + 强制落盘 + 断点） ---
        try:
            with open(output_jsonl_path, file_mode, encoding="utf-8", buffering=1) as f, \
                    tqdm(total=total_target - start_index, desc="路径测试", unit="pair") as pbar:

                # 从断点继续
                k = start_index
                while k < total_target:
                    i, j = pair_at(k)

                    sid, spos, *_ = nav[i]
                    gid, gpos, *_ = nav[j]
                    start = id2coords[sid]
                    goal = id2coords[gid]

                    try:
                        path = self.plan_full_path(start, goal)
                        if path:
                            length = self._path_length(path)
                            ok_cnt += 1
                        else:
                            length = -1.0
                            fail_cnt += 1

                        rec = {
                            "start_point_id": sid,
                            "start_point_pos": spos,
                            "end_point_id": gid,
                            "end_point_pos": gpos,
                            "path": self._to_xyz_dict_list(path),
                            "length": length
                        }
                    except Exception as e:
                        rec = {
                            "start_point_id": sid,
                            "start_point_pos": spos,
                            "end_point_id": gid,
                            "end_point_pos": gpos,
                            "path": [],
                            "length": -1.0,
                            "error": repr(e)
                        }
                        fail_cnt += 1

                    # 立刻写入一行
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    wrote_since_flush += 1

                    # 定期强制落盘 + 保存断点
                    if wrote_since_flush >= flush_every:
                        f.flush()
                        try:
                            os.fsync(f.fileno())
                        except Exception:
                            pass
                        save_ckpt(k + 1)  # 下一个将要处理的线性索引
                        wrote_since_flush = 0

                    pbar.set_postfix({"ok": ok_cnt, "fail": fail_cnt})
                    pbar.update(1)
                    k += 1

                # 收尾：flush+fsync+断点写到“已完成”
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
                save_ckpt(k)

        except KeyboardInterrupt:
            # CTRLC 时也落盘并保存断点
            try:
                save_ckpt(k)
            finally:
                print("\n手动中断，已保存断点。稍后可继续运行。")
                return
        except Exception as e:
            # 其他异常：保存断点便于排查后续续跑
            try:
                save_ckpt(k)
            finally:
                raise e

        print(f"完成：成功 {ok_cnt} 对，失败 {fail_cnt} 对。结果已写入：{output_jsonl_path}")
        print(f"断点文件：{checkpoint_path}（可删除以重头开始）")

    def run_exhaustive_tests_concurrent(
        self,
        output_jsonl_path: str,
        include_exits: bool = False,
        ordered: bool = False,
        max_pairs: Optional[int] = None,
        flush_every: int = 500,
        checkpoint_path: Optional[str] = None,
        num_workers: int = 8,   # >1 启用多线程
    ):
        """
        全量两两配对测试（多线程可选），边算边写，支持断点续跑。
        修改为保存增强格式的路径信息。
        """

        # --- 导航点集合 ---
        nav = self._iter_navigable_points(include_exits=include_exits)
        n = len(nav)
        if n < 2:
            print("可导航点不足 2 个，无法测试。")
            return

        # --- 索引映射（线性 index -> (i, j)），保持确定性顺序 ---
        if ordered:
            total_all = n * (n - 1)

            def pair_at(index: int):
                i, j = divmod(index, n - 1)
                if j >= i:
                    j += 1
                return i, j
        else:
            total_all = n * (n - 1) // 2

            def pair_at(index: int):
                rem = index
                i = 0
                span = n - 1
                while rem >= span:
                    rem -= span
                    i += 1
                    span -= 1
                j = i + 1 + rem
                return i, j

        total_target = min(total_all, max_pairs) if max_pairs is not None else total_all

        # --- 断点文件 ---
        if checkpoint_path is None:
            checkpoint_path = output_jsonl_path + ".ckpt"

        start_index = 0
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as cf:
                    start_index = int(cf.read().strip() or "0")
                print(f"发现断点：从第 {start_index} 对继续。")
            except Exception:
                print("断点文件损坏，忽略并从头开始。")
                start_index = 0

        # 输出文件：有断点则追加，否则覆盖
        file_mode = "a" if start_index > 0 else "w"

        print(f"可导航点数: {n}；计划测试对数: {total_target}（{'有序' if ordered else '组合'}）; 线程数: {num_workers}")

        # 快速索引
        id2coords = {pid: (pos["X"], pos["Y"], pos["Z"]) for (pid, pos, *_rest) in nav}

        # --- 并发控制结构 ---
        lock = threading.Lock()            # 保护写文件、计数、tqdm、断点
        next_index = start_index           # 全局"下一个要处理的线性索引"
        ok_cnt = 0
        fail_cnt = 0
        wrote_since_flush = 0

        def save_ckpt(k: int):
            try:
                with open(checkpoint_path, "w", encoding="utf-8") as cf:
                    cf.write(str(k))
            except Exception:
                pass

        # 工作者：从全局 next_index 领取任务，处理，写入一行
        def worker(fh, pbar):
            nonlocal next_index, ok_cnt, fail_cnt, wrote_since_flush
            while True:
                with lock:
                    k = next_index
                    if k >= total_target:
                        return
                    next_index += 1
                # —— 取到 (i, j) 并计算 ——
                try:
                    i, j = pair_at(k)
                    sid, spos, *_ = nav[i]
                    gid, gpos, *_ = nav[j]
                    start = id2coords[sid]
                    goal = id2coords[gid]

                    path = self.plan_full_path(start, goal)
                    if path:
                        # 生成增强路径信息
                        enhanced_path = self._generate_enhanced_path_info(path, start, goal)
                        total_length = enhanced_path[-1]["prefix_sum_path_length"] if enhanced_path else 0.0
                        with lock:
                            ok_cnt += 1
                    else:
                        enhanced_path = []
                        total_length = -1.0
                        with lock:
                            fail_cnt += 1

                    # 使用新的数据格式
                    rec = {
                        "start_point_id": sid,
                        "start_point_pos": spos,
                        "end_point_id": gid,
                        "end_point_pos": gpos,
                        "path": enhanced_path,  # 增强格式的路径信息
                        "all_length": total_length  # 改名为 all_length
                    }
                except Exception as e:
                    rec = {
                        "start_point_id": int(sid) if 'sid' in locals() else None,
                        "start_point_pos": spos if 'spos' in locals() else None,
                        "end_point_id": int(gid) if 'gid' in locals() else None,
                        "end_point_pos": gpos if 'gpos' in locals() else None,
                        "path": [],
                        "all_length": -1.0,
                        "error": repr(e)
                    }
                    with lock:
                        fail_cnt += 1

                # —— 边算边写（线程安全） ——
                with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    wrote_since_flush += 1
                    if wrote_since_flush >= flush_every:
                        fh.flush()
                        try:
                            os.fsync(fh.fileno())
                        except Exception:
                            pass
                        save_ckpt(next_index)  # 记录"下一个索引"
                        wrote_since_flush = 0

                    pbar.set_postfix({"ok": ok_cnt, "fail": fail_cnt})
                    pbar.update(1)

        # --- 主执行流程 ---
        try:
            with open(output_jsonl_path, file_mode, encoding="utf-8", buffering=1) as f, \
                 tqdm(total=total_target - start_index, desc="路径测试", unit="pair") as pbar:

                # 单线程直接跑，避免线程池开销
                if num_workers <= 1:
                    worker(f, pbar)
                else:
                    # 线程池
                    num_workers = max(1, int(num_workers))
                    with ThreadPoolExecutor(max_workers=num_workers) as ex:
                        futures = [ex.submit(worker, f, pbar) for _ in range(num_workers)]
                        for fu in futures:
                            # 等每个线程结束；异常会在此抛出
                            fu.result()

                # 收尾
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
                save_ckpt(next_index)

        except KeyboardInterrupt:
            try:
                save_ckpt(next_index)
            finally:
                print("\n手动中断，已保存断点。稍后可继续运行。")
                return

        print(f"完成：成功 {ok_cnt} 对，失败 {fail_cnt} 对。结果已写入：{output_jsonl_path}")
        print(f"断点文件：{checkpoint_path}（可删除以重头开始）")

    def quick_sampling_test(self, sample_per_area: int = 2, include_exits: bool = False):
        """
        在全量测试前进行快速采样测试。
        从每个区域采样若干点，测试它们之间的路径规划是否正常工作。

        Args:
            sample_per_area: 每个区域采样的点数
            include_exits: 是否包含出入口点（Blue点）

        Returns:
            bool: 是否通过测试（True=可以进行全量测试）
        """
        import random
        from collections import defaultdict

        print(f"\n=== 快速采样测试 ===")
        print(f"每个区域采样 {sample_per_area} 个点进行测试...")

        # 获取所有可导航点并按区域分组
        all_nav_points = self._iter_navigable_points(include_exits=include_exits)
        if len(all_nav_points) < 4:
            print("可导航点总数不足4个，跳过采样测试")
            return True

        # 按区域分组
        area_points = defaultdict(list)
        for point_info in all_nav_points:
            pid, pos_dict, area_name, color_class = point_info
            area_points[area_name].append(point_info)

        print(f"发现 {len(area_points)} 个区域:")
        for area_name, points in area_points.items():
            print(f"  - {area_name}: {len(points)} 个可导航点")

        # 从每个区域采样点
        sampled_points = []
        for area_name, points in area_points.items():
            if len(points) <= sample_per_area:
                # 如果区域点数不足，全部采样
                sampled_points.extend(points)
                print(f"  - {area_name}: 采样全部 {len(points)} 个点")
            else:
                # 随机采样
                samples = random.sample(points, sample_per_area)
                sampled_points.extend(samples)
                print(f"  - {area_name}: 随机采样 {sample_per_area} 个点")

        print(f"总计采样 {len(sampled_points)} 个测试点")

        if len(sampled_points) < 2:
            print("采样点数不足2个，跳过测试")
            return True

        # 建立id到坐标的映射
        id2coords = {}
        for pid, pos_dict, area_name, color_class in sampled_points:
            id2coords[pid] = (pos_dict["X"], pos_dict["Y"], pos_dict["Z"])

        # 测试所有采样点之间的路径规划
        test_count = 0
        success_count = 0
        failed_pairs = []

        print("\n开始路径规划测试...")
        for i in range(len(sampled_points)):
            for j in range(i + 1, len(sampled_points)):
                start_info = sampled_points[i]
                goal_info = sampled_points[j]

                start_id, start_pos, start_area, _ = start_info
                goal_id, goal_pos, goal_area, _ = goal_info

                start_coords = id2coords[start_id]
                goal_coords = id2coords[goal_id]

                test_count += 1
                print(f"  测试 {test_count}: {start_area}(id:{start_id}) -> {goal_area}(id:{goal_id})", end=" ")

                try:
                    path = self.plan_full_path(start_coords, goal_coords)
                    if path and len(path) >= 2:
                        success_count += 1
                        print("✓")
                    else:
                        print("✗ (无路径)")
                        failed_pairs.append((start_info, goal_info, "无路径"))
                except Exception as e:
                    print(f"✗ (异常: {repr(e)})")
                    failed_pairs.append((start_info, goal_info, f"异常: {repr(e)}"))

        # 结果统计
        success_rate = success_count / test_count if test_count > 0 else 0
        print(f"\n=== 采样测试结果 ===")
        print(f"总测试对数: {test_count}")
        print(f"成功: {success_count} 对")
        print(f"失败: {test_count - success_count} 对")
        print(f"成功率: {success_rate:.1%}")

        # 显示失败详情
        if failed_pairs:
            print(f"\n失败的路径规划:")
            for i, (start_info, goal_info, reason) in enumerate(failed_pairs[:5]):  # 最多显示前5个
                start_id, _, start_area, _ = start_info
                goal_id, _, goal_area, _ = goal_info
                print(f"  {i + 1}. {start_area}(id:{start_id}) -> {goal_area}(id:{goal_id}) - {reason}")
            if len(failed_pairs) > 5:
                print(f"  ... 还有 {len(failed_pairs) - 5} 个失败案例")

        # 判断是否通过测试
        threshold = 1.00  # 100%成功率阈值
        passed = success_rate >= threshold

        if passed:
            print(f"\n✓ 采样测试通过! (成功率 {success_rate:.1%} >= {threshold:.1%})")
            print("可以进行全量测试。")
        else:
            print(f"\n✗ 采样测试未通过! (成功率 {success_rate:.1%} < {threshold:.1%})")
            print("建议检查路径规划算法或数据质量后再进行全量测试。")

        return passed


# ===== 在文件底部 main() 后面追加一个测试入口（或直接改 main 触发）=====
def exhaustive_test_entry():
    print("=== 初始化（基于 UE_map_encode） ===")
    planner = PathPlanner(BASE_PATH)

    # 输出文件路径（可改名/改路径）
    out_path = r"L:\python_projects\AirSim-main\PythonClient\multirotor\nav_pairs_paths_0910.jsonl"

    # 运行全量测试：
    # - include_exits=False 只用“导航点”（不含 Blue 出入口）
    # - ordered=False 组合对（i<j），通常就够；如需方向性测试改为 True
    # - max_pairs=None 跑全部；调试时可设成小数比如 500
    planner.run_exhaustive_tests_concurrent(output_jsonl_path=out_path, include_exits=False, ordered=False,
                                                       max_pairs=None, flush_every=500)
# 同时需要添加一个全局函数来调用这个方法
def quick_sampling_test():
    """
    全局快速采样测试函数，供main()调用
    """
    planner = PathPlanner(BASE_PATH)
    return planner.quick_sampling_test(sample_per_area=3, include_exits=False)


# ------------- 运行入口 -------------
def main():
    # # 一键全量测试
    # exhaustive_test_entry()
    print()

if __name__ == "__main__":
    main()
