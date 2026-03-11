#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于现有PathPlanner类计算每个点到(绿/黄/红)点的最近点（统一nearest_points版本）
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial import KDTree
from tqdm import tqdm

# 导入现有的PathPlanner（确保preprocess_near_points_rely.py在同目录或Python路径中）
from preprocess_near_points_rely import PathPlanner, BASE_PATH


def calculate_distance(p1, p2):
    """计算两点间的欧几里得距离（当前未使用，保留以备扩展）"""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def extract_colored_points(
    planner: PathPlanner,
    target_colors: List[str],
) -> Dict[str, List[Tuple[int, List[float]]]]:
    """
    从PathPlanner中提取指定颜色的点

    Args:
        planner: PathPlanner实例
        target_colors: 目标颜色列表，如['green', 'yellow', 'red']

    Returns:
        字典，键为颜色，值为[(点ID, [x,y,z坐标]), ...]的列表
    """
    colored_points: Dict[str, List[Tuple[int, List[float]]]] = {
        color: [] for color in target_colors
    }

    # 遍历所有区域的点
    for area_name, area_data in planner.areas.items():
        for point in area_data["points"]:
            color_class = point.get("color_class", "").lower()
            if color_class in target_colors:
                point_id = point["id"]
                coords = point["coords"]
                colored_points[color_class].append((point_id, coords))

    # 打印统计信息
    print("提取的各颜色点数量:")
    for color, points in colored_points.items():
        print(f"  {color}: {len(points)} 个点")

    return colored_points


def calculate_all_nearest_points_merged_optimized(
    colored_points: Dict[str, List[Tuple[int, List[float]]]]
) -> Dict[int, Dict[str, List[Dict[str, float]]]]:
    """
    使用KDTree优化的、将green/yellow/red三个颜色合并后的完整距离计算。

    对所有 green/yellow/red 点构建一个全局 KDTree，
    对每个点，计算其到所有(三色)点的距离，并按距离升序保存为 "nearest_points"。

    Returns:
        结果字典:
        {
            point_id: {
                "nearest_points": [
                    {"id": other_id, "distance": dist}, ...
                ]
            },
            ...
        }
    """
    result: Dict[int, Dict[str, List[Dict[str, float]]]] = {}

    # 1) 收集三种颜色的所有点（即只考虑 green/yellow/red）
    all_points: List[Tuple[int, List[float]]] = []
    for color, points in colored_points.items():
        all_points.extend(points)

    total_points = len(all_points)
    if total_points == 0:
        print("错误: 没有任何 green/yellow/red 颜色的点！")
        return result

    print(f"正在构建全局KDTree索引，总点数: {total_points} ...")

    # 2) 构建一个“全局”的KDTree
    all_ids = [pid for pid, _ in all_points]
    all_coords = np.array([coords for _, coords in all_points])
    global_tree = KDTree(all_coords)

    print(f"开始优化计算 {total_points} 个点到所有(三色)点的距离...")

    # 3) 对每个点，查询它到所有 (三色) 点的距离列表
    for (query_id, query_coords) in tqdm(all_points, desc="优化计算统一 nearest_points"):
        query_point = np.array(query_coords)

        # 查询到所有点的距离
        dists, idxs = global_tree.query(query_point, k=total_points)

        # 保证是数组
        if np.isscalar(dists):
            dists = [dists]
            idxs = [idxs]

        nearest_list: List[Dict[str, float]] = []
        for dist, idx in zip(dists, idxs):
            candidate_id = all_ids[idx]
            # 排除自己 和 距离为0的重复点
            if candidate_id == query_id or dist <= 1e-6:
                continue
            nearest_list.append({
                "id": candidate_id,
                "distance": float(dist),
            })

        # KDTree返回结果本身就是距离递增的，这里再 sort 一次以防万一
        nearest_list.sort(key=lambda x: x["distance"])

        # ✅ 统一用 "nearest_points" 作为 key，不再区分颜色
        result[query_id] = {"nearest_points": nearest_list}

    return result


def main_full_distances():
    """计算统一最近点( nearest_points )信息的主函数"""
    print("=== 计算所有(绿/黄/红)点之间的最近点信息 ===")

    # 初始化PathPlanner
    print("正在初始化PathPlanner...")
    planner = PathPlanner(BASE_PATH)

    # 配置参数：只考虑这三类
    target_colors = ['green', 'yellow', 'red']

    # 输出文件路径
    # 这里沿用你之前的 revised_coordinate 命名，也可以自行修改
    output_file = os.path.join(
        BASE_PATH,   r"..\all_nearest_points.json"
    )

    # 提取各颜色的点
    print("正在提取各颜色点...")
    colored_points = extract_colored_points(planner, target_colors)

    # 检查是否有点
    total_points = sum(len(points) for points in colored_points.values())
    if total_points == 0:
        print("错误: 没有找到任何指定颜色的点!")
        return

    # 使用优化算法计算统一最近点（不再按颜色拆分）
    print("正在使用KDTree优化计算每个点到所有(三色)点的距离...")
    result = calculate_all_nearest_points_merged_optimized(colored_points)

    # 保存结果
    print(f"正在保存结果到 {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"完成! 结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return

    print(f"处理了 {len(result)} 个点的完整距离计算")

    # 显示一个示例结果
    if result:
        sample_id = next(iter(result.keys()))
        print(f"\n示例结果 (点ID {sample_id}):")
        nearest = result[sample_id].get("nearest_points", [])
        if nearest:
            print(f"  nearest_points: 共 {len(nearest)} 个点")
            print(f"    最近的5个: {nearest[:5]}")
        else:
            print("  nearest_points: 无数据")

    # 显示统计信息
    print("\n=== 统计信息 ===")
    non_empty_count = sum(
        1 for point_data in result.values()
        if point_data.get("nearest_points")
    )
    print(f"共有 {non_empty_count} 个查询点有对应的最近点数据")


if __name__ == "__main__":
    # 运行统一 nearest_points 版本
    main_full_distances()
