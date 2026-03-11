import json
import math
from typing import Dict, Tuple, List, Any
from copy import deepcopy

def calculate_distance(pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
    """计算两点间的欧几里得距离"""
    dx = pos1['X'] - pos2['X']
    dy = pos1['Y'] - pos2['Y']
    dz = pos1['Z'] - pos2['Z']
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def reverse_path(path_data: Dict[str, Any]) -> Dict[str, Any]:
    """生成反向路径，重新计算prefix_sum_path_length"""
    original_path = path_data['path']
    reversed_path = [deepcopy(point) for point in reversed(original_path)]

    reversed_path[0]['prefix_sum_path_length'] = 0.0
    for i in range(1, len(reversed_path)):
        prev_pos = reversed_path[i - 1]['point_pos']
        curr_pos = reversed_path[i]['point_pos']
        segment_distance = calculate_distance(prev_pos, curr_pos)
        reversed_path[i]['prefix_sum_path_length'] = (
                reversed_path[i - 1]['prefix_sum_path_length'] + segment_distance
        )

    total_length = reversed_path[-1]['prefix_sum_path_length']
    reversed_data = {
        'start_point_id': path_data['end_point_id'],
        'start_point_pos': path_data['end_point_pos'],
        'end_point_id': path_data['start_point_id'],
        'end_point_pos': path_data['start_point_pos'],
        'path': reversed_path,
        'all_length': total_length
    }
    return reversed_data


def build_path_dictionary(jsonl_file_path: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """构建路径查找字典"""
    path_dict = {}

    print("正在读取JSONL文件...")
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        line_count = 0
        for line in file:
            line_count += 1
            if line_count % 10000 == 0:
                print(f"已处理 {line_count} 行")
            try:
                data = json.loads(line.strip())
                start_id = data['start_point_id']
                end_id = data['end_point_id']
                path_dict[(start_id, end_id)] = data
            except json.JSONDecodeError as e:
                print(f"第 {line_count} 行JSON解析错误: {e}")
                continue

    print(f"完成原始路径读取，共 {len(path_dict)} 条路径")

    # 检查并生成缺失的反向路径
    print("正在检查并生成反向路径...")
    original_keys = list(path_dict.keys())
    added_reverse_count = 0

    for start_id, end_id in original_keys:
        reverse_key = (end_id, start_id)
        if reverse_key not in path_dict:
            try:
                original_data = path_dict[(start_id, end_id)]
                reversed_data = reverse_path(original_data)
                path_dict[reverse_key] = reversed_data
                added_reverse_count += 1
                if added_reverse_count % 5000 == 0:
                    print(f"已生成 {added_reverse_count} 条反向路径")
            except Exception as e:
                print(f"生成反向路径失败 ({start_id} -> {end_id}): {e}")

    print(f"完成反向路径生成，新增 {added_reverse_count} 条路径")
    print(f"最终字典包含 {len(path_dict)} 条路径")
    return path_dict


def query_path(path_dict: Dict[Tuple[int, int], Dict[str, Any]],
               global_id_start: int, global_id_end: int) -> Dict[str, Any]:
    """查询路径"""
    key = (global_id_start, global_id_end)
    return path_dict.get(key)


def save_path_dict(path_dict: Dict[Tuple[int, int], Dict[str, Any]],
                   output_file: str):
    """保存路径字典到JSONL文件（每行一个JSON对象）"""
    print(f"正在保存路径字典到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for (start_id, end_id), path_data in path_dict.items():
            # 在路径数据中加入key信息，便于加载时恢复
            save_data = {
                'start_id': start_id,
                'end_id': end_id,
                'path_data': path_data
            }
            f.write(json.dumps(save_data, ensure_ascii=False) + '\n')
    print(f"路径字典已保存到 {output_file}")


def load_path_dict(input_file: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """从JSONL文件加载路径字典"""
    print(f"正在从 {input_file} 加载路径字典...")
    path_dict = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            line_count += 1
            if line_count % 10000 == 0:
                print(f"已加载 {line_count} 条路径")
            try:
                data = json.loads(line.strip())
                start_id = data['start_id']
                end_id = data['end_id']
                path_dict[(start_id, end_id)] = data['path_data']
            except json.JSONDecodeError as e:
                print(f"加载第 {line_count} 行失败: {e}")
                continue
    print(f"路径字典加载完成，包含 {len(path_dict)} 条路径")
    return path_dict


# 主要使用示例
if __name__ == "__main__":
    # 文件路径
    # jsonl_file_path = r"L:\python_projects\AirSim-main\PythonClient\multirotor\reconst\10\nav_pairs_paths.jsonl"
    # dict_save_path = r"L:\python_projects\AirSim-main\PythonClient\multirotor\reconst\10\bidirection_dictionary.jsonl"  # 改为jsonl后缀
    # jsonl_file_path = r"L:\python_projects\AirSim-main\PythonClient\multirotor\reconst\20\nav_pairs_paths.jsonl"
    # dict_save_path = r"L:\python_projects\AirSim-main\PythonClient\multirotor\reconst\20\bidirection_dictionary.jsonl"  # 改为jsonl后缀
    # jsonl_file_path = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario1\nav_pairs_paths.jsonl"
    # dict_save_path = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario1\bidirection_dictionary.jsonl"  # 改为jsonl后缀
    # jsonl_file_path = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario2\nav_pairs_paths.jsonl"
    # dict_save_path = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario2\bidirection_dictionary.jsonl"  # 改为jsonl后缀
    jsonl_file_path = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario3\nav_pairs_paths.jsonl"
    dict_save_path = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario3\bidirection_dictionary.jsonl"  # 改为jsonl后缀

    # 1. 构建路径字典（首次运行时执行）
    print("开始构建路径字典...")
    path_dict = build_path_dictionary(jsonl_file_path)
    save_path_dict(path_dict, dict_save_path)

    # 2. 从JSONL文件加载字典（后续使用时直接加载）
    path_dict = load_path_dict(dict_save_path)

    print("\n=== 路径查询示例 ===")
    # 查询示例路径（请根据实际ID修改）
    path_info_forward = query_path(path_dict, global_id_start=20001, global_id_end=80002)
    if path_info_forward:
        print(f"路径 20001→80002:")
        print(f"  路径长度: {path_info_forward['all_length']}")
        print(f"  路径点数: {len(path_info_forward['path'])}")

    path_info_reverse = query_path(path_dict, global_id_start=80002, global_id_end=20001)
    if path_info_reverse:
        print(f"\n路径 80002→20001:")
        print(f"  路径长度: {path_info_reverse['all_length']}")
        print(f"  路径点数: {len(path_info_reverse['path'])}")