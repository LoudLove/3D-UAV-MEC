# preprocess_ue_markers.py
# -*- coding: utf-8 -*-
import os, re, json, csv
from pathlib import Path
from typing import Dict, List, Tuple

# ====== 配置区 ======
# INPUT_DIR = r"L:\python_projects\AirSim-main\PythonClient\multirotor\reconst\10\UE_points_and_cubes"
# INPUT_DIR = r"L:\python_projects\AirSim-main\PythonClient\multirotor\reconst\20\UE_points_and_cubes"
# INPUT_DIR = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\UE_points_and_cubes"
# INPUT_DIR = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario2\UE_points_and_cubes"
INPUT_DIR = r"L:\python_projects\AirSim-main\PythonClient\multirotor\san_fran\scenario3\UE_points_and_cubes"
# INPUT_DIR = r"L:\python_projects\AirSim-main\PythonClient\multirotor\UE_output\temp"
OUTPUT_DIR_NAME = "../UE_map_encode"  # 会创建在 INPUT_DIR 之下
# ===================

# 来自你给的枚举；用于分配 area_id（从 1 开始）
# CUBES_CLASSNAME_ENUM = [
#     "outside",
#     "zone1",
#     "zone2",
# ]
CUBES_CLASSNAME_ENUM = [
    "outside",
    "zone1",
    "zone2",
    "zone3",
    "zone4",
    "zone5",
    "zone6",
    "zone7",
    "zone8",
    "zone9",
    "zone10"
]

# 文件名到统一场景名的“归一化”规则
def normalize_area_name_from_filename(stem: str) -> str:
    """
    根据文件名（不含扩展名）推断并归一化场景名，使之能对上 CUBES_CLASSNAME_ENUM。
    例如：
      - outdoor_loading_* -> outdoor_load_area
      - production_area_b1_* -> production_area_1
    """
    s = stem.lower()

    # 去掉 _points / _cubes 后缀，得到 area 基名
    s = re.sub(r'_(points|cubes)$', '', s)

    # 常见别名归一化
    s = s.replace("outdoor_loading", "outdoor_load_area")

    # production_area_bN -> production_area_N
    m = re.match(r'(production_area_)[b]?(\d+)$', s)
    if m:
        s = f"{m.group(1)}{m.group(2)}"

    return s

# 将向量字符串解析成 dict 浮点值
def parse_kv_triplet(s: str, keys: Tuple[str, str, str]) -> Dict[str, float]:
    """
    解析形如 "X=-3050.000 Y=12050.000 Z=1650.000" 或 "P=0 Y=0 R=-0" 的字符串
    返回 {key: float}
    """
    out = {}
    # 支持正负号与浮点
    for k in keys:
        # 匹配形如 "X=123.456" 或 "x=-0.0"
        m = re.search(rf'{k}\s*=\s*([+-]?\d+(?:\.\d+)?)', s, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"无法在 '{s}' 中解析键 {k}")
        out[k.upper()] = float(m.group(1))
    return out

def parse_jsonl_loose(text: str) -> List[Dict]:
    """
    容错解析 .jsonl：
    - 有些文件可能是一行一个对象；也可能是 `}{` 紧挨着。
    - 用正则抓取每个 JSON 对象。
    """
    objs = []
    for m in re.finditer(r'\{.*?\}', text, flags=re.DOTALL):
        chunk = m.group(0)
        try:
            objs.append(json.loads(chunk))
        except json.JSONDecodeError as e:
            # 尝试去掉可能的尾部逗号/空白后再试
            chunk2 = re.sub(r',\s*}', '}', chunk)
            objs.append(json.loads(chunk2))
    return objs

def read_jsonl_file(path: Path) -> List[Dict]:
    text = path.read_text(encoding='utf-8-sig', errors='ignore')
    return parse_jsonl_loose(text)

def write_jsonl_file(path: Path, rows: List[Dict]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")

def collect_scene_files(root: Path) -> Dict[str, Dict[str, Path]]:
    """
    返回 {scene_key: {"points": Path或None, "cubes": Path或None}}
    scene_key 为“归一化后”的场景名
    """
    result: Dict[str, Dict[str, Path]] = {}
    for p in root.glob("*.jsonl"):
        stem = p.stem  # e.g. command_room_points
        kind = "points" if stem.endswith("_points") else ("cubes" if stem.endswith("_cubes") else None)
        if not kind:
            continue
        base = normalize_area_name_from_filename(stem)
        result.setdefault(base, {"points": None, "cubes": None})
        result[base][kind] = p
    return result

def build_area_id_mapping(existing_enum: List[str], discovered: List[str]) -> Dict[str, int]:
    """
    基于枚举先分配 ID（从 1 开始），再给未包含的新场景追加 ID。
    返回 {area_name: area_id}
    """
    mapping: Dict[str, int] = {}
    # 先保留枚举顺序
    next_id = 1
    for name in existing_enum:
        mapping[name] = next_id
        next_id += 1
    # 对新发现但不在枚举里的场景名，继续追加
    for name in sorted(discovered):
        if name not in mapping:
            mapping[name] = next_id
            next_id += 1
    return mapping

def main():
    root = Path(INPUT_DIR)
    out_dir = root / OUTPUT_DIR_NAME
    out_dir.mkdir(exist_ok=True, parents=True)

    # 收集所有场景的 points / cubes 文件
    scenes = collect_scene_files(root)
    discovered_names = list(scenes.keys())

    # area_name -> area_id
    areaid_map = build_area_id_mapping(CUBES_CLASSNAME_ENUM, discovered_names)

    # 为了写出映射文件，也构造反查
    # 并保证 ID 稳定从 1 开始连续
    areaid_to_name = {areaid_map[name]: name for name in areaid_map}
    areaid_to_name = dict(sorted(areaid_to_name.items(), key=lambda x: x[0]))

    # 保存映射（json 和 csv）
    (out_dir / "areaid_mapping.json").write_text(
        json.dumps(areaid_to_name, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    with (out_dir / "areaid_mapping.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["area_id", "area_name"])
        for aid, aname in areaid_to_name.items():
            w.writerow([aid, aname])

    # 逐场景处理
    for area_name, pair in scenes.items():
        area_id = areaid_map[area_name]

        # 读取两类
        points_list = read_jsonl_file(pair["points"]) if pair["points"] else []
        cubes_list  = read_jsonl_file(pair["cubes"])  if pair["cubes"]  else []

        # 标记类型，合并（仅用于统一排序与本地编号）
        def prep_rows(rows: List[Dict], kind: str) -> List[Dict]:
            prepped = []
            for r in rows:
                # 解析向量
                loc = parse_kv_triplet(str(r.get("Location", "")), ("X", "Y", "Z"))
                rot = parse_kv_triplet(str(r.get("Rotation", "")), ("P", "Y", "R"))
                scl = parse_kv_triplet(str(r.get("Scale", "")), ("X", "Y", "Z"))
                # 写回为字典（键大写更统一）
                r["Location"] = {"X": loc["X"], "Y": loc["Y"], "Z": loc["Z"]}
                r["Rotation"] = {"P": rot["P"], "Y": rot["Y"], "R": rot["R"]}
                r["Scale"]    = {"X": scl["X"], "Y": scl["Y"], "Z": scl["Z"]}
                # 附加元数据
                r["_type"] = kind  # 内部用，便于分流写回
                prepped.append(r)
            return prepped

        combined = prep_rows(points_list, "points") + prep_rows(cubes_list, "cubes")

        # 统一排序（X 升序，其次 Y 升序）
        combined.sort(key=lambda r: (r["Location"]["X"], r["Location"]["Y"]))

        # 统一编号：本地序号从 1 开始
        for idx, r in enumerate(combined, start=1):
            global_id = area_id * 10000 + idx
            r["id"] = int(global_id)
            r["area_id"] = int(area_id)
            r["area_name"] = area_name

        # 分流写回到各自文件（只带必要字段）
        def strip_internal(r: Dict) -> Dict:
            r2 = dict(r)
            r2.pop("_type", None)  # 去掉内部标记
            return r2

        new_points = [strip_internal(r) for r in combined if r["_type"] == "points"]
        new_cubes  = [strip_internal(r) for r in combined if r["_type"] == "cubes"]

        # 输出文件路径（沿用原名，写到 UE_map_encode_old）
        if pair["points"]:
            out_points = out_dir / pair["points"].name
            write_jsonl_file(out_points, new_points)
        if pair["cubes"]:
            out_cubes = out_dir / pair["cubes"].name
            write_jsonl_file(out_cubes, new_cubes)

    print(f"完成。结果已写入：{out_dir}")
    print("映射文件：areaid_mapping.json、areaid_mapping.csv")

if __name__ == "__main__":
    main()
