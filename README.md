# 复制以下内容到你的 README.md 文件中

# 3D-UAV-MEC: Benchmarking MARL for UAV-Assisted MEC

[![Scenario](SanFrancisco_scenario.png)](SanFrancisco_scenario.png)

**3D-UAV-MEC** is a high-fidelity benchmark for **Multi-Agent Reinforcement Learning (MARL)** in UAV-assisted Mobile Edge Computing. It simulates periodic task offloading in a realistic 3D urban environment (San Francisco) under strict collision-avoidance and deadline constraints.

---

## ### Key Features
* **Realistic 3D Navigation**: Enforces collision-free movement using a navigation graph derived from a San Francisco digital twin.
* **Practical Constraints**: Implements shortest-path execution and periodic task offloading with explicit deadlines/hyperperiods.
* **Comprehensive Metrics**: Standardized evaluation of **success rate**, **end-to-end latency**, and **energy consumption**.
* **Extensible Framework**: Provides reference environments for both discrete and continuous action spaces.

---

## ### System Requirements & Resources

**Software Environment:**
* **OS**: Windows 10
* **GPU**: NVIDIA RTX 3090 (recommended)
* **Tools**: Visual Studio 2022, Unreal Engine (UE)

**External Assets:**
Due to size constraints, the **Unreal Engine Digital Twin** is hosted separately:
* **Download Link**: [Baidu Netdisk](https://pan.baidu.com/s/1QvN_AGWM0D9hF3-Kshla9A?pwd=6666) (Password: `6666`)
* **Contents**: High-fidelity SF scene, navigation primitives, and packaged UE environment.

---

## ### Project Structure (`san_fran/`)
The `san_fran/scenario1/` directory contains precomputed artifacts for reproducibility:
* **Navigation**: `bidirection_dictionary.jsonl` (graph adjacency), `nav_pairs_paths.jsonl` (shortest paths).
* **Map Data**: `UE_map_encode/` (zone-specific navigation points and obstacles).
* **Task Config**: `task_initialization_data.json` (periodic task streams and deadlines).

---

## ### Getting Started: Execution Order

Follow these steps to prepare the environment and run the benchmark:

1.  **Map Encoding**: Convert UE scene data into structured maps.
    `python map_encoder.py`
2.  **Point Precomputation**: Generate nearby-point lookup tables.
    `python precompute_nearby_points.py`
3.  **Graph Construction**: Build the bidirectional collision-free navigation graph.
    `python build_bidirectional_path_permutation_dict_jsonl.py`
4.  **Verification (Optional)**: Preview the map and verify route planning logic.
    `python preview_full_map_and_route_planning.py`
5.  **Task Initialization**: Generate periodic MEC task streams.
    `python ch_imd_task_initializer.py`
6.  **MARL Integration**: Launch the environment.
    * For absolute/discrete actions: `python san_scen1_abso.py`
    * For continuous actions: `python san_scen1_continu.py`

---

## ### Framework Integration
The provided environment files are designed as **reference implementations**. You may need to wrap them depending on your MARL library (e.g., **BenchMARL**, **MAPPO**, or **PettingZoo**).

**Customization Points:**
* **Observation Space**: Tailor the state vector for centralized/decentralized training.
* **Action Space**: Adjust for discrete indexing or continuous control.
* **Reward Function**: Modify reward aggregation (weighted sum, independent, etc.) to suit specific algorithms.

---

## ### License
Code and benchmark artifacts are provided under the repository license. Please ensure compliance with third-party licenses for any UE assets used.
