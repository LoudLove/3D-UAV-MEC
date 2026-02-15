# 3D-UAV-MEC

Benchmark code for "Benchmarking MARL for UAV-Assisted Mobile Edge Computing Under Realistic 3D Collision-Avoidance Navigation Constraints for Periodic Task Offloading". Compares MARL paradigms for joint path planning and periodic task offloading, reporting success, latency, and energy in San Francisco.

![San Francisco Scenario](SanFrancisco_scenario.png)

---

## Overview
This repository provides a reproducible benchmark for **multi-UAV-assisted Mobile Edge Computing (MEC)** serving **cluster-based IoT networks** with **deadline-constrained periodic task offloading** in a realistic urban environment.

Unlike prior work that assumes unconstrained continuous 3D motion, this benchmark enforces **practical feasibility** via a **collision-free 3D navigation graph** (constructed from no-collision regions) and **shortest-path execution** for navigation decisions. On top of this mobility benchmark, we evaluate MARL-based joint navigation and offloading policies in a high-fidelity **San Francisco** digital twin.

This repo includes:
- Benchmark protocol for periodic task offloading under strict deadlines
- Navigation graph artifacts and precomputed shortest paths
- A UE-based 3D environment (distributed separately via a ZIP cloud link)

---

## Key Features
- **Realistic 3D collision avoidance** using a navigation graph (no building collisions)
- **Shortest-path execution** for graph-based navigation decisions
- **Periodic real-time task offloading** with explicit deadlines / hyperperiod settings
- Unified evaluation outputs for:
  - offloading success rate
  - end-to-end latency
  - energy consumption
- Reproducible scenario artifacts for **San Francisco** digital twin benchmarking

---

## UE Environment (Distributed via ZIP Cloud Link)
The Unreal Engine (UE) environment is provided **separately** via a ZIP cloud link (网盘链接) due to asset size/licensing constraints.

**Tested platform**
- Windows 10
- NVIDIA RTX 3090 GPU
- Visual Studio 2022
- Unreal Engine project / packaged environment included in the ZIP

**What the UE ZIP provides**
- A high-fidelity 3D digital-twin scene of San Francisco
- Benchmark-ready environment assets required for visualization and realistic navigation constraints
- (Optional) packaged build and/or UE project files (depending on your distribution)

**How to use**
1. Download and unzip the UE package from the provided cloud link.
2. Open the UE project (or run the packaged executable if provided).
3. Use the provided benchmark scenario files under `san_fran/` to reproduce navigation graphs, shortest paths, and periodic task initialization.

> Note: If your UE bundle includes third-party assets, please ensure redistribution complies with their license terms. Code and non-asset benchmark data can remain under the repository license.

---

## San Francisco Benchmark Scenario (`san_fran/`)
This repository contains benchmark artifacts for the San Francisco digital-twin scenario.

### Folder layout
