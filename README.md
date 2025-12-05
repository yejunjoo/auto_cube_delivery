# 📦 KAIST EE405: Autonomous Cube Logistics

## Team
**Yejun Joo, Jinyeol Kim, Chan-uk Park**

## 📌 Overview
An autonomous mobile manipulation project for **"Daily Cube Delivery"**. The robot interprets LLM instructions to deliver colored cubes to specific zones using VLM, SLAM and robotic arm control.

## 🎯 Mission
1.  **Instruction:** Receive a natural language task (e.g., *"Put blue cube in zone 3"*).
2.  **Execution:** Navigate to zones (1–3), pick up specific cubes, and place them correctly.
3.  **Completion:** Return to the starting point without collisions.

## 🛠 Key Tech
* **Navigation:** Localization (SLAM) & Path Planning in corridors.
* **Manipulation:** Pick & Place control for cubes.
* **AI:** Object detection (ArUco/Color) & LLM Planning.

## 📅 Evaluation
* **Date:** 2025.12.19 (Fri).
* **Scoring:** Task Success (45pt), Manipulation (15pt), Navigation (15pt), AI (15pt), Safety (-1pt/collision).


## How to Build & Run
* Copy this repo under ros2_ws/src/

```bash
cd ~/{ros2_workspace}/src
git clone https://github.com/yejunjoo/auto_cube_delivery.git
```

* ROS2 build
```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select auto_cube_delivery
source install/setup.bash
```


* (Terminal 1)
```bash
sudo systemctl stop start_app_node.service
ros2 launch auto_cube_delivery navigation.launch.py map:=map_01
(change to your map name)
```

* (Terminal 2)
```bash
ros2 launch auto_cube_delivery rviz_navigation.launch.py
```

* (Terminal 3)
```bash
source ~/ros2_ws/install/setup.bash
ros2 run auto_cube_delivery auto_cube_delivery
(ros2 run [package name] [command name])
```

* (Error) ModuleNotFoundError: No module named 'nav2_simple_commander'
```bash
sudo apt install ros-humble-nav2-simple-commander
```