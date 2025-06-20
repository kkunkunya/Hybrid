"""
实时可视化引擎
展示多UAV-USV协同巡检系统的运行过程
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import threading
import queue

from .enhanced_scene_parser import EnhancedSceneParser as SceneParser
from .charging_station import ChargingStationDetector, ChargingStation


class Agent:
    """智能体类"""
    
    def __init__(self, agent_id: str, agent_type: str, start_pos: Tuple[int, int]):
        """
        初始化智能体
        
        Args:
            agent_id: 智能体ID
            agent_type: 类型 ("UAV" 或 "USV")
            start_pos: 起始位置
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.position = start_pos
        self.start_position = start_pos
        self.trajectory = [start_pos]
        self.current_task_id = None
        self.assigned_tasks = []
        self.status = "idle"  # idle, moving, inspecting, returning, charging
        self.battery_level = 100.0
        self.speed = 15.0 if agent_type == "UAV" else 8.0  # m/s
        self.color = self._get_agent_color()
        self.max_trajectory_length = 30  # 限制轨迹长度，减少视觉混乱
        
    def _get_agent_color(self) -> str:
        """获取智能体颜色"""
        if self.agent_type == "UAV":
            colors_uav = ["red", "blue", "green", "orange", "purple"]
            agent_num = int(self.agent_id[-1]) - 1 if self.agent_id[-1].isdigit() else 0
            return colors_uav[agent_num % len(colors_uav)]
        else:  # USV
            return "navy"
    
    def update_position(self, new_pos: Tuple[int, int]):
        """更新位置"""
        self.position = new_pos
        self.trajectory.append(new_pos)
        
        # 限制轨迹长度，避免视觉混乱
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory = self.trajectory[-self.max_trajectory_length:]


class Task:
    """任务类"""
    
    def __init__(self, task_id: int, position: Tuple[int, int], task_type: str = "inspection"):
        """
        初始化任务
        
        Args:
            task_id: 任务ID
            position: 任务位置
            task_type: 任务类型
        """
        self.task_id = task_id
        self.position = position
        self.task_type = task_type
        self.status = "pending"  # pending, assigned, in_progress, completed
        self.assigned_agent = None
        self.priority = 1.0
        self.estimated_duration = 60.0  # 秒
        self.completion_time = None


class RealTimeVisualizer:
    """实时可视化器"""
    
    def __init__(self, scene_parser: SceneParser, window_size: Tuple[int, int] = (1200, 900)):
        """
        初始化可视化器
        
        Args:
            scene_parser: 场景解析器
            window_size: 窗口大小
        """
        self.scene_parser = scene_parser
        self.window_size = window_size
        
        # 可视化状态
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[int, Task] = {}
        self.charging_stations: List[ChargingStation] = []
        self.current_time = 0.0
        self.simulation_speed = 1.0  # 仿真速度倍数
        self.is_running = False
        self.is_paused = False
        
        # 图形界面
        self.fig = None
        self.ax = None
        self.background_image = None
        self.animation = None
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'total_distance': 0.0,
            'total_energy': 0.0,
            'start_time': 0.0
        }
        
        # 事件队列
        self.event_queue = queue.Queue()
        
    def setup_visualization(self) -> bool:
        """
        设置可视化环境
        
        Returns:
            是否设置成功
        """
        try:
            # 创建图形窗口
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            self.fig, self.ax = plt.subplots(figsize=(12, 9))
            self.fig.suptitle(f"多UAV-USV协同巡检系统 - {self.scene_parser.scene_name}", 
                            fontsize=16, fontweight='bold')
            
            # 加载背景图像
            if self.scene_parser.image is not None:
                # OpenCV BGR -> RGB
                self.background_image = cv2.cvtColor(self.scene_parser.image, cv2.COLOR_BGR2RGB)
                self.ax.imshow(self.background_image, extent=[0, self.scene_parser.image_size[0], 
                                                           self.scene_parser.image_size[1], 0])
                print(f"✅ 背景图像已加载: {self.background_image.shape}")
            else:
                print("⚠️ 无背景图像，使用纯色背景")
                self.ax.set_facecolor('lightblue')
            
            # 设置坐标轴
            self.ax.set_xlim(0, self.scene_parser.image_size[0])
            self.ax.set_ylim(self.scene_parser.image_size[1], 0)  # 图像坐标系
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('X坐标 (像素)')
            self.ax.set_ylabel('Y坐标 (像素)')
            
            # 部署充电桩
            self._deploy_charging_stations()
            
            # 绘制静态元素
            self._draw_static_elements()
            
            print("✅ 可视化环境设置完成")
            return True
            
        except Exception as e:
            print(f"❌ 可视化环境设置失败: {e}")
            return False
    
    def _deploy_charging_stations(self):
        """部署充电桩"""
        try:
            if self.scene_parser.image is not None:
                # 使用图像识别部署充电桩
                detector = ChargingStationDetector()
                wind_turbine_positions = self.scene_parser.get_task_positions()
                
                self.charging_stations = detector.deploy_charging_stations(
                    self.scene_parser.image, wind_turbine_positions, 3)
            else:
                # 使用默认位置
                default_positions = [(150, 450), (400, 50), (850, 850)]
                self.charging_stations = []
                for i, pos in enumerate(default_positions):
                    station = ChargingStation(i, pos, "shore")
                    self.charging_stations.append(station)
                
                print(f"✅ 使用默认位置部署 {len(self.charging_stations)} 个充电桩")
                
        except Exception as e:
            print(f"⚠️ 充电桩部署失败，使用默认位置: {e}")
            # 备用方案
            default_positions = [(150, 450), (400, 50), (850, 850)]
            self.charging_stations = []
            for i, pos in enumerate(default_positions):
                station = ChargingStation(i, pos, "shore")
                self.charging_stations.append(station)
    
    def _draw_static_elements(self):
        """绘制静态元素（风机、陆地等）"""
        # 绘制风机
        for turbine_data in self.scene_parser.create_visualization_data()['wind_turbines']:
            center = turbine_data['center']
            
            # 绘制风机圆圈
            circle = plt.Circle(center, 15, color='yellow', alpha=0.6, 
                              linewidth=2, edgecolor='orange')
            self.ax.add_patch(circle)
            
            # 添加风机标签
            self.ax.text(center[0], center[1]-25, f"T{turbine_data['id']:02d}", 
                        ha='center', va='center', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        
        # 绘制陆地区域
        for land_data in self.scene_parser.create_visualization_data()['land_areas']:
            bbox = land_data['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            rect = patches.Rectangle((bbox[0], bbox[1]), width, height,
                                   linewidth=2, edgecolor='brown', facecolor='tan', alpha=0.3)
            self.ax.add_patch(rect)
        
        # 绘制充电桩
        for station in self.charging_stations:
            center = station.position
            
            # 充电桩图标 - 使用绿色正方形
            square = patches.Rectangle((center[0]-8, center[1]-8), 16, 16,
                                     linewidth=3, edgecolor='darkgreen', 
                                     facecolor='lightgreen', alpha=0.8)
            self.ax.add_patch(square)
            
            # 添加充电桩标签
            self.ax.text(center[0], center[1]-25, f"⚡C{station.station_id}", 
                        ha='center', va='center', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
    
    def add_agent(self, agent_id: str, agent_type: str, start_pos: Tuple[int, int]):
        """
        添加智能体
        
        Args:
            agent_id: 智能体ID
            agent_type: 类型
            start_pos: 起始位置
        """
        agent = Agent(agent_id, agent_type, start_pos)
        self.agents[agent_id] = agent
        print(f"✅ 添加智能体: {agent_id} ({agent_type}) 于位置 {start_pos}")
    
    def add_task(self, task_id: int, position: Tuple[int, int], task_type: str = "inspection"):
        """
        添加任务
        
        Args:
            task_id: 任务ID
            position: 任务位置
            task_type: 任务类型
        """
        task = Task(task_id, position, task_type)
        self.tasks[task_id] = task
        self.stats['total_tasks'] += 1
    
    def assign_tasks(self, assignments: Dict[str, List[int]]):
        """
        分配任务
        
        Args:
            assignments: 任务分配 {agent_id: [task_id_list]}
        """
        for agent_id, task_ids in assignments.items():
            if agent_id in self.agents:
                self.agents[agent_id].assigned_tasks = task_ids
                for task_id in task_ids:
                    if task_id in self.tasks:
                        self.tasks[task_id].assigned_agent = agent_id
                        self.tasks[task_id].status = "assigned"
                print(f"✅ {agent_id} 分配任务: {task_ids}")
    
    def update_agent_position(self, agent_id: str, new_pos: Tuple[int, int], 
                            status: str = None, battery: float = None):
        """
        更新智能体位置和状态
        
        Args:
            agent_id: 智能体ID
            new_pos: 新位置
            status: 新状态
            battery: 电池电量
        """
        if agent_id in self.agents:
            self.agents[agent_id].update_position(new_pos)
            if status:
                self.agents[agent_id].status = status
            if battery is not None:
                self.agents[agent_id].battery_level = battery
    
    def complete_task(self, task_id: int, completion_time: float):
        """
        完成任务
        
        Args:
            task_id: 任务ID
            completion_time: 完成时间
        """
        if task_id in self.tasks:
            self.tasks[task_id].status = "completed"
            self.tasks[task_id].completion_time = completion_time
            self.stats['completed_tasks'] += 1
    
    def _draw_frame(self, frame_num: int):
        """
        绘制单帧
        
        Args:
            frame_num: 帧编号
        """
        # 清除动态元素
        for artist in self.ax.lines + self.ax.collections:
            if hasattr(artist, '_is_dynamic'):
                artist.remove()
        
        # 绘制智能体轨迹 - 优化显示效果
        for agent in self.agents.values():
            if len(agent.trajectory) > 1:
                traj_x = [pos[0] for pos in agent.trajectory]
                traj_y = [pos[1] for pos in agent.trajectory]
                
                # 渐变透明度效果
                n_points = len(traj_x)
                for i in range(n_points - 1):
                    alpha = 0.3 + 0.4 * (i / max(1, n_points - 1))  # 从0.3到0.7的渐变
                    line, = self.ax.plot([traj_x[i], traj_x[i+1]], [traj_y[i], traj_y[i+1]], 
                                       color=agent.color, alpha=alpha, linewidth=2, linestyle='-')
                    line._is_dynamic = True
        
        # 绘制智能体当前位置
        for agent in self.agents.values():
            x, y = agent.position
            
            # 智能体图标
            if agent.agent_type == "UAV":
                marker = "^"
                size = 200
            else:  # USV
                marker = "s"
                size = 150
            
            scatter = self.ax.scatter(x, y, c=agent.color, marker=marker, s=size, 
                                    edgecolors='black', linewidth=2, alpha=0.9)
            scatter._is_dynamic = True
            
            # 智能体状态信息 - 动态位置避免重叠
            status_color = {
                'idle': 'lightblue',
                'moving': 'orange', 
                'inspecting': 'yellow',
                'returning': 'lightcoral',
                'charging': 'lightgreen'
            }.get(agent.status, 'white')
            
            # 根据智能体类型调整文本位置
            if agent.agent_type == "UAV":
                text_offset = (25, -15)
            else:  # USV
                text_offset = (25, 15)
            
            status_text = f"{agent.agent_id}\\n{agent.status}\\n{agent.battery_level:.0f}%"
            text = self.ax.text(x + text_offset[0], y + text_offset[1], status_text, 
                              fontsize=8, ha='left', va='center',
                              bbox=dict(boxstyle="round,pad=0.2", facecolor=status_color, alpha=0.9))
            text._is_dynamic = True
        
        # 绘制任务状态
        for task in self.tasks.values():
            x, y = task.position
            
            # 任务状态颜色
            if task.status == "completed":
                color = "green"
                alpha = 0.3
            elif task.status == "in_progress":
                color = "orange"
                alpha = 0.8
            elif task.status == "assigned":
                color = "blue"
                alpha = 0.6
            else:  # pending
                color = "red"
                alpha = 0.4
            
            # 任务标记
            circle = plt.Circle((x, y), 10, color=color, alpha=alpha, linewidth=2)
            self.ax.add_patch(circle)
            circle._is_dynamic = True
        
        # 更新标题和统计信息
        completion_rate = (self.stats['completed_tasks'] / max(self.stats['total_tasks'], 1)) * 100
        elapsed_time = self.current_time - self.stats['start_time']
        
        title = (f"多UAV-USV协同巡检系统 - {self.scene_parser.scene_name}\\n"
                f"时间: {elapsed_time:.1f}s | 完成率: {completion_rate:.1f}% "
                f"({self.stats['completed_tasks']}/{self.stats['total_tasks']})")
        self.fig.suptitle(title, fontsize=14, fontweight='bold')
        
        self.current_time += 0.1  # 每帧增加0.1秒
    
    def start_animation(self, interval: int = 100):
        """
        启动动画
        
        Args:
            interval: 帧间隔（毫秒）
        """
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        self.animation = FuncAnimation(self.fig, self._draw_frame, interval=interval, 
                                     blit=False, repeat=True)
        
        print("🎬 动画已启动")
        plt.show()
    
    def save_animation(self, filename: str, duration: int = 10, fps: int = 10):
        """
        保存动画
        
        Args:
            filename: 文件名
            duration: 持续时间（秒）
            fps: 帧率
        """
        frames = duration * fps
        
        self.animation = FuncAnimation(self.fig, self._draw_frame, frames=frames, 
                                     interval=1000//fps, blit=False, repeat=False)
        
        try:
            self.animation.save(filename, writer='pillow', fps=fps)
            print(f"✅ 动画已保存: {filename}")
        except Exception as e:
            print(f"❌ 动画保存失败: {e}")
    
    def create_static_visualization(self) -> bool:
        """
        创建静态可视化图像
        
        Returns:
            是否创建成功
        """
        try:
            if not self.setup_visualization():
                return False
            
            # 添加示例数据
            safe_positions = self.scene_parser.get_safe_spawn_positions(4)
            
            # 添加智能体
            for i in range(3):  # 3个UAV
                self.add_agent(f"uav{i+1}", "UAV", safe_positions[i])
            self.add_agent("usv1", "USV", safe_positions[3])  # 1个USV
            
            # 添加任务（所有风机位置）
            task_positions = self.scene_parser.get_task_positions()
            for i, pos in enumerate(task_positions[:10]):  # 前10个任务
                self.add_task(i, pos)
            
            # 示例任务分配
            assignments = {
                "uav1": [0, 1, 2],
                "uav2": [3, 4, 5],
                "uav3": [6, 7],
                "usv1": [8, 9]
            }
            self.assign_tasks(assignments)
            
            # 绘制初始状态
            self._draw_frame(0)
            
            return True
            
        except Exception as e:
            print(f"❌ 静态可视化创建失败: {e}")
            return False


def create_demo_visualization():
    """创建演示可视化"""
    try:
        # 初始化场景解析器
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        data_dir = project_root / "data"
        
        parser = SceneParser("兴化湾海上风电场")
        
        if not parser.load_scene(data_dir):
            print("❌ 场景加载失败")
            return False
        
        # 创建可视化器
        visualizer = RealTimeVisualizer(parser)
        
        if visualizer.create_static_visualization():
            print("✅ 可视化创建成功")
            plt.show()
            return True
        else:
            print("❌ 可视化创建失败")
            return False
            
    except Exception as e:
        print(f"❌ 演示可视化失败: {e}")
        return False


if __name__ == "__main__":
    create_demo_visualization()