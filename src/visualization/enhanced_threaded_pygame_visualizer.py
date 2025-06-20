"""
增强版多线程Pygame动态可视化引擎
修复窗口控制、轨迹显示、透明度等问题
"""
import pygame
import pygame.gfxdraw
import cv2
import numpy as np
import math
import time
import threading
import queue
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from .enhanced_scene_parser import EnhancedSceneParser as SceneParser
from .charging_station import ChargingStationDetector, ChargingStation


class EnhancedThreadedAgent:
    """增强版多线程智能体类"""
    
    def __init__(self, agent_id: str, agent_type: str, start_pos: Tuple[int, int]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.position = list(start_pos)
        self.start_position = start_pos
        self.target_position = None
        self.trajectory = [start_pos]
        self.assigned_tasks = []
        self.current_task_index = 0
        self.status = "idle"
        self.battery_level = 100.0
        self.max_battery = 100.0
        self.speed = 5.0 if agent_type == "UAV" else 4.0
        self.color = self._get_agent_color()
        self.size = 14 if agent_type == "UAV" else 18
        self.inspection_timer = 0
        self.inspection_duration = 60
        
        # 线程安全锁
        self.lock = threading.Lock()
        
    def _get_agent_color(self) -> Tuple[int, int, int]:
        """获取智能体颜色"""
        if self.agent_type == "UAV":
            colors_uav = [
                (255, 80, 80),    # 红色
                (80, 80, 255),    # 蓝色
                (80, 255, 80),    # 绿色
                (255, 140, 0),    # 橙色
            ]
            agent_num = int(self.agent_id[-1]) - 1 if self.agent_id[-1].isdigit() else 0
            return colors_uav[agent_num % len(colors_uav)]
        else:  # USV
            return (0, 0, 139)
    
    def update(self, tasks: Dict[int, Any], charging_stations: List[ChargingStation]):
        """更新智能体状态 - 线程安全版"""
        with self.lock:
            # 电量管理
            if self.status == "moving":
                self.battery_level -= 0.04
            elif self.status == "inspecting":
                self.battery_level -= 0.1
            elif self.status == "charging" and self.battery_level < self.max_battery:
                self.battery_level = min(self.max_battery, self.battery_level + 0.8)
            
            self.battery_level = max(0, self.battery_level)
            
            # 状态机
            if self.status == "idle":
                self._handle_idle_state(tasks)
            elif self.status == "moving":
                self._handle_moving_state()
            elif self.status == "inspecting":
                self._handle_inspecting_state(tasks)
            elif self.status == "returning":
                self._handle_returning_state(charging_stations)
            elif self.status == "charging":
                self._handle_charging_state()
    
    def _handle_idle_state(self, tasks: Dict[int, Any]):
        """处理空闲状态"""
        if self.assigned_tasks and self.current_task_index < len(self.assigned_tasks):
            task_id = self.assigned_tasks[self.current_task_index]
            if task_id in tasks:
                task = tasks[task_id]
                if task['status'] == 'assigned':
                    self.target_position = task['position']
                    self.status = "moving"
                    task['status'] = 'in_progress'
        elif self.battery_level < 20:
            self.status = "returning"
    
    def _handle_moving_state(self):
        """处理移动状态"""
        if self.target_position:
            dx = self.target_position[0] - self.position[0]
            dy = self.target_position[1] - self.position[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < 10:
                self.position = list(self.target_position)
                self.trajectory.append(tuple(self.position))
                
                if self.status == "moving":
                    self.status = "inspecting"
                    self.inspection_timer = 0
                elif self.status == "returning":
                    self.status = "charging"
                
                self.target_position = None
            else:
                move_x = (dx / distance) * self.speed
                move_y = (dy / distance) * self.speed
                
                self.position[0] += move_x
                self.position[1] += move_y
                
                # 增强轨迹记录 - 更密集的记录
                if len(self.trajectory) == 0 or (
                    (self.position[0] - self.trajectory[-1][0])**2 + 
                    (self.position[1] - self.trajectory[-1][1])**2 > 150
                ):
                    self.trajectory.append(tuple(self.position))
                    
                    # 增加轨迹长度
                    if len(self.trajectory) > 25:
                        self.trajectory.pop(0)
    
    def _handle_inspecting_state(self, tasks: Dict[int, Any]):
        """处理检查状态"""
        self.inspection_timer += 1
        
        if self.inspection_timer >= self.inspection_duration:
            if self.assigned_tasks and self.current_task_index < len(self.assigned_tasks):
                task_id = self.assigned_tasks[self.current_task_index]
                if task_id in tasks:
                    tasks[task_id]['status'] = 'completed'
                
                self.current_task_index += 1
            
            self.status = "idle"
    
    def _handle_returning_state(self, charging_stations: List[ChargingStation]):
        """处理返回充电状态"""
        if not self.target_position:
            min_distance = float('inf')
            nearest_station = None
            
            for station in charging_stations:
                dx = station.position[0] - self.position[0]
                dy = station.position[1] - self.position[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_station = station
            
            if nearest_station:
                self.target_position = nearest_station.position
        
        self._handle_moving_state()
    
    def _handle_charging_state(self):
        """处理充电状态"""
        if self.battery_level >= 85:
            self.status = "idle"
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """获取线程安全的状态快照"""
        with self.lock:
            return {
                'agent_id': self.agent_id,
                'agent_type': self.agent_type,
                'position': tuple(self.position),
                'trajectory': list(self.trajectory),
                'status': self.status,
                'battery_level': self.battery_level,
                'color': self.color,
                'size': self.size
            }


class EnhancedThreadedPygameVisualizer:
    """增强版多线程Pygame动态可视化器"""
    
    def __init__(self, scene_parser: SceneParser, window_size: Tuple[int, int] = (1400, 1000)):
        """初始化增强版多线程可视化器"""
        self.scene_parser = scene_parser
        self.window_size = window_size
        
        # pygame初始化
        pygame.init()
        pygame.font.init()
        
        # 创建可调整大小的窗口 - 修复窗口控制问题
        self.screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        pygame.display.set_caption(f"Enhanced Multi-UAV-USV Cooperative Inspection System - {scene_parser.scene_name}")
        
        # 设置窗口图标和属性
        try:
            # 设置窗口最小尺寸
            pygame.display.set_mode(window_size, pygame.RESIZABLE)
        except:
            pass
        
        # 解决中文字符问题
        try:
            import platform
            if platform.system() == "Windows":
                self.font_small = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 16)
                self.font_medium = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 20)
                self.font_large = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 24)
            else:
                self.font_small = pygame.font.Font(None, 18)
                self.font_medium = pygame.font.Font(None, 22)
                self.font_large = pygame.font.Font(None, 26)
        except:
            self.font_small = pygame.font.Font(None, 18)
            self.font_medium = pygame.font.Font(None, 22)
            self.font_large = pygame.font.Font(None, 26)
        
        # 缩放因子
        self.scale_x = window_size[0] / scene_parser.image_size[0]
        self.scale_y = window_size[1] / scene_parser.image_size[1]
        
        # 可视化状态
        self.agents: Dict[str, EnhancedThreadedAgent] = {}
        self.tasks: Dict[int, Dict[str, Any]] = {}
        self.charging_stations: List[ChargingStation] = []
        self.background_surface = None
        
        # 多线程控制
        self.is_running = False
        self.is_paused = False
        self.logic_thread = None
        
        # 线程间通信
        self.state_queue = queue.Queue(maxsize=10)
        self.command_queue = queue.Queue()
        self.shared_state_lock = threading.Lock()
        
        # 渲染控制
        self.clock = pygame.time.Clock()
        self.fps = 60  # 保持60fps
        self.simulation_speed = 1.0
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'simulation_time': 0.0,
            'start_time': time.time()
        }
        
        # UI状态
        self.show_trajectories = True
        self.show_info = True
        
    def setup_visualization(self) -> bool:
        """设置可视化环境"""
        try:
            print("🎮 Setting up Enhanced Threaded Pygame Visualization...")
            print(f"   Window Size: {self.window_size} (Resizable)")
            print("   Thread Architecture: UI Rendering + Logic Computation Separated")
            
            # 背景图像处理
            if self.scene_parser.image is not None:
                print(f"   Processing Background Image: {self.scene_parser.image.shape}")
                
                image_rgb = cv2.cvtColor(self.scene_parser.image, cv2.COLOR_BGR2RGB)
                image_resized = cv2.resize(image_rgb, self.window_size, interpolation=cv2.INTER_AREA)
                
                self.background_surface = pygame.surfarray.make_surface(
                    image_resized.swapaxes(0, 1))
                print("   Background Surface Created Successfully")
            else:
                self.background_surface = pygame.Surface(self.window_size)
                self.background_surface.fill((100, 149, 237))
            
            # 部署充电桩
            print("   Deploying Charging Stations...")
            self._deploy_charging_stations()
            
            print("✅ Enhanced Threaded Pygame Visualization Setup Complete")
            return True
            
        except Exception as e:
            print(f"❌ Enhanced Threaded Pygame Visualization Setup Failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _deploy_charging_stations(self):
        """部署充电桩"""
        try:
            if self.scene_parser.image is not None:
                detector = ChargingStationDetector()
                wind_turbine_positions = self.scene_parser.get_task_positions()
                
                self.charging_stations = detector.deploy_charging_stations(
                    self.scene_parser.image, wind_turbine_positions, 3)
            else:
                default_positions = [(120, 400), (350, 50), (800, 650)]
                self.charging_stations = []
                for i, pos in enumerate(default_positions):
                    station = ChargingStation(i, pos, "shore")
                    self.charging_stations.append(station)
                
                print(f"✅ Deployed {len(self.charging_stations)} charging stations at default positions")
                
        except Exception as e:
            print(f"⚠️ Charging station deployment failed: {e}")
            default_positions = [(120, 400), (350, 50), (800, 650)]
            self.charging_stations = []
            for i, pos in enumerate(default_positions):
                station = ChargingStation(i, pos, "shore")
                self.charging_stations.append(station)
    
    def add_agent(self, agent_id: str, agent_type: str, start_pos: Tuple[int, int]):
        """添加智能体"""
        agent = EnhancedThreadedAgent(agent_id, agent_type, start_pos)
        self.agents[agent_id] = agent
        print(f"✅ Added Agent: {agent_id} ({agent_type}) at position {start_pos}")
    
    def add_task(self, task_id: int, position: Tuple[int, int], task_type: str = "inspection"):
        """添加任务"""
        with self.shared_state_lock:
            self.tasks[task_id] = {
                'task_id': task_id,
                'position': position,
                'type': task_type,
                'status': 'pending',
                'assigned_agent': None,
                'priority': 1.0,
                'completion_time': None
            }
            self.stats['total_tasks'] += 1
    
    def assign_tasks(self, assignments: Dict[str, List[int]]):
        """分配任务"""
        with self.shared_state_lock:
            for agent_id, task_ids in assignments.items():
                if agent_id in self.agents and task_ids:
                    self.agents[agent_id].assigned_tasks = task_ids
                    for task_id in task_ids:
                        if task_id in self.tasks:
                            self.tasks[task_id]['assigned_agent'] = agent_id
                            self.tasks[task_id]['status'] = 'assigned'
                    print(f"✅ {agent_id} assigned tasks: {task_ids}")
    
    def _logic_thread_worker(self):
        """逻辑计算线程工作函数"""
        print("🧠 Logic Computation Thread Started")
        
        logic_clock = pygame.time.Clock()
        
        while self.is_running:
            try:
                # 处理命令
                try:
                    command = self.command_queue.get_nowait()
                    if command == 'pause':
                        self.is_paused = not self.is_paused
                    elif command == 'toggle_trajectories':
                        self.show_trajectories = not self.show_trajectories
                    elif command == 'toggle_info':
                        self.show_info = not self.show_info
                    elif command == 'quit':
                        break
                except queue.Empty:
                    pass
                
                # 更新仿真
                if not self.is_paused:
                    with self.shared_state_lock:
                        # 更新仿真时间
                        self.stats['simulation_time'] += (1.0 / 30) * self.simulation_speed
                        
                        # 更新智能体
                        for agent in self.agents.values():
                            agent.update(self.tasks, self.charging_stations)
                        
                        # 更新统计信息
                        self.stats['completed_tasks'] = sum(
                            1 for task in self.tasks.values() if task['status'] == 'completed'
                        )
                
                # 发送状态更新到渲染线程
                try:
                    agent_states = {}
                    for agent_id, agent in self.agents.items():
                        agent_states[agent_id] = agent.get_state_snapshot()
                    
                    with self.shared_state_lock:
                        state_data = {
                            'agents': agent_states,
                            'tasks': dict(self.tasks),
                            'stats': dict(self.stats),
                            'ui_state': {
                                'show_trajectories': self.show_trajectories,
                                'show_info': self.show_info,
                                'is_paused': self.is_paused
                            }
                        }
                    
                    # 非阻塞发送状态
                    if not self.state_queue.full():
                        self.state_queue.put(state_data)
                except:
                    pass
                
                # 控制逻辑线程频率
                logic_clock.tick(30)
                
            except Exception as e:
                print(f"⚠️ Logic Thread Error: {e}")
                time.sleep(0.1)
        
        print("🧠 Logic Computation Thread Ended")
    
    def _scale_position(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """缩放位置坐标"""
        return (int(pos[0] * self.scale_x), int(pos[1] * self.scale_y))
    
    def _draw_background(self):
        """绘制背景"""
        # 如果窗口大小改变，重新缩放背景
        current_size = self.screen.get_size()
        if current_size != self.window_size:
            self.window_size = current_size
            self.scale_x = current_size[0] / self.scene_parser.image_size[0]
            self.scale_y = current_size[1] / self.scene_parser.image_size[1]
            
            # 重新缩放背景
            if self.scene_parser.image is not None:
                image_rgb = cv2.cvtColor(self.scene_parser.image, cv2.COLOR_BGR2RGB)
                image_resized = cv2.resize(image_rgb, current_size)
                self.background_surface = pygame.surfarray.make_surface(
                    image_resized.swapaxes(0, 1))
        
        self.screen.blit(self.background_surface, (0, 0))
    
    def _draw_dashed_line(self, surface, color, start_pos, end_pos, width=2, dash_length=8):
        """绘制虚线"""
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance == 0:
            return
        
        dashes = int(distance // (dash_length * 2))
        
        for i in range(dashes + 1):
            start = (
                start_pos[0] + (dx * i * 2 * dash_length) / distance,
                start_pos[1] + (dy * i * 2 * dash_length) / distance,
            )
            end = (
                min(end_pos[0], start_pos[0] + (dx * (i * 2 + 1) * dash_length) / distance),
                min(end_pos[1], start_pos[1] + (dy * (i * 2 + 1) * dash_length) / distance),
            )
            
            if start[0] <= end_pos[0] and start[1] <= end_pos[1]:
                pygame.draw.line(surface, color, start, end, width)
    
    def _draw_static_elements(self):
        """绘制静态元素 - 增强版（更透明）"""
        # 绘制风机 - 更透明
        viz_data = self.scene_parser.create_visualization_data()
        for turbine_data in viz_data['wind_turbines']:
            center = self._scale_position(turbine_data['center'])
            
            # 创建更透明的风机surface
            turbine_surface = pygame.Surface((30, 30), pygame.SRCALPHA)
            
            # 风机圆圈 - 大幅降低透明度
            pygame.draw.circle(turbine_surface, (255, 215, 0, 100), (15, 15), 10)  # 减少到100
            pygame.draw.circle(turbine_surface, (255, 165, 0, 120), (15, 15), 10, 2)  # 边框稍微明显
            
            self.screen.blit(turbine_surface, (center[0] - 15, center[1] - 15))
            
            # 风机标签 - 使用英文，更透明背景
            text = self.font_small.render(f"T{turbine_data['id']:02d}", True, (0, 0, 0))
            text_rect = text.get_rect(center=(center[0], center[1] - 25))
            
            bg_surface = pygame.Surface((text_rect.width + 4, text_rect.height + 2), pygame.SRCALPHA)
            bg_surface.fill((255, 255, 255, 140))  # 降低透明度
            self.screen.blit(bg_surface, text_rect.inflate(4, 2))
            self.screen.blit(text, text_rect)
        
        # 绘制陆地区域 - 适度降低透明度
        for land_data in viz_data['land_areas']:
            bbox = land_data['bbox']
            scaled_rect = pygame.Rect(
                int(bbox[0] * self.scale_x),
                int(bbox[1] * self.scale_y),
                int((bbox[2] - bbox[0]) * self.scale_x),
                int((bbox[3] - bbox[1]) * self.scale_y)
            )
            
            # 陆地填充 - 稍微降低透明度
            land_surface = pygame.Surface((scaled_rect.width, scaled_rect.height), pygame.SRCALPHA)
            land_surface.fill((210, 180, 140, 100))  # 从120降到100
            self.screen.blit(land_surface, scaled_rect)
            
            # 边框
            pygame.draw.rect(self.screen, (139, 69, 19, 130), scaled_rect, 2)
        
        # 绘制充电桩 - 保持清晰
        for station in self.charging_stations:
            center = self._scale_position(station.position)
            
            station_surface = pygame.Surface((20, 20), pygame.SRCALPHA)
            pygame.draw.rect(station_surface, (144, 238, 144, 220), (2, 2, 16, 16))
            pygame.draw.rect(station_surface, (0, 100, 0, 255), (2, 2, 16, 16), 3)
            self.screen.blit(station_surface, (center[0] - 10, center[1] - 10))
            
            text = self.font_small.render(f"C{station.station_id}", True, (0, 100, 0))
            text_rect = text.get_rect(center=(center[0], center[1] - 28))
            
            bg_surface = pygame.Surface((text_rect.width + 4, text_rect.height + 2), pygame.SRCALPHA)
            bg_surface.fill((255, 255, 255, 180))
            self.screen.blit(bg_surface, text_rect.inflate(4, 2))
            self.screen.blit(text, text_rect)
    
    def _draw_state_data(self, state_data: Dict[str, Any]):
        """绘制状态数据"""
        # 绘制虚线半透明轨迹 - 修复轨迹显示
        if state_data['ui_state']['show_trajectories']:
            for agent_state in state_data['agents'].values():
                if len(agent_state['trajectory']) > 1:
                    scaled_points = [self._scale_position(pos) for pos in agent_state['trajectory']]
                    
                    if len(scaled_points) > 1:
                        # 创建轨迹surface
                        trajectory_surface = pygame.Surface(self.window_size, pygame.SRCALPHA)
                        
                        # 绘制虚线轨迹
                        for i in range(len(scaled_points) - 1):
                            # 渐变透明度
                            alpha = int(40 + 60 * (i / max(1, len(scaled_points) - 1)))  # 40-100的透明度
                            color_with_alpha = (*agent_state['color'], alpha)
                            
                            # 绘制虚线段
                            self._draw_dashed_line(
                                trajectory_surface,
                                color_with_alpha,
                                scaled_points[i],
                                scaled_points[i + 1],
                                width=2,
                                dash_length=6
                            )
                        
                        # 将轨迹surface绘制到主screen
                        self.screen.blit(trajectory_surface, (0, 0))
        
        # 绘制任务
        for task in state_data['tasks'].values():
            pos = self._scale_position(task['position'])
            
            color = {
                'completed': (0, 255, 0),
                'in_progress': (255, 165, 0),
                'assigned': (0, 0, 255),
                'pending': (255, 0, 0)
            }.get(task['status'], (128, 128, 128))
            
            pygame.draw.circle(self.screen, color, pos, 8)
            pygame.draw.circle(self.screen, (0, 0, 0), pos, 8, 2)
        
        # 绘制智能体
        for agent_state in state_data['agents'].values():
            pos = self._scale_position(agent_state['position'])
            
            if agent_state['agent_type'] == "UAV":
                points = [
                    (pos[0], pos[1] - agent_state['size']),
                    (pos[0] - agent_state['size']//2, pos[1] + agent_state['size']//2),
                    (pos[0] + agent_state['size']//2, pos[1] + agent_state['size']//2)
                ]
                pygame.draw.polygon(self.screen, agent_state['color'], points)
                pygame.draw.polygon(self.screen, (0, 0, 0), points, 2)
            else:
                rect = pygame.Rect(pos[0] - agent_state['size']//2, pos[1] - agent_state['size']//2, 
                                 agent_state['size'], agent_state['size'])
                pygame.draw.rect(self.screen, agent_state['color'], rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)
            
            # 状态信息
            if state_data['ui_state']['show_info']:
                status_color = {
                    'idle': (173, 216, 230),
                    'moving': (255, 165, 0),
                    'inspecting': (255, 255, 0),
                    'returning': (255, 182, 193),
                    'charging': (144, 238, 144)
                }.get(agent_state['status'], (255, 255, 255))
                
                status_text = {
                    'idle': 'IDLE',
                    'moving': 'MOVE',
                    'inspecting': 'INSPECT',
                    'returning': 'RETURN',
                    'charging': 'CHARGE'
                }.get(agent_state['status'], 'UNKNOWN')
                
                info_text = f"{agent_state['agent_id']} | {status_text} | {agent_state['battery_level']:.0f}%"
                text = self.font_small.render(info_text, True, (0, 0, 0))
                
                text_rect = text.get_rect()
                text_rect.topleft = (pos[0] + 18, pos[1] - 10)
                bg_rect = text_rect.inflate(4, 2)
                
                bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
                bg_surface.fill((*status_color, 200))
                self.screen.blit(bg_surface, bg_rect)
                
                pygame.draw.rect(self.screen, (0, 0, 0), bg_rect, 1)
                self.screen.blit(text, text_rect)
        
        # 绘制UI信息
        completed = state_data['stats']['completed_tasks']
        total = state_data['stats']['total_tasks']
        completion_rate = (completed / max(1, total)) * 100
        
        info_lines = [
            f"Time: {state_data['stats']['simulation_time']:.1f}s",
            f"Completion: {completion_rate:.0f}% ({completed}/{total})",
            f"Threads: UI + Logic Separated",
            f"Status: {'⏸️PAUSED' if state_data['ui_state']['is_paused'] else '▶️RUNNING'}",
            f"Window: {self.window_size[0]}x{self.window_size[1]} (Resizable)",
            "",
            "SPACE:Pause  T:Trajectories  I:Info  ESC:Exit"
        ]
        
        y_offset = 10
        for line in info_lines:
            if line:
                text = self.font_small.render(line, True, (255, 255, 255))
                text_rect = text.get_rect()
                text_rect.topleft = (10, y_offset)
                
                bg_surface = pygame.Surface((text_rect.width + 4, text_rect.height + 2), pygame.SRCALPHA)
                bg_surface.fill((0, 0, 0, 150))
                self.screen.blit(bg_surface, text_rect.inflate(4, 2))
                
                self.screen.blit(text, text_rect)
            
            y_offset += 20
    
    def run(self):
        """运行多线程可视化主循环"""
        print("🎮 Starting Enhanced Threaded Pygame Visualization...")
        print(f"   UI Thread: {self.fps}fps Rendering")
        print(f"   Logic Thread: 30fps Simulation")
        print(f"   Agents: {len(self.agents)}, Tasks: {len(self.tasks)}")
        print("   Window: Resizable with full controls")
        
        self.is_running = True
        
        # 启动逻辑线程
        self.logic_thread = threading.Thread(target=self._logic_thread_worker, daemon=True)
        self.logic_thread.start()
        
        # UI渲染主循环
        latest_state = None
        
        try:
            while self.is_running:
                # 处理pygame事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.command_queue.put('quit')
                        self.is_running = False
                        break
                    elif event.type == pygame.VIDEORESIZE:
                        # 处理窗口大小改变
                        self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                        print(f"Window resized to: {event.size}")
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.command_queue.put('quit')
                            self.is_running = False
                            break
                        elif event.key == pygame.K_SPACE:
                            self.command_queue.put('pause')
                        elif event.key == pygame.K_t:
                            self.command_queue.put('toggle_trajectories')
                        elif event.key == pygame.K_i:
                            self.command_queue.put('toggle_info')
                
                # 获取最新状态
                try:
                    while not self.state_queue.empty():
                        latest_state = self.state_queue.get_nowait()
                except queue.Empty:
                    pass
                
                # 渲染
                self.screen.fill((0, 0, 0))
                self._draw_background()
                self._draw_static_elements()
                
                if latest_state:
                    self._draw_state_data(latest_state)
                
                pygame.display.flip()
                self.clock.tick(self.fps)
                
        except KeyboardInterrupt:
            print("\n⏹️ User Interrupted")
        except Exception as e:
            print(f"❌ Render Thread Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            
            # 等待逻辑线程结束
            if self.logic_thread and self.logic_thread.is_alive():
                self.logic_thread.join(timeout=2)
            
            pygame.quit()
            print("✅ Enhanced Threaded Pygame Exited")