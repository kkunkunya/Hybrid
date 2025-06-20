"""
增强版Pygame动态可视化引擎
解决中文字符乱码、半透明显示、虚线轨迹等视觉问题
"""
import pygame
import pygame.gfxdraw
import cv2
import numpy as np
import math
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from .enhanced_scene_parser import EnhancedSceneParser as SceneParser
from .charging_station import ChargingStationDetector, ChargingStation


class EnhancedPygameAgent:
    """增强版Pygame智能体类"""
    
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
        self.speed = 3.0 if agent_type == "UAV" else 2.0
        self.color = self._get_agent_color()
        self.size = 15 if agent_type == "UAV" else 20
        self.inspection_timer = 0
        self.inspection_duration = 180
        
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
            return (0, 0, 139)  # 深蓝色
    
    def update(self, tasks: Dict[int, Any], charging_stations: List[ChargingStation]):
        """更新智能体状态"""
        # 消耗电量
        if self.status == "moving":
            self.battery_level -= 0.02
        elif self.status == "inspecting":
            self.battery_level -= 0.05
        elif self.status == "charging" and self.battery_level < self.max_battery:
            self.battery_level = min(self.max_battery, self.battery_level + 0.3)
        
        self.battery_level = max(0, self.battery_level)
        
        # 状态机逻辑
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
        elif self.battery_level < 30:
            self.status = "returning"
    
    def _handle_moving_state(self):
        """处理移动状态"""
        if self.target_position:
            dx = self.target_position[0] - self.position[0]
            dy = self.target_position[1] - self.position[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < 5:
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
                
                # 优化轨迹记录
                if len(self.trajectory) == 0 or (
                    (self.position[0] - self.trajectory[-1][0])**2 + 
                    (self.position[1] - self.trajectory[-1][1])**2 > 100
                ):
                    self.trajectory.append(tuple(self.position))
                    
                    if len(self.trajectory) > 30:  # 增加轨迹长度
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
        if self.battery_level >= 95:
            self.status = "idle"


class EnhancedPygameVisualizer:
    """增强版Pygame动态可视化器"""
    
    def __init__(self, scene_parser: SceneParser, window_size: Tuple[int, int] = (1400, 1000)):
        """初始化增强版可视化器"""
        self.scene_parser = scene_parser
        self.window_size = window_size
        
        # pygame初始化
        pygame.init()
        pygame.font.init()
        
        # 创建窗口
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption(f"Multi-UAV-USV Cooperative Inspection System - {scene_parser.scene_name}")
        
        # 解决中文字符问题 - 使用系统字体
        try:
            # 尝试加载中文字体
            import platform
            if platform.system() == "Windows":
                self.font_small = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 16)
                self.font_medium = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 20)
                self.font_large = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 24)
            else:
                # Linux/Mac 使用默认字体，但用英文标签
                self.font_small = pygame.font.Font(None, 18)
                self.font_medium = pygame.font.Font(None, 22)
                self.font_large = pygame.font.Font(None, 26)
        except:
            # 回退到默认字体
            self.font_small = pygame.font.Font(None, 18)
            self.font_medium = pygame.font.Font(None, 22)
            self.font_large = pygame.font.Font(None, 26)
        
        # 缩放因子
        self.scale_x = window_size[0] / scene_parser.image_size[0]
        self.scale_y = window_size[1] / scene_parser.image_size[1]
        
        # 可视化状态
        self.agents: Dict[str, EnhancedPygameAgent] = {}
        self.tasks: Dict[int, Dict[str, Any]] = {}
        self.charging_stations: List[ChargingStation] = []
        self.background_surface = None
        
        # 控制状态
        self.is_running = False
        self.is_paused = False
        self.clock = pygame.time.Clock()
        self.fps = 60
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
            print("🎮 Setting up Enhanced Pygame Visualization...")
            print(f"   Window Size: {self.window_size}")
            print(f"   Scene Image: {'Loaded' if self.scene_parser.image is not None else 'Not Loaded'}")
            
            # 加载背景图像
            if self.scene_parser.image is not None:
                print(f"   Original Image Size: {self.scene_parser.image.shape}")
                
                image_rgb = cv2.cvtColor(self.scene_parser.image, cv2.COLOR_BGR2RGB)
                image_resized = cv2.resize(image_rgb, self.window_size)
                
                self.background_surface = pygame.surfarray.make_surface(
                    image_resized.swapaxes(0, 1))
                print(f"   Background Surface Created: {self.background_surface.get_size()}")
            else:
                print("   Using Solid Color Background")
                self.background_surface = pygame.Surface(self.window_size)
                self.background_surface.fill((100, 149, 237))
            
            # 部署充电桩
            print("   Deploying Charging Stations...")
            self._deploy_charging_stations()
            
            print("✅ Enhanced Pygame Visualization Setup Complete")
            return True
            
        except Exception as e:
            print(f"❌ Enhanced Pygame Visualization Setup Failed: {e}")
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
                default_positions = [(150, 450), (400, 50), (850, 850)]
                self.charging_stations = []
                for i, pos in enumerate(default_positions):
                    station = ChargingStation(i, pos, "shore")
                    self.charging_stations.append(station)
                
                print(f"✅ Deployed {len(self.charging_stations)} charging stations at default positions")
                
        except Exception as e:
            print(f"⚠️ Charging station deployment failed: {e}")
            default_positions = [(150, 450), (400, 50), (850, 850)]
            self.charging_stations = []
            for i, pos in enumerate(default_positions):
                station = ChargingStation(i, pos, "shore")
                self.charging_stations.append(station)
    
    def add_agent(self, agent_id: str, agent_type: str, start_pos: Tuple[int, int]):
        """添加智能体"""
        agent = EnhancedPygameAgent(agent_id, agent_type, start_pos)
        self.agents[agent_id] = agent
        print(f"✅ Added Agent: {agent_id} ({agent_type}) at position {start_pos}")
    
    def add_task(self, task_id: int, position: Tuple[int, int], task_type: str = "inspection"):
        """添加任务"""
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
        for agent_id, task_ids in assignments.items():
            if agent_id in self.agents and task_ids:
                self.agents[agent_id].assigned_tasks = task_ids
                for task_id in task_ids:
                    if task_id in self.tasks:
                        self.tasks[task_id]['assigned_agent'] = agent_id
                        self.tasks[task_id]['status'] = 'assigned'
                print(f"✅ {agent_id} assigned tasks: {task_ids}")
    
    def _scale_position(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """缩放位置坐标"""
        return (int(pos[0] * self.scale_x), int(pos[1] * self.scale_y))
    
    def _draw_background(self):
        """绘制背景"""
        self.screen.blit(self.background_surface, (0, 0))
    
    def _draw_dashed_line(self, surface, color, start_pos, end_pos, width=2, dash_length=10):
        """绘制虚线"""
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance == 0:
            return
        
        dashes = int(distance // (dash_length * 2))
        
        for i in range(dashes):
            start = (
                start_pos[0] + (dx * i * 2 * dash_length) / distance,
                start_pos[1] + (dy * i * 2 * dash_length) / distance,
            )
            end = (
                start_pos[0] + (dx * (i * 2 + 1) * dash_length) / distance,
                start_pos[1] + (dy * (i * 2 + 1) * dash_length) / distance,
            )
            
            pygame.draw.line(surface, color, start, end, width)
    
    def _draw_static_elements(self):
        """绘制静态元素 - 增强版（半透明）"""
        # 绘制风机 - 半透明
        for turbine_data in self.scene_parser.create_visualization_data()['wind_turbines']:
            center = self._scale_position(turbine_data['center'])
            
            # 创建半透明surface
            turbine_surface = pygame.Surface((30, 30), pygame.SRCALPHA)
            
            # 风机圆圈 - 半透明黄色
            pygame.draw.circle(turbine_surface, (255, 215, 0, 180), (15, 15), 12)
            pygame.draw.circle(turbine_surface, (255, 165, 0, 200), (15, 15), 12, 2)
            
            # 绘制到主surface
            self.screen.blit(turbine_surface, (center[0] - 15, center[1] - 15))
            
            # 风机标签 - 使用英文避免乱码
            text = self.font_small.render(f"T{turbine_data['id']:02d}", True, (0, 0, 0))
            text_rect = text.get_rect(center=(center[0], center[1] - 25))
            
            # 半透明背景
            bg_surface = pygame.Surface((text_rect.width + 4, text_rect.height + 2), pygame.SRCALPHA)
            bg_surface.fill((255, 255, 255, 180))
            self.screen.blit(bg_surface, text_rect.inflate(4, 2))
            self.screen.blit(text, text_rect)
        
        # 绘制陆地区域 - 半透明
        for land_data in self.scene_parser.create_visualization_data()['land_areas']:
            bbox = land_data['bbox']
            scaled_rect = pygame.Rect(
                int(bbox[0] * self.scale_x),
                int(bbox[1] * self.scale_y),
                int((bbox[2] - bbox[0]) * self.scale_x),
                int((bbox[3] - bbox[1]) * self.scale_y)
            )
            
            # 半透明陆地填充
            land_surface = pygame.Surface((scaled_rect.width, scaled_rect.height), pygame.SRCALPHA)
            land_surface.fill((210, 180, 140, 120))  # 半透明棕色
            self.screen.blit(land_surface, scaled_rect)
            
            # 边框
            pygame.draw.rect(self.screen, (139, 69, 19, 150), scaled_rect, 2)
        
        # 绘制充电桩 - 半透明
        for station in self.charging_stations:
            center = self._scale_position(station.position)
            
            # 半透明充电桩
            station_surface = pygame.Surface((20, 20), pygame.SRCALPHA)
            pygame.draw.rect(station_surface, (144, 238, 144, 200), (2, 2, 16, 16))
            pygame.draw.rect(station_surface, (0, 100, 0, 255), (2, 2, 16, 16), 3)
            self.screen.blit(station_surface, (center[0] - 10, center[1] - 10))
            
            # 充电桩标签 - 使用英文
            text = self.font_small.render(f"C{station.station_id}", True, (0, 100, 0))
            text_rect = text.get_rect(center=(center[0], center[1] - 28))
            
            # 半透明背景
            bg_surface = pygame.Surface((text_rect.width + 4, text_rect.height + 2), pygame.SRCALPHA)
            bg_surface.fill((255, 255, 255, 180))
            self.screen.blit(bg_surface, text_rect.inflate(4, 2))
            self.screen.blit(text, text_rect)
    
    def _draw_trajectories(self):
        """绘制轨迹 - 增强版（虚线半透明）"""
        if not self.show_trajectories:
            return
            
        for agent in self.agents.values():
            if len(agent.trajectory) > 1:
                scaled_points = [self._scale_position(pos) for pos in agent.trajectory]
                
                # 创建半透明surface用于轨迹
                if len(scaled_points) > 1:
                    # 计算轨迹颜色（半透明）
                    trajectory_color = (*agent.color, 100)  # 添加alpha通道
                    
                    # 创建轨迹surface
                    trajectory_surface = pygame.Surface(self.window_size, pygame.SRCALPHA)
                    
                    # 绘制虚线轨迹
                    for i in range(len(scaled_points) - 1):
                        # 渐变透明度
                        alpha = int(60 + 80 * (i / max(1, len(scaled_points) - 1)))
                        color_with_alpha = (*agent.color, alpha)
                        
                        # 绘制虚线段
                        self._draw_dashed_line(
                            trajectory_surface,
                            color_with_alpha,
                            scaled_points[i],
                            scaled_points[i + 1],
                            width=2,
                            dash_length=8
                        )
                    
                    # 将轨迹surface绘制到主screen
                    self.screen.blit(trajectory_surface, (0, 0))
    
    def _draw_agents(self):
        """绘制智能体"""
        for agent in self.agents.values():
            pos = self._scale_position(agent.position)
            
            # 绘制智能体
            if agent.agent_type == "UAV":
                # UAV - 三角形
                points = [
                    (pos[0], pos[1] - agent.size),
                    (pos[0] - agent.size//2, pos[1] + agent.size//2),
                    (pos[0] + agent.size//2, pos[1] + agent.size//2)
                ]
                pygame.draw.polygon(self.screen, agent.color, points)
                pygame.draw.polygon(self.screen, (0, 0, 0), points, 2)
            else:
                # USV - 方形
                rect = pygame.Rect(pos[0] - agent.size//2, pos[1] - agent.size//2, 
                                 agent.size, agent.size)
                pygame.draw.rect(self.screen, agent.color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)
            
            # 状态信息
            if self.show_info:
                status_color = {
                    'idle': (173, 216, 230),
                    'moving': (255, 165, 0),
                    'inspecting': (255, 255, 0),
                    'returning': (255, 182, 193),
                    'charging': (144, 238, 144)
                }.get(agent.status, (255, 255, 255))
                
                # 使用英文标签避免乱码
                status_text = {
                    'idle': 'IDLE',
                    'moving': 'MOVE',
                    'inspecting': 'INSPECT',
                    'returning': 'RETURN',
                    'charging': 'CHARGE'
                }.get(agent.status, 'UNKNOWN')
                
                info_text = f"{agent.agent_id} | {status_text} | {agent.battery_level:.0f}%"
                text = self.font_small.render(info_text, True, (0, 0, 0))
                
                # 半透明背景
                text_rect = text.get_rect()
                text_rect.topleft = (pos[0] + 20, pos[1] - 10)
                bg_rect = text_rect.inflate(4, 2)
                
                bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
                bg_surface.fill((*status_color, 200))
                self.screen.blit(bg_surface, bg_rect)
                
                pygame.draw.rect(self.screen, (0, 0, 0), bg_rect, 1)
                self.screen.blit(text, text_rect)
    
    def _draw_tasks(self):
        """绘制任务"""
        for task in self.tasks.values():
            pos = self._scale_position(task['position'])
            
            # 任务状态颜色
            color = {
                'completed': (0, 255, 0),
                'in_progress': (255, 165, 0),
                'assigned': (0, 0, 255),
                'pending': (255, 0, 0)
            }.get(task['status'], (128, 128, 128))
            
            # 绘制任务圆圈
            pygame.draw.circle(self.screen, color, pos, 8)
            pygame.draw.circle(self.screen, (0, 0, 0), pos, 8, 2)
    
    def _draw_ui(self):
        """绘制UI信息 - 增强版（使用英文）"""
        # 统计信息
        completed = sum(1 for task in self.tasks.values() if task['status'] == 'completed')
        completion_rate = (completed / max(1, len(self.tasks))) * 100
        
        info_lines = [
            f"Time: {self.stats['simulation_time']:.1f}s",
            f"Completion: {completion_rate:.1f}% ({completed}/{len(self.tasks)})",
            f"Agents: {len(self.agents)}",
            f"FPS: {self.clock.get_fps():.0f}",
            "",
            "Controls:",
            "SPACE: Pause/Resume",
            "T: Toggle Trajectories",
            "I: Toggle Info",
            "ESC: Exit"
        ]
        
        y_offset = 10
        for line in info_lines:
            if line:
                text = self.font_small.render(line, True, (255, 255, 255))
                text_rect = text.get_rect()
                text_rect.topleft = (10, y_offset)
                
                # 半透明背景
                bg_surface = pygame.Surface((text_rect.width + 4, text_rect.height + 2), pygame.SRCALPHA)
                bg_surface.fill((0, 0, 0, 150))
                self.screen.blit(bg_surface, text_rect.inflate(4, 2))
                
                self.screen.blit(text, text_rect)
            
            y_offset += 22
    
    def update(self):
        """更新仿真状态"""
        if not self.is_paused:
            # 更新仿真时间
            self.stats['simulation_time'] += 1.0 / self.fps * self.simulation_speed
            
            # 更新智能体
            for agent in self.agents.values():
                agent.update(self.tasks, self.charging_stations)
            
            # 更新统计信息
            self.stats['completed_tasks'] = sum(
                1 for task in self.tasks.values() if task['status'] == 'completed'
            )
    
    def draw(self):
        """绘制整个场景"""
        # 清屏
        self.screen.fill((0, 0, 0))
        
        # 绘制背景
        self._draw_background()
        
        # 绘制静态元素
        self._draw_static_elements()
        
        # 绘制轨迹
        self._draw_trajectories()
        
        # 绘制任务
        self._draw_tasks()
        
        # 绘制智能体
        self._draw_agents()
        
        # 绘制UI
        self._draw_ui()
        
        # 更新显示
        pygame.display.flip()
    
    def handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
                    print(f"{'Paused' if self.is_paused else 'Resumed'}")
                elif event.key == pygame.K_t:
                    self.show_trajectories = not self.show_trajectories
                    print(f"Trajectories: {'ON' if self.show_trajectories else 'OFF'}")
                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info
                    print(f"Info Display: {'ON' if self.show_info else 'OFF'}")
        
        return True
    
    def run(self):
        """运行可视化主循环"""
        print("🎮 Starting Enhanced Pygame Visualization...")
        print(f"   Initial Status Check:")
        print(f"   - Background Surface: {'✅' if self.background_surface else '❌'}")
        print(f"   - Agent Count: {len(self.agents)}")
        print(f"   - Task Count: {len(self.tasks)}")
        print(f"   - Charging Station Count: {len(self.charging_stations)}")
        
        self.is_running = True
        frame_count = 0
        
        try:
            while self.is_running:
                # 处理事件
                if not self.handle_events():
                    break
                
                # 更新状态
                self.update()
                
                # 绘制场景
                self.draw()
                
                # 控制帧率
                self.clock.tick(self.fps)
                
                # 每60帧打印一次调试信息
                frame_count += 1
                if frame_count % 60 == 0:
                    print(f"Running normally: Frame {frame_count}, FPS: {self.clock.get_fps():.1f}")
                
        except KeyboardInterrupt:
            print("\n⏹️ User Interrupted")
        except Exception as e:
            print(f"❌ Runtime Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            pygame.quit()
            print("✅ Enhanced Pygame Exited")