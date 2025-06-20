"""
优化版Pygame动态可视化引擎
解决性能问题和无响应问题
"""
import pygame
import pygame.gfxdraw
import cv2
import numpy as np
import math
import time
import threading
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from .scene_parser import SceneParser
from .charging_station import ChargingStationDetector, ChargingStation


class OptimizedPygameAgent:
    """优化版Pygame智能体类"""
    
    def __init__(self, agent_id: str, agent_type: str, start_pos: Tuple[int, int]):
        """初始化智能体"""
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
        self.speed = 4.0 if agent_type == "UAV" else 3.0  # 增加速度，减少移动时间
        self.color = self._get_agent_color()
        self.size = 12 if agent_type == "UAV" else 16  # 稍微缩小尺寸
        self.inspection_timer = 0
        self.inspection_duration = 90  # 缩短检查时间 1.5秒 @30fps
        
    def _get_agent_color(self) -> Tuple[int, int, int]:
        """获取智能体颜色"""
        if self.agent_type == "UAV":
            colors_uav = [
                (255, 100, 100),  # 红色
                (100, 100, 255),  # 蓝色
                (100, 255, 100),  # 绿色
                (255, 165, 0),    # 橙色
            ]
            agent_num = int(self.agent_id[-1]) - 1 if self.agent_id[-1].isdigit() else 0
            return colors_uav[agent_num % len(colors_uav)]
        else:  # USV
            return (0, 0, 139)  # 深蓝色
    
    def update(self, tasks: Dict[int, Any], charging_stations: List[ChargingStation]):
        """更新智能体状态 - 优化版"""
        # 简化电量消耗计算
        if self.status == "moving":
            self.battery_level -= 0.03
        elif self.status == "inspecting":
            self.battery_level -= 0.08
        elif self.status == "charging" and self.battery_level < self.max_battery:
            self.battery_level = min(self.max_battery, self.battery_level + 0.5)
        
        self.battery_level = max(0, self.battery_level)
        
        # 简化状态机
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
        elif self.battery_level < 25:
            self.status = "returning"
    
    def _handle_moving_state(self):
        """处理移动状态"""
        if self.target_position:
            dx = self.target_position[0] - self.position[0]
            dy = self.target_position[1] - self.position[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < 8:  # 增加到达阈值
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
                    (self.position[1] - self.trajectory[-1][1])**2 > 200
                ):
                    self.trajectory.append(tuple(self.position))
                    
                    # 限制轨迹长度
                    if len(self.trajectory) > 20:
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
        if self.battery_level >= 90:
            self.status = "idle"


class OptimizedPygameVisualizer:
    """优化版Pygame动态可视化器"""
    
    def __init__(self, scene_parser: SceneParser, window_size: Tuple[int, int] = (1000, 700)):
        """初始化优化版可视化器"""
        self.scene_parser = scene_parser
        self.window_size = window_size
        
        # pygame初始化
        pygame.init()
        pygame.font.init()
        
        # 创建窗口
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption(f"多UAV-USV协同巡检系统 (优化版) - {scene_parser.scene_name}")
        
        # 字体
        self.font_small = pygame.font.Font(None, 18)
        self.font_medium = pygame.font.Font(None, 22)
        
        # 缩放因子
        self.scale_x = window_size[0] / scene_parser.image_size[0]
        self.scale_y = window_size[1] / scene_parser.image_size[1]
        
        # 可视化状态
        self.agents: Dict[str, OptimizedPygameAgent] = {}
        self.tasks: Dict[int, Dict[str, Any]] = {}
        self.charging_stations: List[ChargingStation] = []
        self.background_surface = None
        
        # 优化的控制状态
        self.is_running = False
        self.is_paused = False
        self.clock = pygame.time.Clock()
        self.fps = 30  # 降低帧率
        self.simulation_speed = 1.5  # 加快仿真速度
        
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
        
        # 性能监控
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.performance_warning = False
        
    def setup_visualization(self) -> bool:
        """设置可视化环境 - 优化版"""
        try:
            print("🎮 设置优化版Pygame可视化环境...")
            print(f"   窗口大小: {self.window_size}")
            print(f"   目标帧率: {self.fps}fps")
            print(f"   仿真速度: {self.simulation_speed}x")
            
            # 优化的背景图像处理
            if self.scene_parser.image is not None:
                print(f"   原始图像尺寸: {self.scene_parser.image.shape}")
                
                # 优化图像处理 - 使用更高效的缩放
                image_rgb = cv2.cvtColor(self.scene_parser.image, cv2.COLOR_BGR2RGB)
                image_resized = cv2.resize(image_rgb, self.window_size, interpolation=cv2.INTER_LINEAR)
                
                # 创建pygame surface
                self.background_surface = pygame.surfarray.make_surface(
                    image_resized.swapaxes(0, 1))
                print(f"   背景Surface创建成功: {self.background_surface.get_size()}")
            else:
                print("   使用纯色背景")
                self.background_surface = pygame.Surface(self.window_size)
                self.background_surface.fill((100, 149, 237))
            
            # 部署充电桩
            print("   部署充电桩...")
            self._deploy_charging_stations()
            
            print("✅ 优化版Pygame可视化环境设置完成")
            return True
            
        except Exception as e:
            print(f"❌ 优化版Pygame可视化环境设置失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _deploy_charging_stations(self):
        """部署充电桩 - 优化版"""
        try:
            if self.scene_parser.image is not None:
                detector = ChargingStationDetector()
                wind_turbine_positions = self.scene_parser.get_task_positions()
                
                self.charging_stations = detector.deploy_charging_stations(
                    self.scene_parser.image, wind_turbine_positions, 3)
            else:
                # 使用默认位置
                default_positions = [(100, 350), (300, 40), (700, 600)]
                self.charging_stations = []
                for i, pos in enumerate(default_positions):
                    station = ChargingStation(i, pos, "shore")
                    self.charging_stations.append(station)
                
                print(f"✅ 使用默认位置部署 {len(self.charging_stations)} 个充电桩")
                
        except Exception as e:
            print(f"⚠️ 充电桩部署失败: {e}")
            # 备用方案
            default_positions = [(100, 350), (300, 40), (700, 600)]
            self.charging_stations = []
            for i, pos in enumerate(default_positions):
                station = ChargingStation(i, pos, "shore")
                self.charging_stations.append(station)
    
    def add_agent(self, agent_id: str, agent_type: str, start_pos: Tuple[int, int]):
        """添加智能体"""
        agent = OptimizedPygameAgent(agent_id, agent_type, start_pos)
        self.agents[agent_id] = agent
        print(f"✅ 添加智能体: {agent_id} ({agent_type}) 于位置 {start_pos}")
    
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
                print(f"✅ {agent_id} 分配任务: {task_ids}")
    
    def _scale_position(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """缩放位置坐标"""
        return (int(pos[0] * self.scale_x), int(pos[1] * self.scale_y))
    
    def _draw_background(self):
        """绘制背景"""
        self.screen.blit(self.background_surface, (0, 0))
    
    def _draw_static_elements(self):
        """绘制静态元素 - 优化版"""
        # 绘制风机 - 简化版本
        viz_data = self.scene_parser.create_visualization_data()
        for turbine_data in viz_data['wind_turbines']:
            center = self._scale_position(turbine_data['center'])
            
            # 简化的风机绘制
            pygame.draw.circle(self.screen, (255, 215, 0), center, 8)
            pygame.draw.circle(self.screen, (255, 165, 0), center, 8, 2)
        
        # 绘制陆地区域 - 简化版本
        for land_data in viz_data['land_areas']:
            bbox = land_data['bbox']
            scaled_rect = pygame.Rect(
                int(bbox[0] * self.scale_x),
                int(bbox[1] * self.scale_y),
                int((bbox[2] - bbox[0]) * self.scale_x),
                int((bbox[3] - bbox[1]) * self.scale_y)
            )
            pygame.draw.rect(self.screen, (210, 180, 140), scaled_rect)
            pygame.draw.rect(self.screen, (139, 69, 19), scaled_rect, 2)
        
        # 绘制充电桩
        for station in self.charging_stations:
            center = self._scale_position(station.position)
            
            # 充电桩方块
            rect = pygame.Rect(center[0] - 6, center[1] - 6, 12, 12)
            pygame.draw.rect(self.screen, (144, 238, 144), rect)
            pygame.draw.rect(self.screen, (0, 100, 0), rect, 2)
            
            # 充电桩标签
            text = self.font_small.render(f"C{station.station_id}", True, (0, 100, 0))
            text_rect = text.get_rect(center=(center[0], center[1] - 20))
            self.screen.blit(text, text_rect)
    
    def _draw_trajectories(self):
        """绘制轨迹 - 优化版"""
        if not self.show_trajectories:
            return
            
        for agent in self.agents.values():
            if len(agent.trajectory) > 1:
                # 简化轨迹绘制
                scaled_points = [self._scale_position(pos) for pos in agent.trajectory]
                
                # 使用简单的线条绘制
                if len(scaled_points) > 1:
                    pygame.draw.lines(self.screen, agent.color, False, scaled_points, 2)
    
    def _draw_agents(self):
        """绘制智能体 - 优化版"""
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
            
            # 简化的状态信息
            if self.show_info:
                status_color = {
                    'idle': (173, 216, 230),
                    'moving': (255, 165, 0),
                    'inspecting': (255, 255, 0),
                    'returning': (255, 182, 193),
                    'charging': (144, 238, 144)
                }.get(agent.status, (255, 255, 255))
                
                info_text = f"{agent.agent_id}|{agent.battery_level:.0f}%"
                text = self.font_small.render(info_text, True, (0, 0, 0))
                
                text_rect = text.get_rect()
                text_rect.topleft = (pos[0] + 15, pos[1] - 8)
                bg_rect = text_rect.inflate(2, 2)
                
                pygame.draw.rect(self.screen, status_color, bg_rect)
                self.screen.blit(text, text_rect)
    
    def _draw_tasks(self):
        """绘制任务 - 优化版"""
        for task in self.tasks.values():
            pos = self._scale_position(task['position'])
            
            # 任务状态颜色
            color = {
                'completed': (0, 255, 0),
                'in_progress': (255, 165, 0),
                'assigned': (0, 0, 255),
                'pending': (255, 0, 0)
            }.get(task['status'], (128, 128, 128))
            
            # 简化的任务圆圈
            pygame.draw.circle(self.screen, color, pos, 6)
            pygame.draw.circle(self.screen, (0, 0, 0), pos, 6, 1)
    
    def _draw_ui(self):
        """绘制UI信息 - 优化版"""
        # 简化的统计信息
        completed = sum(1 for task in self.tasks.values() if task['status'] == 'completed')
        completion_rate = (completed / max(1, len(self.tasks))) * 100
        
        info_lines = [
            f"时间: {self.stats['simulation_time']:.1f}s",
            f"完成: {completion_rate:.0f}% ({completed}/{len(self.tasks)})",
            f"FPS: {self.clock.get_fps():.0f}",
            "",
            "空格:暂停 T:轨迹 I:信息 ESC:退出"
        ]
        
        y_offset = 10
        for line in info_lines:
            if line:
                text = self.font_small.render(line, True, (255, 255, 255))
                text_rect = text.get_rect()
                text_rect.topleft = (10, y_offset)
                
                # 简化的背景
                bg_rect = text_rect.inflate(2, 2)
                pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)
                pygame.draw.rect(self.screen, (255, 255, 255), bg_rect, 1)
                
                self.screen.blit(text, text_rect)
            
            y_offset += 20
    
    def update(self):
        """更新仿真状态 - 优化版"""
        if not self.is_paused:
            # 更新仿真时间
            self.stats['simulation_time'] += (1.0 / self.fps) * self.simulation_speed
            
            # 更新智能体
            for agent in self.agents.values():
                agent.update(self.tasks, self.charging_stations)
            
            # 更新统计信息
            self.stats['completed_tasks'] = sum(
                1 for task in self.tasks.values() if task['status'] == 'completed'
            )
    
    def draw(self):
        """绘制整个场景 - 优化版"""
        # 清屏
        self.screen.fill((0, 0, 0))
        
        # 绘制各个元素
        self._draw_background()
        self._draw_static_elements()
        self._draw_trajectories()
        self._draw_tasks()
        self._draw_agents()
        self._draw_ui()
        
        # 更新显示
        pygame.display.flip()
    
    def handle_events(self):
        """处理事件 - 优化版"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
                    print(f"{'⏸️ 暂停' if self.is_paused else '▶️ 继续'}")
                elif event.key == pygame.K_t:
                    self.show_trajectories = not self.show_trajectories
                    print(f"轨迹显示: {'开' if self.show_trajectories else '关'}")
                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info
                    print(f"信息显示: {'开' if self.show_info else '关'}")
        
        return True
    
    def run(self):
        """运行可视化主循环 - 优化版"""
        print("🎮 启动优化版Pygame可视化...")
        print(f"   性能设置: {self.fps}fps, {self.simulation_speed}x速度")
        print(f"   智能体: {len(self.agents)}, 任务: {len(self.tasks)}, 充电桩: {len(self.charging_stations)}")
        
        self.is_running = True
        frame_count = 0
        last_performance_check = time.time()
        
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
                
                # 性能监控
                frame_count += 1
                current_time = time.time()
                
                if current_time - last_performance_check >= 5.0:  # 每5秒一次
                    fps = self.clock.get_fps()
                    if fps < self.fps * 0.8:  # 如果FPS下降20%以上
                        if not self.performance_warning:
                            print(f"⚠️ 性能警告: FPS={fps:.1f}, 建议降低分辨率或关闭轨迹显示")
                            self.performance_warning = True
                    else:
                        self.performance_warning = False
                    
                    print(f"📊 性能监控: 第{frame_count}帧, FPS={fps:.1f}")
                    last_performance_check = current_time
                
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断")
        except Exception as e:
            print(f"❌ 运行时错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            pygame.quit()
            print("✅ 优化版Pygame已退出")