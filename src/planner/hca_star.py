"""
HCA-A* 路径规划器
Hierarchical Cooperative A* - 代价感知路径规划（耗时 & 耗能双目标）
"""
import heapq
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from ..utils.energy import EnergyCalculator, AgentType
from .base_planner import BasePlanner


@dataclass
class GridNode:
    """网格节点"""
    x: int
    y: int
    g_cost: float = float('inf')  # 起点到当前节点的实际代价
    h_cost: float = 0.0  # 启发式代价（当前节点到终点）
    f_cost: float = float('inf')  # 总代价 f = g + h
    parent: Optional['GridNode'] = None
    energy_cost: float = float('inf')  # 能量代价
    time_cost: float = float('inf')  # 时间代价
    is_obstacle: bool = False
    
    def __lt__(self, other):
        """比较函数，用于优先队列"""
        return self.f_cost < other.f_cost
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Environment:
    """环境抽象类"""
    
    @abstractmethod
    def is_obstacle(self, x: float, y: float, agent_type: str) -> bool:
        """检查位置是否为障碍物"""
        pass
    
    @abstractmethod
    def get_map_bounds(self) -> Tuple[int, int, int, int]:
        """获取地图边界 (min_x, min_y, max_x, max_y)"""
        pass


class SimpleEnvironment(Environment):
    """简单环境实现"""
    
    def __init__(self, width: int, height: int, obstacles: List[Tuple[int, int, int, int]] = None):
        """
        初始化简单环境
        
        Args:
            width, height: 地图尺寸
            obstacles: 障碍物列表 [(xmin, ymin, xmax, ymax), ...]
        """
        self.width = width
        self.height = height
        self.obstacles = obstacles or []
        
    
    def is_obstacle(self, x: float, y: float, agent_type: str) -> bool:
        """检查是否为障碍物"""
        # UAV可以飞越大部分障碍物，USV不能上陆地
        if agent_type == 'uav':
            return False  # 简化：UAV可以飞越所有障碍物
        
        # USV需要避开陆地障碍物
        for xmin, ymin, xmax, ymax in self.obstacles:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return True
        return False
    
    def get_map_bounds(self) -> Tuple[int, int, int, int]:
        """获取地图边界"""
        return 0, 0, self.width, self.height


class HCAStarPlanner(BasePlanner):
    """HCA-A* 路径规划器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化HCA-A*规划器
        
        Args:
            config: 配置参数
        """
        super().__init__(config)
        self.energy_calculator = EnergyCalculator()
        
        # 规划参数
        self.grid_resolution = config.get('grid_resolution', 10.0)  # 网格分辨率(m)
        self.time_weight = config.get('time_weight', 1.0)  # 时间权重
        self.energy_weight = config.get('energy_weight', 1.0)  # 能量权重
        self.safety_margin = config.get('safety_margin', 1.2)  # 安全系数
        
        # 搜索参数
        self.max_iterations = config.get('max_iterations', 10000)
        self.neighbor_directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
    
    def plan_path(self, 
                  start_pos: Tuple[float, float], 
                  target_pos: Tuple[float, float],
                  agent_type: str,
                  environment: Environment,
                  constraints: Optional[Dict[str, Any]] = None) -> Tuple[List[Tuple[float, float]], float, float]:
        """
        规划从起点到终点的最优路径
        
        Args:
            start_pos: 起始位置 (x, y)
            target_pos: 目标位置 (x, y)  
            agent_type: 智能体类型 ('uav' 或 'usv')
            environment: 环境对象
            constraints: 额外约束条件
            
        Returns:
            路径点列表, 总时间成本, 总能量成本
        """
        constraints = constraints or {}
        
        # 转换为网格坐标
        start_grid = self._world_to_grid(start_pos)
        target_grid = self._world_to_grid(target_pos)
        
        # 获取地图边界
        min_x, min_y, max_x, max_y = environment.get_map_bounds()
        grid_bounds = (
            int(min_x / self.grid_resolution),
            int(min_y / self.grid_resolution),
            int(max_x / self.grid_resolution),
            int(max_y / self.grid_resolution)
        )
        
        # 执行A*搜索
        path_nodes = self._astar_search(start_grid, target_grid, agent_type, environment, grid_bounds)
        
        if not path_nodes:
            print(f"未找到从 {start_pos} 到 {target_pos} 的路径")
            return [], float('inf'), float('inf')
        
        # 转换回世界坐标并计算成本
        world_path = [self._grid_to_world((node.x, node.y)) for node in path_nodes]
        total_time, total_energy = self._calculate_path_cost(world_path, agent_type, environment)
        
        return world_path, total_time, total_energy
    
    def plan_multi_target(self, 
                         start_pos: Tuple[float, float],
                         targets: List[Tuple[float, float]], 
                         agent_type: str,
                         environment: Environment) -> Tuple[List[Tuple[float, float]], float, float]:
        """
        规划多目标访问路径（不优化顺序，按给定顺序访问）
        
        Args:
            start_pos: 起始位置
            targets: 目标位置列表
            agent_type: 智能体类型
            environment: 环境对象
            
        Returns:
            完整路径, 总时间, 总能量
        """
        if not targets:
            return [start_pos], 0.0, 0.0
        
        complete_path = [start_pos]
        total_time = 0.0
        total_energy = 0.0
        
        current_pos = start_pos
        
        # 依次规划到每个目标的路径
        for target in targets:
            segment_path, segment_time, segment_energy = self.plan_path(
                current_pos, target, agent_type, environment
            )
            
            if not segment_path or segment_time == float('inf'):
                print(f"无法到达目标 {target}")
                return [], float('inf'), float('inf')
            
            # 添加路径段（去除重复的起点）
            complete_path.extend(segment_path[1:])
            total_time += segment_time
            total_energy += segment_energy
            
            current_pos = target
        
        return complete_path, total_time, total_energy
    
    def _astar_search(self, 
                     start: Tuple[int, int], 
                     goal: Tuple[int, int],
                     agent_type: str,
                     environment: Environment,
                     bounds: Tuple[int, int, int, int]) -> List[GridNode]:
        """
        A*搜索算法实现
        
        Args:
            start: 起始网格坐标
            goal: 目标网格坐标
            agent_type: 智能体类型
            environment: 环境对象
            bounds: 地图边界
            
        Returns:
            路径节点列表
        """
        min_x, min_y, max_x, max_y = bounds
        
        # 初始化起始节点
        start_node = GridNode(start[0], start[1])
        start_node.g_cost = 0.0
        start_node.h_cost = self._heuristic_cost(start, goal)
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        start_node.energy_cost = 0.0
        start_node.time_cost = 0.0
        
        # 开放列表和关闭列表
        open_list = [start_node]
        closed_set: Set[Tuple[int, int]] = set()
        
        # 所有节点的最佳成本记录
        best_costs: Dict[Tuple[int, int], float] = {start: 0.0}
        
        iterations = 0
        
        while open_list and iterations < self.max_iterations:
            iterations += 1
            
            # 获取f值最小的节点
            current = heapq.heappop(open_list)
            current_pos = (current.x, current.y)
            
            # 到达目标
            if current_pos == goal:
                return self._reconstruct_path(current)
            
            # 加入关闭列表
            closed_set.add(current_pos)
            
            # 扩展邻居节点
            for dx, dy in self.neighbor_directions:
                neighbor_x = current.x + dx
                neighbor_y = current.y + dy
                neighbor_pos = (neighbor_x, neighbor_y)
                
                # 检查边界
                if not (min_x <= neighbor_x <= max_x and min_y <= neighbor_y <= max_y):
                    continue
                
                # 已在关闭列表中
                if neighbor_pos in closed_set:
                    continue
                
                # 检查障碍物
                world_pos = self._grid_to_world(neighbor_pos)
                if environment.is_obstacle(world_pos[0], world_pos[1], agent_type):
                    continue
                
                # 计算移动成本
                move_cost = self._calculate_move_cost(
                    self._grid_to_world(current_pos),
                    world_pos,
                    agent_type,
                    environment
                )
                
                if move_cost == float('inf'):
                    continue
                
                # 计算新的g值
                tentative_g = current.g_cost + move_cost
                
                # 检查是否找到更好的路径
                if neighbor_pos in best_costs and tentative_g >= best_costs[neighbor_pos]:
                    continue
                
                # 创建邻居节点
                neighbor = GridNode(neighbor_x, neighbor_y)
                neighbor.g_cost = tentative_g
                neighbor.h_cost = self._heuristic_cost(neighbor_pos, goal)
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                neighbor.parent = current
                
                # 更新最佳成本记录
                best_costs[neighbor_pos] = tentative_g
                
                # 加入开放列表
                heapq.heappush(open_list, neighbor)
        
        print(f"A*搜索失败，迭代 {iterations} 次")
        return []
    
    def _calculate_move_cost(self, 
                           from_pos: Tuple[float, float],
                           to_pos: Tuple[float, float],
                           agent_type: str,
                           environment: Environment) -> float:
        """
        计算移动成本（综合时间和能量）
        
        Args:
            from_pos: 起始位置
            to_pos: 目标位置
            agent_type: 智能体类型
            environment: 环境对象
            
        Returns:
            移动成本
        """
        # 计算距离
        distance = np.sqrt((to_pos[0] - from_pos[0])**2 + (to_pos[1] - from_pos[1])**2)
        
        if distance == 0:
            return 0.0
        
        # 选择合适的速度（经济巡航速度）
        agent_type_enum = AgentType.UAV if agent_type == 'uav' else AgentType.USV
        max_speed = self.energy_calculator.params[agent_type_enum].max_speed
        cruise_speed = max_speed * 0.7  # 经济巡航速度
        
        # 计算能量和时间消耗
        energy_cost, time_cost = self.energy_calculator.calculate_movement_energy(
            agent_type_enum, distance, cruise_speed
        )
        
        # 综合成本（加权）
        combined_cost = (self.time_weight * time_cost + 
                        self.energy_weight * energy_cost) * self.safety_margin
        
        return combined_cost
    
    def _heuristic_cost(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        启发式成本函数（欧几里得距离）
        
        Args:
            pos: 当前位置
            goal: 目标位置
            
        Returns:
            启发式成本
        """
        dx = abs(pos[0] - goal[0])
        dy = abs(pos[1] - goal[1])
        
        # 欧几里得距离，转换为世界坐标距离
        grid_distance = np.sqrt(dx*dx + dy*dy)
        world_distance = grid_distance * self.grid_resolution
        
        # 估算最小时间成本（假设以最大速度直线飞行）
        estimated_time = world_distance / 10.0  # 假设平均速度10m/s
        
        return estimated_time * self.time_weight
    
    def _reconstruct_path(self, node: GridNode) -> List[GridNode]:
        """
        重构路径
        
        Args:
            node: 目标节点
            
        Returns:
            路径节点列表
        """
        path = []
        current = node
        
        while current is not None:
            path.append(current)
            current = current.parent
        
        return path[::-1]  # 反转路径
    
    def _world_to_grid(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """世界坐标转网格坐标"""
        return (
            int(world_pos[0] / self.grid_resolution),
            int(world_pos[1] / self.grid_resolution)
        )
    
    def _grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """网格坐标转世界坐标"""
        return (
            grid_pos[0] * self.grid_resolution + self.grid_resolution / 2,
            grid_pos[1] * self.grid_resolution + self.grid_resolution / 2
        )
    
    def _calculate_path_cost(self, 
                           path: List[Tuple[float, float]], 
                           agent_type: str,
                           environment: Environment) -> Tuple[float, float]:
        """
        计算路径的总时间和能量成本
        
        Args:
            path: 路径点列表
            agent_type: 智能体类型
            environment: 环境对象
            
        Returns:
            (总时间, 总能量)
        """
        if len(path) < 2:
            return 0.0, 0.0
        
        total_time = 0.0
        total_energy = 0.0
        
        agent_type_enum = AgentType.UAV if agent_type == 'uav' else AgentType.USV
        max_speed = self.energy_calculator.params[agent_type_enum].max_speed
        cruise_speed = max_speed * 0.7
        
        for i in range(len(path) - 1):
            from_pos = path[i]
            to_pos = path[i + 1]
            
            distance = np.sqrt((to_pos[0] - from_pos[0])**2 + (to_pos[1] - from_pos[1])**2)
            
            if distance > 0:
                segment_energy, segment_time = self.energy_calculator.calculate_movement_energy(
                    agent_type_enum, distance, cruise_speed
                )
                
                total_time += segment_time
                total_energy += segment_energy
        
        return total_time, total_energy


# 演示用法
if __name__ == "__main__":
    # 创建简单环境
    environment = SimpleEnvironment(1000, 1000, [(200, 200, 300, 300)])  # 1km x 1km，有一个障碍物
    
    # 配置HCA-A*规划器
    config = {
        'grid_resolution': 20.0,  # 20m网格
        'time_weight': 1.0,
        'energy_weight': 0.5,
        'safety_margin': 1.2
    }
    
    planner = HCAStarPlanner(config)
    
    # 测试单点路径规划
    start = (50.0, 50.0)
    target = (800.0, 800.0)
    
    print("=== HCA-A* 路径规划测试 ===")
    path, time_cost, energy_cost = planner.plan_path(start, target, 'uav', environment)
    
    if path:
        print(f"路径规划成功！")
        print(f"路径点数: {len(path)}")
        print(f"总时间: {time_cost:.1f} 秒")
        print(f"总能量: {energy_cost:.2f} Wh")
        print(f"起点: {path[0]}")
        print(f"终点: {path[-1]}")
    else:
        print("路径规划失败！")
    
    # 测试多目标路径规划
    targets = [(200.0, 100.0), (500.0, 300.0), (800.0, 600.0)]
    multi_path, multi_time, multi_energy = planner.plan_multi_target(start, targets, 'uav', environment)
    
    print(f"\\n多目标路径规划:")
    print(f"总路径点数: {len(multi_path)}")
    print(f"总时间: {multi_time:.1f} 秒")
    print(f"总能量: {multi_energy:.2f} Wh")