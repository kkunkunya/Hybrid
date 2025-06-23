"""
能源感知任务分配器
基于改进匈牙利算法的多目标优化任务分配系统
"""
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import copy
from scipy.optimize import linear_sum_assignment
import math

from .base_scheduler import BaseScheduler
from ..utils.energy import EnergyCalculator, AgentType


@dataclass
class TaskAssignment:
    """任务分配结果"""
    uav_id: str
    task_ids: List[int]
    total_energy_cost: float
    total_time_cost: float
    total_distance: float
    feasibility_score: float


@dataclass
class CostWeights:
    """成本权重配置"""
    distance: float = 0.3
    energy: float = 0.4
    time: float = 0.2
    risk: float = 0.1


class EnergyAwareScheduler(BaseScheduler):
    """能源感知任务分配器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化能源感知调度器
        
        Args:
            config: 配置参数字典
        """
        super().__init__(config)
        
        # 获取能源感知分配器配置
        allocator_config = config.get('energy_aware_allocator', {})
        
        # 成本权重配置
        cost_weights = allocator_config.get('cost_weights', {})
        self.cost_weights = CostWeights(
            distance=cost_weights.get('distance', 0.3),
            energy=cost_weights.get('energy', 0.4),
            time=cost_weights.get('time', 0.2),
            risk=cost_weights.get('risk', 0.1)
        )
        
        # 安全参数
        self.safety_margin = allocator_config.get('safety_margin', 0.25)
        self.max_tasks_per_uav = allocator_config.get('max_tasks_per_uav', 15)  # 增加到15个任务
        self.energy_threshold = allocator_config.get('energy_threshold', 0.10)  # 降低到10%以执行更多任务
        self.min_energy_reserve = allocator_config.get('min_energy_reserve', 0.1)
        
        # 自适应权重参数
        adaptive_config = allocator_config.get('adaptive_weights', {})
        self.adaptive_weights_enabled = adaptive_config.get('enable', True)
        self.adjustment_factor = adaptive_config.get('adjustment_factor', 0.1)
        self.load_threshold = adaptive_config.get('load_threshold', 0.8)
        self.energy_crisis_threshold = adaptive_config.get('energy_crisis_threshold', 0.2)
        
        # 工具类
        self.energy_calculator = EnergyCalculator()
        
        # 统计信息
        self.allocation_history = []
        self.performance_metrics = {
            'total_allocations': 0,
            'successful_allocations': 0,
            'average_energy_efficiency': 0.0,
            'average_completion_time': 0.0
        }
    
    def plan(self, 
             agents_state: Dict[str, Dict[str, Any]], 
             tasks: List[Dict[str, Any]], 
             environment_state: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        制定能源感知的任务分配方案
        
        Args:
            agents_state: 智能体状态字典
            tasks: 任务列表
            environment_state: 环境状态
            
        Returns:
            任务分配结果: {agent_id: [task_id_list]}
        """
        if not tasks:
            return {agent_id: [] for agent_id in agents_state.keys()}
        
        # 过滤出UAV智能体（只有UAV执行巡检任务）
        uav_agents = {aid: state for aid, state in agents_state.items() 
                     if state.get('type', 'uav') == 'uav'}
        
        if not uav_agents:
            print("警告：没有可用的UAV智能体执行任务")
            return {agent_id: [] for agent_id in agents_state.keys()}
        
        # 执行分配流程
        assignment_result = self._execute_allocation_pipeline(
            uav_agents, tasks, environment_state
        )
        
        # 构建完整的分配结果（包括USV）
        full_assignment = {}
        for agent_id in agents_state.keys():
            if agent_id in assignment_result:
                full_assignment[agent_id] = assignment_result[agent_id]
            else:
                full_assignment[agent_id] = []  # USV或未分配任务的UAV
        
        # 更新统计信息
        self._update_performance_metrics(assignment_result, uav_agents, tasks)
        
        return full_assignment
    
    def _execute_allocation_pipeline(self, 
                                   uav_agents: Dict[str, Dict[str, Any]], 
                                   tasks: List[Dict[str, Any]], 
                                   environment_state: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        执行分配流程
        
        Args:
            uav_agents: UAV智能体状态
            tasks: 任务列表
            environment_state: 环境状态
            
        Returns:
            UAV任务分配结果
        """
        print(f"开始能源感知任务分配：{len(uav_agents)}个UAV，{len(tasks)}个任务")
        
        # 1. 预处理：筛选可用UAV和任务
        available_uavs = self._filter_available_uavs(uav_agents)
        valid_tasks = self._validate_tasks(tasks)
        
        if not available_uavs:
            print("警告：没有可用的UAV执行任务")
            return {}
        
        print(f"可用UAV: {len(available_uavs)}个，有效任务: {len(valid_tasks)}个")
        
        # 2. 自适应权重调整
        if self.adaptive_weights_enabled:
            self._adapt_weights(available_uavs, valid_tasks, environment_state)
        
        # 3. 构建成本矩阵
        cost_matrix = self._build_enhanced_cost_matrix(
            available_uavs, valid_tasks, environment_state
        )
        
        # 4. 执行改进的匈牙利算法求解
        assignment = self._solve_constrained_assignment(
            cost_matrix, available_uavs, valid_tasks, environment_state
        )
        
        # 5. 验证和调整分配方案
        validated_assignment = self._validate_and_adjust_assignment(
            assignment, available_uavs, valid_tasks, environment_state
        )
        
        print(f"分配完成：{len(validated_assignment)}个UAV获得任务")
        # 确保至少有一些任务被分配
        total_assigned = sum(len(tasks) for tasks in validated_assignment.values())
        total_available_tasks = len(tasks)
        
        if total_assigned == 0 and total_available_tasks > 0:
            print("⚠️ 警告：没有任何任务通过验证，尝试强制分配")
            
            # 为每个有足够电量的UAV至少分配一个任务
            for uav_id, uav_state in uav_agents.items():
                if uav_id not in validated_assignment:
                    validated_assignment[uav_id] = []
                
                energy_ratio = uav_state['energy'] / uav_state['max_energy']
                if energy_ratio > 0.15 and len(validated_assignment[uav_id]) == 0:
                    # 找到最近的未分配任务
                    uav_pos = uav_state['position']
                    min_dist = float('inf')
                    nearest_task_id = None
                    
                    for i, task in enumerate(tasks):
                        if any(i in task_ids for task_ids in validated_assignment.values()):
                            continue  # 跳过已分配的任务
                        
                        task_pos = task['position']
                        dist = math.sqrt((task_pos[0] - uav_pos[0])**2 + 
                                       (task_pos[1] - uav_pos[1])**2)
                        
                        if dist < min_dist:
                            min_dist = dist
                            nearest_task_id = i
                    
                    if nearest_task_id is not None:
                        validated_assignment[uav_id] = [nearest_task_id]
                        print(f"  强制为 {uav_id} 分配任务 {nearest_task_id}")
        
        return validated_assignment
    
    def _filter_available_uavs(self, uav_agents: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """筛选可用的UAV智能体"""
        available_uavs = {}
        
        for uav_id, state in uav_agents.items():
            # 检查UAV状态
            if state.get('status', 'idle') in ['idle', 'available']:
                # 检查电量是否足够
                current_energy = state.get('energy', 0)
                max_energy = state.get('max_energy', 300.0)
                energy_ratio = current_energy / max_energy if max_energy > 0 else 0
                
                if energy_ratio > self.energy_threshold:
                    available_uavs[uav_id] = state
                else:
                    print(f"UAV {uav_id} 电量过低 ({energy_ratio:.1%})，暂时不分配任务")
        
        return available_uavs
    
    def _validate_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证和预处理任务列表"""
        valid_tasks = []
        
        for i, task in enumerate(tasks):
            # 确保任务有必要的字段
            if 'position' in task:
                task_copy = task.copy()
                # 添加默认值
                task_copy.setdefault('task_id', i)
                task_copy.setdefault('priority', 1.0)
                task_copy.setdefault('estimated_duration', 60.0)
                task_copy.setdefault('energy_requirement', 10.0)
                
                valid_tasks.append(task_copy)
        
        return valid_tasks
    
    def _adapt_weights(self, 
                      uav_agents: Dict[str, Dict[str, Any]], 
                      tasks: List[Dict[str, Any]], 
                      environment_state: Dict[str, Any]):
        """自适应权重调整"""
        
        # 计算系统负载
        system_load = len(tasks) / max(len(uav_agents), 1)
        
        # 计算平均电量水平
        total_energy_ratio = 0
        for uav_state in uav_agents.values():
            current_energy = uav_state.get('energy', 0)
            max_energy = uav_state.get('max_energy', 300.0)
            total_energy_ratio += current_energy / max_energy if max_energy > 0 else 0
        
        avg_energy_ratio = total_energy_ratio / len(uav_agents) if uav_agents else 0
        
        # 权重调整逻辑
        adjustment = self.adjustment_factor
        old_weights = {
            'distance': self.cost_weights.distance,
            'energy': self.cost_weights.energy,
            'time': self.cost_weights.time,
            'risk': self.cost_weights.risk
        }
        
        # 添加权重调整历史记录，避免频繁调整
        if not hasattr(self, '_last_adjustment_time'):
            self._last_adjustment_time = 0
        
        import time
        current_time = time.time()
        
        # 限制调整频率：至少5秒间隔
        if system_load > self.load_threshold and (current_time - self._last_adjustment_time) > 5.0:
            # 高负载：优先时间效率，但限制调整幅度
            max_adjustment = 0.05  # 最大单次调整幅度
            actual_adjustment = min(adjustment, max_adjustment)
            
            old_time_weight = self.cost_weights.time
            self.cost_weights.time += actual_adjustment
            self.cost_weights.distance -= actual_adjustment * 0.5
            
            # 限制权重范围
            self.cost_weights.time = min(0.8, self.cost_weights.time)  # 时间权重不超过80%
            self.cost_weights.distance = max(0.1, self.cost_weights.distance)  # 距离权重不低于10%
            
            self._last_adjustment_time = current_time
            print(f"🔍 高负载调整 (负载:{system_load:.2f}): 时间权重 {old_time_weight:.3f}→{self.cost_weights.time:.3f}")
        elif system_load > self.load_threshold:
            print(f"⏰ 跳过权重调整 (距离上次调整不足5秒)")
        
        if avg_energy_ratio < self.energy_crisis_threshold:
            # 能源危机：优先节能
            self.cost_weights.energy += adjustment
            self.cost_weights.distance -= adjustment * 0.5
            print(f"能源危机调整：提高能耗权重至 {self.cost_weights.energy:.2f}")
        
        # 检查权重是否超出合理范围，如果是则重置
        max_reasonable_weight = 2.0  # 任何单一权重不应超过2.0
        if (self.cost_weights.time > max_reasonable_weight or 
            self.cost_weights.energy > max_reasonable_weight or
            self.cost_weights.distance < 0 or 
            self.cost_weights.risk < 0):
            print(f"⚠️ 权重超出合理范围，重置权重")
            print(f"  当前权重: 距离={self.cost_weights.distance:.3f}, 能耗={self.cost_weights.energy:.3f}, "
                  f"时间={self.cost_weights.time:.3f}, 风险={self.cost_weights.risk:.3f}")
            
            # 重置为默认权重
            self.cost_weights.distance = 0.3
            self.cost_weights.energy = 0.4
            self.cost_weights.time = 0.2
            self.cost_weights.risk = 0.1
            print(f"  重置后权重: 距离={self.cost_weights.distance:.3f}, 能耗={self.cost_weights.energy:.3f}, "
                  f"时间={self.cost_weights.time:.3f}, 风险={self.cost_weights.risk:.3f}")
        
        # 检查权重是否超出合理范围，如果是则重置
        max_reasonable_weight = 2.0  # 任何单一权重不应超过2.0
        if (self.cost_weights.time > max_reasonable_weight or 
            self.cost_weights.energy > max_reasonable_weight or
            self.cost_weights.distance < 0 or 
            self.cost_weights.risk < 0):
            print(f"⚠️ 权重超出合理范围，重置权重")
            print(f"  当前权重: 距离={self.cost_weights.distance:.3f}, 能耗={self.cost_weights.energy:.3f}, "
                  f"时间={self.cost_weights.time:.3f}, 风险={self.cost_weights.risk:.3f}")
            
            # 重置为默认权重
            self.cost_weights.distance = 0.3
            self.cost_weights.energy = 0.4
            self.cost_weights.time = 0.2
            self.cost_weights.risk = 0.1
            print(f"  重置后权重: 距离={self.cost_weights.distance:.3f}, 能耗={self.cost_weights.energy:.3f}, "
                  f"时间={self.cost_weights.time:.3f}, 风险={self.cost_weights.risk:.3f}")
        
        # 确保权重和为1
        total_weight = (self.cost_weights.distance + self.cost_weights.energy + 
                       self.cost_weights.time + self.cost_weights.risk)
        if total_weight > 0:
            self.cost_weights.distance /= total_weight
            self.cost_weights.energy /= total_weight
            self.cost_weights.time /= total_weight
            self.cost_weights.risk /= total_weight
    
    def _build_enhanced_cost_matrix(self, 
                                  uav_agents: Dict[str, Dict[str, Any]], 
                                  tasks: List[Dict[str, Any]], 
                                  environment_state: Dict[str, Any]) -> np.ndarray:
        """构建增强型成本矩阵"""
        
        uav_list = list(uav_agents.keys())
        n_uavs = len(uav_list)
        n_tasks = len(tasks)
        
        # 创建成本矩阵
        cost_matrix = np.full((n_uavs, n_tasks), float('inf'))
        
        for i, uav_id in enumerate(uav_list):
            uav_state = uav_agents[uav_id]
            uav_pos = uav_state['position']
            
            for j, task in enumerate(tasks):
                task_pos = task['position']
                
                # 计算各项成本
                distance_cost = self._calculate_distance_cost(uav_pos, task_pos)
                energy_cost = self._calculate_energy_cost(uav_state, task, environment_state)
                time_cost = self._calculate_time_cost(uav_state, task, environment_state)
                risk_cost = self._calculate_risk_cost(task, environment_state)
                
                # 检查基本可行性
                if self._check_basic_feasibility(uav_state, task):
                    # 综合成本计算
                    total_cost = (self.cost_weights.distance * distance_cost +
                                 self.cost_weights.energy * energy_cost +
                                 self.cost_weights.time * time_cost +
                                 self.cost_weights.risk * risk_cost)
                    
                    cost_matrix[i, j] = total_cost
                # 否则保持inf，表示不可行
        
        return cost_matrix
    
    def _calculate_distance_cost(self, uav_pos: Tuple[float, float], task_pos: Tuple[float, float]) -> float:
        """计算距离成本"""
        distance = math.sqrt((task_pos[0] - uav_pos[0])**2 + (task_pos[1] - uav_pos[1])**2)
        return distance / 1000.0  # 归一化到km
    
    def _calculate_energy_cost(self, 
                             uav_state: Dict[str, Any], 
                             task: Dict[str, Any], 
                             environment_state: Dict[str, Any]) -> float:
        """计算能耗成本"""
        
        uav_pos = uav_state['position']
        task_pos = task['position']
        
        # 移动到任务点的能耗
        distance = math.sqrt((task_pos[0] - uav_pos[0])**2 + (task_pos[1] - uav_pos[1])**2)
        move_energy, _ = self.energy_calculator.calculate_movement_energy(
            AgentType.UAV, distance, 10.0  # 假设10m/s巡航速度
        )
        
        # 任务执行能耗
        task_energy = self.energy_calculator.calculate_task_energy(
            AgentType.UAV, 
            task.get('estimated_duration', 60.0),
            task.get('intensity', 1.0)
        )
        
        total_energy = move_energy + task_energy
        
        # 归一化：除以UAV最大电池容量
        max_energy = uav_state.get('max_energy', 300.0)
        return total_energy / max_energy if max_energy > 0 else total_energy / 300.0
    
    def _calculate_time_cost(self, 
                           uav_state: Dict[str, Any], 
                           task: Dict[str, Any], 
                           environment_state: Dict[str, Any]) -> float:
        """计算时间成本"""
        
        uav_pos = uav_state['position']
        task_pos = task['position']
        
        # 移动时间
        distance = math.sqrt((task_pos[0] - uav_pos[0])**2 + (task_pos[1] - uav_pos[1])**2)
        travel_time = distance / 10.0  # 假设10m/s速度
        
        # 任务执行时间
        task_time = task.get('estimated_duration', 60.0)
        
        total_time = travel_time + task_time
        
        # 归一化：除以1小时
        return total_time / 3600.0
    
    def _calculate_risk_cost(self, task: Dict[str, Any], environment_state: Dict[str, Any]) -> float:
        """计算风险成本"""
        
        # 基础风险评估
        priority = task.get('priority', 1.0)
        
        # 任务优先级越高，风险成本越低（越重要的任务优先执行）
        risk_cost = max(0.1, 2.0 - priority)
        
        # 环境风险因子
        weather = environment_state.get('weather', 'clear')
        if weather != 'clear':
            risk_cost *= 1.5  # 恶劣天气增加风险
        
        return risk_cost / 2.0  # 归一化
    
    def _check_basic_feasibility(self, uav_state: Dict[str, Any], task: Dict[str, Any]) -> bool:
        """检查UAV执行任务的基本可行性"""
        
        current_energy = uav_state.get('energy', 0)
        max_energy = uav_state.get('max_energy', 300.0)
        
        # 计算任务能耗需求
        uav_pos = uav_state['position']
        task_pos = task['position']
        
        distance = math.sqrt((task_pos[0] - uav_pos[0])**2 + (task_pos[1] - uav_pos[1])**2)
        move_energy, _ = self.energy_calculator.calculate_movement_energy(
            AgentType.UAV, distance, 10.0
        )
        
        task_energy = self.energy_calculator.calculate_task_energy(
            AgentType.UAV, task.get('estimated_duration', 60.0)
        )
        
        # 返回基地的能耗（简化计算）
        # 如果UAV电量高于30%，允许执行任务
        if current_energy / max_energy > 0.3:
            # 只考虑去程和任务执行的能耗
            immediate_energy_needed = (move_energy + task_energy) * (1 + self.safety_margin * 0.5)
            return current_energy >= immediate_energy_needed
        else:
            # 低电量时考虑返回基地的能耗
            return_energy = move_energy * 0.5
            total_energy_needed = (move_energy + task_energy + return_energy) * (1 + self.safety_margin)
            return current_energy >= total_energy_needed
    
    def _cluster_tasks_by_region(self, tasks: List[Dict[str, Any]], n_clusters: int) -> List[List[int]]:
        """将任务按区域聚类"""
        if len(tasks) <= n_clusters:
            return [[i] for i in range(len(tasks))]
        
        # 简单的K-means聚类
        import random
        
        # 初始化聚类中心
        centers = []
        task_positions = [task['position'] for task in tasks]
        
        # 随机选择初始中心
        center_indices = random.sample(range(len(tasks)), min(n_clusters, len(tasks)))
        centers = [task_positions[i] for i in center_indices]
        
        # 迭代优化
        for _ in range(10):  # 最多10次迭代
            # 分配任务到最近的中心
            clusters = [[] for _ in range(len(centers))]
            
            for task_idx, pos in enumerate(task_positions):
                min_dist = float('inf')
                best_cluster = 0
                
                for c_idx, center in enumerate(centers):
                    dist = math.sqrt((pos[0] - center[0])**2 + (pos[1] - center[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = c_idx
                
                clusters[best_cluster].append(task_idx)
            
            # 更新中心
            new_centers = []
            for cluster in clusters:
                if cluster:
                    avg_x = sum(task_positions[idx][0] for idx in cluster) / len(cluster)
                    avg_y = sum(task_positions[idx][1] for idx in cluster) / len(cluster)
                    new_centers.append((avg_x, avg_y))
                else:
                    # 空聚类，保持原中心
                    new_centers.append(centers[len(new_centers)])
            
            centers = new_centers
        
        # 移除空聚类
        clusters = [c for c in clusters if c]
        
        print(f"  任务聚类完成: {len(tasks)}个任务分为{len(clusters)}个区域")
        for i, cluster in enumerate(clusters):
            print(f"    区域{i+1}: {len(cluster)}个任务")
        
        return clusters
    
    def _optimize_task_sequence(self, task_indices: List[int], tasks: List[Dict[str, Any]], 
                               start_pos: Tuple[float, float]) -> List[int]:
        """优化任务执行序列，使用贪心最近邻算法"""
        if len(task_indices) <= 1:
            return task_indices
        
        optimized = []
        remaining = set(task_indices)
        current_pos = start_pos
        
        while remaining:
            # 找到最近的任务
            min_dist = float('inf')
            nearest_task = None
            
            for task_idx in remaining:
                task_pos = tasks[task_idx]['position']
                dist = math.sqrt((task_pos[0] - current_pos[0])**2 + 
                               (task_pos[1] - current_pos[1])**2)
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_task = task_idx
            
            if nearest_task is not None:
                optimized.append(nearest_task)
                remaining.remove(nearest_task)
                current_pos = tasks[nearest_task]['position']
        
        return optimized
    
    def _solve_constrained_assignment(self, 
                                    cost_matrix: np.ndarray,
                                    uav_agents: Dict[str, Dict[str, Any]], 
                                    tasks: List[Dict[str, Any]], 
                                    environment_state: Dict[str, Any]) -> Dict[str, List[int]]:
        """使用约束匈牙利算法求解任务分配"""
        
        n_uavs, n_tasks = cost_matrix.shape
        uav_list = list(uav_agents.keys())
        
        assignment = {uav_id: [] for uav_id in uav_list}
        
        if n_tasks == 0:
            return assignment
        
        # 处理UAV数量少于任务数量的情况
        if n_uavs < n_tasks:
            # 扩展成本矩阵，添加虚拟UAV
            extended_matrix = np.full((n_tasks, n_tasks), cost_matrix.max() * 2)
            extended_matrix[:n_uavs, :] = cost_matrix
            cost_matrix = extended_matrix
        
        # 使用改进的多任务分配算法
        try:
            # 第一轮：使用匈牙利算法分配初始任务
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # 记录已分配的任务
            assigned_tasks = set()
            
            # 构建初始分配
            for row, col in zip(row_indices, col_indices):
                if row < len(uav_list) and cost_matrix[row, col] != float('inf'):
                    uav_id = uav_list[row]
                    assignment[uav_id].append(col)
                    assigned_tasks.add(col)
            
            # 第二轮：基于区域聚类的任务分配
            remaining_tasks = [i for i in range(n_tasks) if i not in assigned_tasks]
            
            if remaining_tasks:
                print(f"  第二轮分配: {len(remaining_tasks)}个剩余任务")
                
                # 对剩余任务进行区域聚类
                n_available_uavs = sum(1 for uav_id in uav_list 
                                     if len(assignment[uav_id]) < self.max_tasks_per_uav)
                
                if n_available_uavs > 0:
                    # 获取剩余任务对象
                    remaining_task_objects = [tasks[i] for i in remaining_tasks]
                    
                    # 聚类
                    task_clusters = self._cluster_tasks_by_region(
                        remaining_task_objects, 
                        min(n_available_uavs, len(remaining_tasks))
                    )
                
                # 按UAV剩余能量排序
                uav_energy_list = []
                for uav_id, uav_state in uav_agents.items():
                    current_energy = uav_state.get('energy', 0)
                    # 计算已分配任务的能量消耗
                    allocated_energy = 0
                    for task_idx in assignment[uav_id]:
                        task = tasks[task_idx]
                        allocated_energy += self._estimate_task_energy(uav_state, task)
                    
                    remaining_energy = current_energy - allocated_energy
                    uav_energy_list.append((uav_id, remaining_energy))
                
                # 按剩余能量降序排序
                uav_energy_list.sort(key=lambda x: x[1], reverse=True)
                
                # 贪心分配剩余任务
                for task_idx in remaining_tasks:
                    task = tasks[task_idx]
                    best_uav = None
                    best_cost = float('inf')
                    
                    for uav_id, remaining_energy in uav_energy_list:
                        # 检查任务数限制
                        if len(assignment[uav_id]) >= self.max_tasks_per_uav:
                            continue
                        
                        # 检查能量约束
                        task_energy = self._estimate_task_energy(uav_agents[uav_id], task)
                        if remaining_energy < task_energy * (1 + self.safety_margin):
                            continue
                        
                        # 计算成本
                        uav_idx = uav_list.index(uav_id)
                        if cost_matrix[uav_idx, task_idx] < best_cost:
                            best_cost = cost_matrix[uav_idx, task_idx]
                            best_uav = (uav_id, uav_idx)
                    
                    # 分配任务
                    if best_uav:
                        uav_id, uav_idx = best_uav
                        assignment[uav_id].append(task_idx)
                        # 更新剩余能量
                        for i, (uid, energy) in enumerate(uav_energy_list):
                            if uid == uav_id:
                                task_energy = self._estimate_task_energy(uav_agents[uav_id], task)
                                uav_energy_list[i] = (uid, energy - task_energy)
                                break
                        
                print(f"  多任务分配完成: {sum(len(tasks) for tasks in assignment.values())}个任务")
        
        except Exception as e:
            print(f"多任务分配算法失败: {e}")
            # 回退到贪心分配
            assignment = self._fallback_greedy_assignment(uav_agents, tasks, environment_state)
        
        return assignment
    
    def _fallback_greedy_assignment(self, 
                                  uav_agents: Dict[str, Dict[str, Any]], 
                                  tasks: List[Dict[str, Any]], 
                                  environment_state: Dict[str, Any]) -> Dict[str, List[int]]:
        """贪心分配算法作为备选方案"""
        
        assignment = {uav_id: [] for uav_id in uav_agents.keys()}
        assigned_tasks = set()
        
        # 按任务优先级排序
        sorted_tasks = sorted(enumerate(tasks), key=lambda x: x[1].get('priority', 1.0), reverse=True)
        
        for task_idx, task in sorted_tasks:
            if task_idx in assigned_tasks:
                continue
            
            best_uav = None
            best_cost = float('inf')
            
            for uav_id, uav_state in uav_agents.items():
                # 检查UAV是否已经有太多任务
                if len(assignment[uav_id]) >= self.max_tasks_per_uav:
                    continue
                
                # 检查可行性
                if not self._check_basic_feasibility(uav_state, task):
                    continue
                
                # 计算成本
                distance_cost = self._calculate_distance_cost(uav_state['position'], task['position'])
                energy_cost = self._calculate_energy_cost(uav_state, task, environment_state)
                
                total_cost = distance_cost + energy_cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_uav = uav_id
            
            if best_uav:
                assignment[best_uav].append(task_idx)
                assigned_tasks.add(task_idx)
        
        return assignment
    
    def _validate_and_adjust_assignment(self, 
                                      assignment: Dict[str, List[int]],
                                      uav_agents: Dict[str, Dict[str, Any]], 
                                      tasks: List[Dict[str, Any]], 
                                      environment_state: Dict[str, Any]) -> Dict[str, List[int]]:
        """验证和调整分配方案"""
        
        validated_assignment = {}
        
        for uav_id, task_ids in assignment.items():
            if not task_ids:
                validated_assignment[uav_id] = []
                continue
            
            uav_state = uav_agents[uav_id]
            assigned_tasks = [tasks[tid] for tid in task_ids]
            
            # 验证总能耗约束
            if self._validate_total_energy_constraint(uav_state, assigned_tasks):
                # 优化任务执行序列
                optimized_sequence = self._optimize_task_sequence(
                    task_ids, tasks, uav_state['position']
                )
                
                # 再次验证优化后的序列
                optimized_tasks = [tasks[tid] for tid in optimized_sequence]
                if self._validate_total_energy_constraint(uav_state, optimized_tasks):
                    validated_assignment[uav_id] = optimized_sequence
                    energy_pct = (uav_state['energy'] / uav_state['max_energy']) * 100
                    
                    # 计算优化后的总路径长度
                    total_distance = 0
                    current_pos = uav_state['position']
                    for tid in optimized_sequence:
                        task_pos = tasks[tid]['position']
                        total_distance += math.sqrt((task_pos[0] - current_pos[0])**2 + 
                                                  (task_pos[1] - current_pos[1])**2)
                        current_pos = task_pos
                    
                    print(f"UAV {uav_id} (电量{energy_pct:.1f}%): 分配 {len(optimized_sequence)} 个任务, "
                          f"总路径: {total_distance:.0f}m")
                    
                    if len(optimized_sequence) > 5:
                        print(f"  任务序列: {optimized_sequence[:3]}...{optimized_sequence[-2:]}")
                    else:
                        print(f"  任务序列: {optimized_sequence}")
                else:
                    # 优化后仍不可行，减少任务数
                    validated_assignment[uav_id] = optimized_sequence[:len(optimized_sequence)//2]
                    print(f"UAV {uav_id}: 优化后减少任务到 {len(validated_assignment[uav_id])} 个")
            else:
                # 尝试减少任务数量
                reduced_tasks = self._reduce_task_assignment(uav_state, assigned_tasks)
                validated_assignment[uav_id] = [tasks.index(task) for task in reduced_tasks 
                                               if task in tasks]
                print(f"UAV {uav_id}: 调整后分配 {len(validated_assignment[uav_id])} 个任务")
        
        # 确保至少有一些任务被分配
        total_assigned = sum(len(tasks) for tasks in validated_assignment.values())
        total_available_tasks = len(tasks)
        
        if total_assigned == 0 and total_available_tasks > 0:
            print("⚠️ 警告：没有任何任务通过验证，尝试强制分配")
            
            # 为每个有足够电量的UAV至少分配一个任务
            for uav_id, uav_state in uav_agents.items():
                if uav_id not in validated_assignment:
                    validated_assignment[uav_id] = []
                
                energy_ratio = uav_state['energy'] / uav_state['max_energy']
                if energy_ratio > 0.15 and len(validated_assignment[uav_id]) == 0:
                    # 找到最近的未分配任务
                    uav_pos = uav_state['position']
                    min_dist = float('inf')
                    nearest_task_id = None
                    
                    for i, task in enumerate(tasks):
                        if any(i in task_ids for task_ids in validated_assignment.values()):
                            continue  # 跳过已分配的任务
                        
                        task_pos = task['position']
                        dist = math.sqrt((task_pos[0] - uav_pos[0])**2 + 
                                       (task_pos[1] - uav_pos[1])**2)
                        
                        if dist < min_dist:
                            min_dist = dist
                            nearest_task_id = i
                    
                    if nearest_task_id is not None:
                        validated_assignment[uav_id] = [nearest_task_id]
                        print(f"  强制为 {uav_id} 分配任务 {nearest_task_id}")
        
        return validated_assignment
    
    def _validate_total_energy_constraint(self, 
                                        uav_state: Dict[str, Any], 
                                        assigned_tasks: List[Dict[str, Any]]) -> bool:
        """验证总能耗约束"""
        
        current_energy = uav_state.get('energy', 0)
        current_pos = uav_state['position']
        
        total_energy_needed = 0
        
        for task in assigned_tasks:
            # 移动能耗
            distance = math.sqrt((task['position'][0] - current_pos[0])**2 + 
                               (task['position'][1] - current_pos[1])**2)
            move_energy, _ = self.energy_calculator.calculate_movement_energy(
                AgentType.UAV, distance, 10.0
            )
            
            # 任务执行能耗
            task_energy = self.energy_calculator.calculate_task_energy(
                AgentType.UAV, task.get('estimated_duration', 60.0)
            )
            
            total_energy_needed += move_energy + task_energy
            current_pos = task['position']
        
        # 返回基地能耗
        home_distance = math.sqrt((current_pos[0] - uav_state['position'][0])**2 + 
                                (current_pos[1] - uav_state['position'][1])**2)
        return_energy, _ = self.energy_calculator.calculate_movement_energy(
            AgentType.UAV, home_distance, 10.0
        )
        
        total_energy_needed += return_energy
        
        # 添加安全余量
        required_energy = total_energy_needed * (1 + self.safety_margin)
        
        return current_energy >= required_energy
    
    def _reduce_task_assignment(self, 
                              uav_state: Dict[str, Any], 
                              assigned_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """减少任务分配以满足能耗约束"""
        
        # 按优先级排序
        sorted_tasks = sorted(assigned_tasks, key=lambda x: x.get('priority', 1.0), reverse=True)
        
        # 逐个添加任务，直到能耗约束被违反
        valid_tasks = []
        for task in sorted_tasks:
            test_tasks = valid_tasks + [task]
            if self._validate_total_energy_constraint(uav_state, test_tasks):
                valid_tasks.append(task)
            else:
                break
        
        return valid_tasks
    
    def _update_performance_metrics(self, 
                                  assignment: Dict[str, List[int]], 
                                  uav_agents: Dict[str, Dict[str, Any]], 
                                  tasks: List[Dict[str, Any]]):
        """更新性能指标"""
        
        self.performance_metrics['total_allocations'] += 1
        
        # 计算分配成功率
        total_tasks_assigned = sum(len(task_ids) for task_ids in assignment.values())
        if total_tasks_assigned > 0:
            self.performance_metrics['successful_allocations'] += 1
        
        # 记录分配历史
        allocation_record = {
            'timestamp': len(self.allocation_history),
            'uav_count': len(uav_agents),
            'task_count': len(tasks),
            'assigned_tasks': total_tasks_assigned,
            'assignment_details': copy.deepcopy(assignment)
        }
        
        self.allocation_history.append(allocation_record)
        
        # 保持历史记录不超过100条
        if len(self.allocation_history) > 100:
            self.allocation_history.pop(0)
    
    def evaluate(self, 
                 assignment: Dict[str, List[int]], 
                 agents_state: Dict[str, Dict[str, Any]], 
                 tasks: List[Dict[str, Any]]) -> float:
        """
        评估分配方案的质量
        
        Args:
            assignment: 任务分配方案
            agents_state: 智能体状态
            tasks: 任务列表
            
        Returns:
            评估分数（越高越好，0-10分）
        """
        if not assignment or not tasks:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        # 过滤UAV
        uav_agents = {aid: state for aid, state in agents_state.items() 
                     if state.get('type', 'uav') == 'uav'}
        
        for uav_id, task_ids in assignment.items():
            if uav_id not in uav_agents or not task_ids:
                continue
            
            uav_state = uav_agents[uav_id]
            assigned_tasks = [tasks[tid] for tid in task_ids if tid < len(tasks)]
            
            if not assigned_tasks:
                continue
            
            # 任务价值评分 (0-3分)
            task_value_score = min(3.0, sum(task.get('priority', 1.0) for task in assigned_tasks))
            
            # 能源效率评分 (0-3分)
            energy_efficiency = self._calculate_energy_efficiency_score(uav_state, assigned_tasks)
            
            # 负载均衡评分 (0-2分)
            load_balance_score = min(2.0, 2.0 * len(task_ids) / max(len(tasks) / len(uav_agents), 1))
            
            # 可行性评分 (0-2分)
            feasibility_score = 2.0 if self._validate_total_energy_constraint(uav_state, assigned_tasks) else 0.5
            
            # 加权总分
            uav_score = task_value_score + energy_efficiency + load_balance_score + feasibility_score
            total_score += uav_score
            total_weight += 1.0
        
        # 返回平均分
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_energy_efficiency_score(self, 
                                         uav_state: Dict[str, Any], 
                                         assigned_tasks: List[Dict[str, Any]]) -> float:
        """计算能源效率评分"""
        
        current_energy = uav_state.get('energy', 0)
        max_energy = uav_state.get('max_energy', 300.0)
        
        if not assigned_tasks:
            return 0.0
        
        # 计算能量利用率
        total_energy_needed = 0
        current_pos = uav_state['position']
        
        for task in assigned_tasks:
            distance = math.sqrt((task['position'][0] - current_pos[0])**2 + 
                               (task['position'][1] - current_pos[1])**2)
            move_energy, _ = self.energy_calculator.calculate_movement_energy(
                AgentType.UAV, distance, 10.0
            )
            task_energy = self.energy_calculator.calculate_task_energy(
                AgentType.UAV, task.get('estimated_duration', 60.0)
            )
            total_energy_needed += move_energy + task_energy
            current_pos = task['position']
        
        # 能效评分：任务价值 / 能耗比例
        energy_ratio = total_energy_needed / current_energy if current_energy > 0 else 1.0
        task_value = sum(task.get('priority', 1.0) for task in assigned_tasks)
        
        if energy_ratio <= 0.5:  # 能耗少于50%
            return 3.0
        elif energy_ratio <= 0.7:  # 能耗50-70%
            return 2.0 + (0.7 - energy_ratio) * 5  # 线性映射到2-3分
        elif energy_ratio <= 0.9:  # 能耗70-90%
            return 1.0 + (0.9 - energy_ratio) * 5  # 线性映射到1-2分
        else:  # 能耗超过90%
            return max(0.1, 1.0 - (energy_ratio - 0.9) * 10)
    
    
    def _calculate_remaining_energy_after_tasks(self, 
                                               uav_state: Dict[str, Any], 
                                               assigned_tasks: List[Dict[str, Any]]) -> float:
        """计算执行已分配任务后的剩余能量"""
        current_energy = uav_state.get('energy', 0)
        
        # 估算已分配任务的总能耗
        total_energy_cost = 0
        current_pos = uav_state['position']
        
        for task in assigned_tasks:
            # 移动到任务点的能耗
            distance = math.sqrt((task['position'][0] - current_pos[0])**2 + 
                               (task['position'][1] - current_pos[1])**2)
            move_energy, _ = self.energy_calculator.calculate_movement_energy(
                AgentType.UAV, distance, 10.0
            )
            
            # 任务执行能耗
            task_energy = self.energy_calculator.calculate_task_energy(
                AgentType.UAV, task.get('estimated_duration', 60.0)
            )
            
            total_energy_cost += move_energy + task_energy
            current_pos = task['position']
        
        return current_energy - total_energy_cost
    
    def _can_execute_additional_task(self, 
                                   uav_state: Dict[str, Any], 
                                   new_task: Dict[str, Any],
                                   current_tasks: List[int],
                                   all_tasks: List[Dict[str, Any]]) -> bool:
        """检查UAV是否能执行额外任务"""
        
        # 计算执行当前任务后的剩余能量
        assigned_task_data = [all_tasks[tid] for tid in current_tasks]
        remaining_energy = self._calculate_remaining_energy_after_tasks(
            uav_state, assigned_task_data
        )
        
        # 计算新任务的能耗需求
        last_pos = uav_state['position']
        if assigned_task_data:
            last_pos = assigned_task_data[-1]['position']
        
        distance = math.sqrt((new_task['position'][0] - last_pos[0])**2 + 
                           (new_task['position'][1] - last_pos[1])**2)
        move_energy, _ = self.energy_calculator.calculate_movement_energy(
            AgentType.UAV, distance, 10.0
        )
        
        task_energy = self.energy_calculator.calculate_task_energy(
            AgentType.UAV, new_task.get('estimated_duration', 60.0)
        )
        
        new_task_total_energy = move_energy + task_energy
        
        # 检查安全余量
        safety_energy = self.safety_margin * new_task_total_energy
        required_energy = new_task_total_energy + safety_energy
        
        return remaining_energy >= required_energy
    def get_allocation_statistics(self) -> Dict[str, Any]:
        """获取分配统计信息"""
        
        if self.performance_metrics['total_allocations'] == 0:
            return {"message": "暂无分配记录"}
        
        success_rate = (self.performance_metrics['successful_allocations'] / 
                       self.performance_metrics['total_allocations'])
        
        return {
            "总分配次数": self.performance_metrics['total_allocations'],
            "成功分配次数": self.performance_metrics['successful_allocations'],
            "成功率": f"{success_rate:.2%}",
            "当前权重配置": {
                "距离权重": self.cost_weights.distance,
                "能耗权重": self.cost_weights.energy,
                "时间权重": self.cost_weights.time,
                "风险权重": self.cost_weights.risk
            },
            "安全参数": {
                "安全余量": f"{self.safety_margin:.1%}",
                "能量阈值": f"{self.energy_threshold:.1%}",
                "最大任务数": self.max_tasks_per_uav
            }
        }


# 演示用法
if __name__ == "__main__":
    # 测试配置
    config = {
        'energy_aware_allocator': {
            'cost_weights': {'distance': 0.3, 'energy': 0.4, 'time': 0.2, 'risk': 0.1},
            'safety_margin': 0.25,
            'max_tasks_per_uav': 5,
            'energy_threshold': 0.15
        }
    }
    
    # 创建调度器
    scheduler = EnergyAwareScheduler(config)
    
    # 测试数据
    agents_state = {
        'uav1': {'position': (100, 100), 'energy': 250, 'max_energy': 300, 'type': 'uav', 'status': 'idle'},
        'uav2': {'position': (200, 200), 'energy': 200, 'max_energy': 300, 'type': 'uav', 'status': 'idle'},
        'uav3': {'position': (300, 300), 'energy': 180, 'max_energy': 300, 'type': 'uav', 'status': 'idle'},
        'usv1': {'position': (150, 150), 'energy': 800, 'max_energy': 1000, 'type': 'usv', 'status': 'idle'}
    }
    
    tasks = [
        {'task_id': 0, 'position': (150, 150), 'priority': 2.0, 'estimated_duration': 120, 'energy_requirement': 15.0},
        {'task_id': 1, 'position': (250, 250), 'priority': 1.5, 'estimated_duration': 90, 'energy_requirement': 12.0},
        {'task_id': 2, 'position': (350, 350), 'priority': 1.8, 'estimated_duration': 100, 'energy_requirement': 20.0},
        {'task_id': 3, 'position': (400, 200), 'priority': 1.2, 'estimated_duration': 80, 'energy_requirement': 10.0},
        {'task_id': 4, 'position': (100, 400), 'priority': 2.5, 'estimated_duration': 150, 'energy_requirement': 25.0}
    ]
    
    environment_state = {'weather': 'clear', 'visibility': 10000.0}
    
    print("=== 能源感知任务分配器测试 ===")
    
    # 执行任务分配
    assignment = scheduler.plan(agents_state, tasks, environment_state)
    print(f"\n任务分配结果:")
    for agent_id, task_ids in assignment.items():
        agent_type = agents_state[agent_id].get('type', 'unknown')
        if task_ids:
            print(f"  {agent_id} ({agent_type}): 任务 {task_ids}")
        else:
            print(f"  {agent_id} ({agent_type}): 无分配任务")
    
    # 评估分配质量
    score = scheduler.evaluate(assignment, agents_state, tasks)
    print(f"\n分配质量评分: {score:.2f}/10.0")
    
    # 显示统计信息
    stats = scheduler.get_allocation_statistics()
    print(f"\n分配统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")