"""
USV后勤智能调度器
实现USV支援UAV的智能调度，包括支援效益评估、电量管理和多USV协调
基于动态规划和贪心策略的组合算法
"""
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import heapq
import copy
import math

from .base_scheduler import BaseScheduler
from ..utils.energy import EnergyCalculator, AgentType


class USVStatus(Enum):
    """USV状态枚举"""
    IDLE = "idle"                    # 空闲待命
    MOVING_TO_SUPPORT = "moving"     # 正在前往支援
    SUPPORTING = "supporting"        # 正在充电支援
    RETURNING = "returning"          # 返回基地
    CHARGING = "charging"            # 在基地充电
    EMERGENCY = "emergency"          # 紧急状态
    PATROLLING = "patrolling"        # 巡逻状态


@dataclass
class SupportRequest:
    """UAV支援请求"""
    uav_id: str
    position: Tuple[float, float]
    current_energy: float
    energy_needed: float
    priority: float
    request_time: float
    deadline: float  # 支援截止时间
    at_risk_tasks: List[Dict[str, Any]]  # 处于风险的任务
    
    def __lt__(self, other):
        """用于优先队列排序"""
        # 优先级越高、deadline越近的请求优先
        return (self.priority / (self.deadline - self.request_time)) > \
               (other.priority / (other.deadline - other.request_time))


@dataclass
class SupportMission:
    """USV支援任务"""
    usv_id: str
    target_uav_id: str
    start_position: Tuple[float, float]
    target_position: Tuple[float, float]
    energy_to_transfer: float
    expected_arrival_time: float
    expected_completion_time: float
    mission_benefit: float
    mission_cost: float
    status: USVStatus = USVStatus.IDLE
    mission_type: str = "support"  # support或patrol


@dataclass
class USVState:
    """USV状态扩展"""
    usv_id: str
    position: Tuple[float, float]
    current_energy: float
    max_energy: float
    status: USVStatus
    current_mission: Optional[SupportMission] = None
    home_base: Tuple[float, float] = (0.0, 0.0)
    charging_capacity: float = 200.0  # 可用于支援的电量
    charging_rate: float = 50.0  # W
    supported_uavs: Set[str] = field(default_factory=set)  # 已支援的UAV集合
    mission_history: List[SupportMission] = field(default_factory=list)
    patrol_waypoints: List[Tuple[float, float]] = field(default_factory=list)  # 巡逻路径点
    current_patrol_index: int = 0  # 当前巡逻点索引


class USVLogisticsScheduler(BaseScheduler):
    """USV后勤智能调度器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化USV后勤调度器
        
        Args:
            config: 配置参数字典
        """
        super().__init__(config)
        
        # 获取USV后勤配置
        usv_config = config.get('usv_logistics', {})
        
        # 支援效益权重
        benefit_weights = usv_config.get('support_benefit_weights', {})
        self.benefit_weights = {
            'task_value': benefit_weights.get('task_value', 0.4),
            'energy_cost': benefit_weights.get('energy_cost', 0.3),
            'time_cost': benefit_weights.get('time_cost', 0.2),
            'distance_cost': benefit_weights.get('distance_cost', 0.1)
        }
        
        # 支援参数
        self.min_reserve_energy_ratio = usv_config.get('min_reserve_energy_ratio', 0.3)
        self.max_support_distance = usv_config.get('max_support_distance', 5000.0)
        self.charging_transfer_efficiency = usv_config.get('charging_transfer_efficiency', 0.85)
        self.support_response_time = usv_config.get('support_response_time', 300.0)
        
        # 协调参数
        self.max_concurrent_supports = usv_config.get('max_concurrent_supports', 2)
        self.coordination_radius = usv_config.get('coordination_radius', 2000.0)
        self.emergency_energy_threshold = usv_config.get('emergency_energy_threshold', 0.1)
        
        # 工具类
        self.energy_calculator = EnergyCalculator()
        
        # 内部状态
        self.usv_states: Dict[str, USVState] = {}
        self.support_requests: List[SupportRequest] = []
        self.active_missions: Dict[str, SupportMission] = {}
        self.request_history: List[SupportRequest] = []
        
        # 区域划分（用于多USV协调）
        self.responsibility_zones: Dict[str, List[Tuple[float, float]]] = {}
        
        # 性能统计
        self.performance_stats = {
            'total_supports': 0,
            'successful_supports': 0,
            'average_response_time': 0.0,
            'total_energy_transferred': 0.0,
            'support_efficiency': 0.0,
            'patrol_distance': 0.0,
            'patrol_time': 0.0
        }
        
        # 巡逻参数
        self.patrol_enabled = usv_config.get('patrol_enabled', True)
        self.patrol_radius = usv_config.get('patrol_radius', 3000.0)  # 巡逻半径
        self.patrol_waypoint_count = usv_config.get('patrol_waypoint_count', 4)  # 巡逻路径点数量
    
    def plan(self, 
             agents_state: Dict[str, Dict[str, Any]], 
             tasks: List[Dict[str, Any]], 
             environment_state: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        制定USV支援调度方案
        
        注意：此方法主要用于USV的任务规划，返回格式保持与基类一致
        实际的支援调度通过 schedule_support 方法实现
        
        Args:
            agents_state: 所有智能体状态
            tasks: 任务列表（这里主要是USV的充电支援任务）
            environment_state: 环境状态
            
        Returns:
            USV的任务分配结果
        """
        # 更新USV状态
        self._update_usv_states(agents_state)
        
        # 分离UAV和USV状态
        uav_states = {}
        usv_states = {}
        for agent_id, state in agents_state.items():
            if state.get('type', '').lower() == 'usv' or 'usv' in agent_id.lower():
                usv_states[agent_id] = state
            else:
                uav_states[agent_id] = state
        
        # 调用支援调度
        import time
        current_time = time.time()
        support_missions = self.schedule_support(uav_states, usv_states, environment_state, current_time)
        
        # 转换为标准格式
        assignments = {}
        # 初始化所有USV为空任务列表
        for usv_id in usv_states:
            assignments[usv_id] = []
        
        # 分配支援任务
        for mission in support_missions:
            if mission.usv_id in assignments:
                # 创建虚拟任务ID
                task_id = hash(f"support_{mission.target_uav_id}_{mission.expected_arrival_time}") % 10000
                assignments[mission.usv_id].append(task_id)
        
        return assignments
    
    def schedule_support(self, 
                        uav_states: Dict[str, Dict[str, Any]], 
                        usv_states: Dict[str, Dict[str, Any]],
                        environment_state: Dict[str, Any],
                        current_time: float) -> List[SupportMission]:
        """
        主要的USV支援调度方法
        
        Args:
            uav_states: UAV状态字典
            usv_states: USV状态字典
            environment_state: 环境状态
            current_time: 当前时间
            
        Returns:
            支援任务列表
        """
        print(f"\n🚢 USV支援调度器开始评估...")
        print(f"  UAV数量: {len(uav_states)}, USV数量: {len(usv_states)}")
        
        # 显示UAV电量状态
        for uav_id, state in uav_states.items():
            energy_pct = (state.get('energy', 0) / state.get('max_energy', 300.0)) * 100
            print(f"  {uav_id}: 电量 {energy_pct:.1f}%")
        
        # 1. 更新状态
        self._update_usv_states(usv_states)
        
        # 2. 收集支援请求
        new_requests = self._collect_support_requests(uav_states, current_time)
        if new_requests:
            print(f"  🆘 收到 {len(new_requests)} 个支援请求:")
            for req in new_requests:
                energy_pct = (req.current_energy / 300.0) * 100
                print(f"    {req.uav_id}: 需要 {req.energy_needed:.1f}Wh (当前电量:{energy_pct:.1f}%)")
        else:
            # 检查是否真的没有需要支援的UAV
            low_battery_uavs = []
            for uav_id, state in uav_states.items():
                energy_pct = (state.get('energy', 0) / state.get('max_energy', 300.0)) * 100
                if energy_pct < 40:
                    low_battery_uavs.append(f"{uav_id}({energy_pct:.1f}%)")
            
            if low_battery_uavs:
                print(f"  ⚠️ 检测到低电量UAV但未生成支援请求: {', '.join(low_battery_uavs)}")
            else:
                print("  ✅ 当前没有UAV需要支援（电量都在40%以上）")
        
        self.support_requests.extend(new_requests)
        
        # 3. 过滤过期请求
        self.support_requests = [req for req in self.support_requests 
                               if req.deadline > current_time]
        
        # 4. 评估和分配支援任务
        support_missions = []
        
        # 处理紧急请求（贪心策略）
        emergency_missions = self._handle_emergency_requests(current_time)
        support_missions.extend(emergency_missions)
        print(f"  🚨 紧急支援任务: {len(emergency_missions)}个")
        
        # 处理常规请求（动态规划）
        regular_missions = self._optimize_support_allocation(current_time)
        support_missions.extend(regular_missions)
        print(f"  📋 常规支援任务: {len(regular_missions)}个")
        
        # 调试：如果有支援请求但没有支援任务，输出详细信息
        if self.support_requests and len(support_missions) == 0:
            print(f"  🔍 调试：有{len(self.support_requests)}个支援请求但分配了0个支援任务")
            for req in self.support_requests:
                energy_pct = (req.current_energy / 300.0) * 100
                print(f"    请求: {req.uav_id} 电量{energy_pct:.1f}% 需要{req.energy_needed:.1f}Wh")
        
        # 5. 更新活动任务
        for mission in support_missions:
            self.active_missions[mission.usv_id] = mission
            self.usv_states[mission.usv_id].current_mission = mission
            self.usv_states[mission.usv_id].status = USVStatus.MOVING_TO_SUPPORT
            print(f"  📍 {mission.usv_id} 开始支援任务: 前往支援 {mission.target_uav_id}")
        
        # 6. 为空闲的USV生成巡逻任务
        if self.patrol_enabled:
            patrol_missions = self._generate_patrol_missions(uav_states, current_time)
            for mission in patrol_missions:
                self.active_missions[mission.usv_id] = mission
                self.usv_states[mission.usv_id].current_mission = mission
                self.usv_states[mission.usv_id].status = USVStatus.PATROLLING
                print(f"  🚢 {mission.usv_id} 开始巡逻任务")
            support_missions.extend(patrol_missions)
        
        # 显示USV当前状态
        print("\n  🚢 USV当前状态:")
        for usv_id, usv_state in self.usv_states.items():
            status_str = usv_state.status.value
            energy_pct = (usv_state.current_energy / usv_state.max_energy) * 100
            print(f"    {usv_id}: 状态={status_str}, 电量={energy_pct:.1f}%, 位置={usv_state.position}")
            if usv_state.current_mission:
                if usv_state.current_mission.mission_type == "support":
                    print(f"      当前任务: 支援 {usv_state.current_mission.target_uav_id}")
                elif usv_state.current_mission.mission_type == "patrol":
                    print(f"      当前任务: 巡逻中 -> 目标点 {usv_state.current_mission.target_position}")
        
        return support_missions
    
    def _update_usv_states(self, agents_state: Dict[str, Dict[str, Any]]):
        """更新USV状态信息
        
        Args:
            agents_state: 所有智能体状态（可以是完整状态或仅USV状态）
        """
        for agent_id, state in agents_state.items():
            if state.get('type') == 'usv' or 'usv' in agent_id.lower():
                if agent_id not in self.usv_states:
                    # 创建新的USV状态
                    self.usv_states[agent_id] = USVState(
                        usv_id=agent_id,
                        position=tuple(state.get('position', [0.0, 0.0])),
                        current_energy=state.get('energy', 1000.0),
                        max_energy=state.get('max_energy', 1000.0),
                        status=USVStatus.IDLE,
                        home_base=tuple(state.get('home_base', [0.0, 0.0])),
                        charging_capacity=state.get('charging_capacity', 200.0),
                        charging_rate=state.get('charging_rate', 50.0)
                    )
                else:
                    # 更新现有状态
                    usv_state = self.usv_states[agent_id]
                    usv_state.position = tuple(state.get('position', usv_state.position))
                    usv_state.current_energy = state.get('energy', usv_state.current_energy)
                    
                    # 检查是否需要返回充电
                    if self._needs_recharge(usv_state):
                        usv_state.status = USVStatus.RETURNING
    
    def _collect_support_requests(self, 
                                uav_states: Dict[str, Dict[str, Any]], 
                                current_time: float) -> List[SupportRequest]:
        """收集需要支援的UAV请求"""
        requests = []
        
        for uav_id, state in uav_states.items():
            if state.get('type') == 'uav' or 'uav' in uav_id.lower():
                # 检查是否需要支援
                current_energy = state.get('energy', 0)
                max_energy = state.get('max_energy', 300.0)
                energy_percentage = current_energy / max_energy if max_energy > 0 else 0
                
                if energy_percentage < 0.4:  # 低于40%电量需要支援
                    # 评估风险任务
                    at_risk_tasks = self._evaluate_at_risk_tasks(state)
                    
                    # 计算需要的能量
                    energy_needed = self._calculate_energy_needed(state, at_risk_tasks)
                    
                    # 计算优先级
                    priority = self._calculate_request_priority(state, at_risk_tasks, energy_percentage)
                    
                    # 计算截止时间
                    deadline = current_time + self._estimate_energy_depletion_time(state)
                    
                    request = SupportRequest(
                        uav_id=uav_id,
                        position=tuple(state.get('position', [0, 0])),
                        current_energy=current_energy,
                        energy_needed=energy_needed,
                        priority=priority,
                        request_time=current_time,
                        deadline=deadline,
                        at_risk_tasks=at_risk_tasks
                    )
                    
                    # 避免重复请求
                    if not self._is_duplicate_request(request):
                        requests.append(request)
        
        return requests
    
    def _handle_emergency_requests(self, current_time: float) -> List[SupportMission]:
        """处理紧急支援请求（贪心策略）"""
        emergency_missions = []
        
        # 筛选紧急请求 - 修复电量计算
        emergency_requests = []
        for req in self.support_requests:
            energy_ratio = req.current_energy / 300.0
            # 使用更宽松的紧急阈值：25%或更低
            if energy_ratio < 0.25:
                emergency_requests.append(req)
        
        # 按优先级排序
        emergency_requests.sort(reverse=True)
        
        for request in emergency_requests:
            # 找到最近的可用USV
            best_usv = self._find_best_available_usv(request, emergency=True)
            
            if best_usv:
                # 创建紧急支援任务
                mission = self._create_support_mission(
                    best_usv, request, current_time, emergency=True
                )
                
                if mission:
                    emergency_missions.append(mission)
                    # 标记USV为忙碌
                    best_usv.status = USVStatus.MOVING_TO_SUPPORT
                    best_usv.supported_uavs.add(request.uav_id)
                    # 从请求列表中移除
                    self.support_requests.remove(request)
        
        return emergency_missions
    
    def _optimize_support_allocation(self, current_time: float) -> List[SupportMission]:
        """优化常规支援分配（动态规划）"""
        regular_missions = []
        
        # 获取可用的USV和待处理请求（包括巡逻中的USV）
        available_usvs = [usv for usv in self.usv_states.values() 
                         if usv.status == USVStatus.IDLE or usv.status == USVStatus.PATROLLING]
        pending_requests = [req for req in self.support_requests]
        
        print(f"    可用USV: {len(available_usvs)}个, 待处理请求: {len(pending_requests)}个")
        for usv in available_usvs:
            print(f"      {usv.usv_id}: 状态={usv.status.value}")
        
        if not available_usvs or not pending_requests:
            print(f"    跳过常规分配: 可用USV={len(available_usvs)}, 请求={len(pending_requests)}")
            return regular_missions
        
        # 强制分配：如果有低电量UAV且有可用USV，强制分配支援
        force_assignments = []
        for req in pending_requests:
            energy_ratio = req.current_energy / 300.0
            if energy_ratio < 0.3 and available_usvs:  # 30%以下强制支援
                best_usv = available_usvs[0]  # 选择第一个可用的USV
                force_assignments.append((0, pending_requests.index(req)))
                print(f"    🚨 强制分配: {best_usv.usv_id} → {req.uav_id} (电量{energy_ratio:.1%})")
                available_usvs.remove(best_usv)  # 移除已分配的USV
                break
        
        if force_assignments:
            # 创建强制支援任务
            for usv_idx, req_idx in force_assignments:
                usv = [usv for usv in self.usv_states.values() 
                      if usv.status == USVStatus.IDLE or usv.status == USVStatus.PATROLLING][usv_idx]
                request = pending_requests[req_idx]
                
                mission = self._create_support_mission(usv, request, current_time)
                if mission:
                    regular_missions.append(mission)
                    # 如果USV正在巡逻，先中断巡逻
                    if usv.status == USVStatus.PATROLLING:
                        self.interrupt_patrol_for_support(usv.usv_id)
                    # 更新状态
                    usv.status = USVStatus.MOVING_TO_SUPPORT
                    usv.supported_uavs.add(request.uav_id)
                    # 从请求列表中移除
                    self.support_requests.remove(request)
                    print(f"    ✅ 强制支援任务创建成功: {usv.usv_id} → {request.uav_id}")
        
        # 构建效益矩阵
        n_usvs = len(available_usvs)
        n_requests = len(pending_requests)
        benefit_matrix = np.zeros((n_usvs, n_requests))
        
        for i, usv in enumerate(available_usvs):
            for j, request in enumerate(pending_requests):
                benefit_matrix[i, j] = self._calculate_support_benefit(
                    usv, request, current_time
                )
        
        # 使用动态规划求解最优分配
        assignments = self._solve_assignment_dp(benefit_matrix, available_usvs, pending_requests)
        
        # 创建支援任务
        for usv_idx, req_idx in assignments:
            usv = available_usvs[usv_idx]
            request = pending_requests[req_idx]
            
            mission = self._create_support_mission(usv, request, current_time)
            if mission:
                regular_missions.append(mission)
                # 如果USV正在巡逻，先中断巡逻
                if usv.status == USVStatus.PATROLLING:
                    self.interrupt_patrol_for_support(usv.usv_id)
                # 更新状态
                usv.status = USVStatus.MOVING_TO_SUPPORT
                usv.supported_uavs.add(request.uav_id)
                # 从请求列表中移除
                self.support_requests.remove(request)
        
        return regular_missions
    
    def _calculate_support_benefit(self, 
                                 usv: USVState, 
                                 request: SupportRequest,
                                 current_time: float) -> float:
        """计算USV支援特定UAV的效益"""
        # 1. 计算可拯救的任务价值
        task_value = sum(task.get('priority', 1.0) * task.get('completion_probability', 0.8) 
                        for task in request.at_risk_tasks)
        
        # 2. 计算支援成本
        # 移动距离
        distance = self._calculate_distance(usv.position, request.position)
        
        # 移动能耗
        travel_energy, travel_time = self.energy_calculator.calculate_movement_energy(
            AgentType.USV, distance, 6.0  # USV巡航速度
        )
        
        # 充电能量转移
        charging_energy = min(request.energy_needed, usv.charging_capacity)
        charging_time = charging_energy / usv.charging_rate * 3600  # 转换为秒
        
        # 返回基地能耗
        return_distance = self._calculate_distance(request.position, usv.home_base)
        return_energy, return_time = self.energy_calculator.calculate_movement_energy(
            AgentType.USV, return_distance, 6.0
        )
        
        # 总成本
        total_energy_cost = travel_energy + charging_energy + return_energy
        total_time_cost = travel_time + charging_time + return_time
        
        # 3. 计算时间紧迫性
        time_urgency = 1.0 / max(1.0, request.deadline - current_time)
        
        # 4. 综合效益计算
        if total_energy_cost > 0:
            benefit = (self.benefit_weights['task_value'] * task_value * time_urgency) / (
                self.benefit_weights['energy_cost'] * total_energy_cost + 
                self.benefit_weights['time_cost'] * total_time_cost +
                self.benefit_weights['distance_cost'] * distance
            )
        else:
            benefit = float('inf')
        
        # 5. 考虑USV自身约束
        # 检查能量是否足够
        if not self._has_sufficient_energy(usv, total_energy_cost):
            benefit *= 0.1  # 大幅降低效益
        
        # 检查距离约束
        if distance > self.max_support_distance:
            benefit *= 0.5  # 降低效益
        
        # 考虑区域责任
        if self._is_in_responsibility_zone(usv.usv_id, request.position):
            benefit *= 1.2  # 提高效益
        
        return benefit
    
    def _solve_assignment_dp(self, 
                           benefit_matrix: np.ndarray,
                           usvs: List[USVState],
                           requests: List[SupportRequest]) -> List[Tuple[int, int]]:
        """使用动态规划求解分配问题"""
        n_usvs = len(usvs)
        n_requests = len(requests)
        
        # 如果请求少于USV，每个请求分配一个USV
        if n_requests <= n_usvs:
            # 贪心选择：为每个请求选择效益最高的USV
            assignments = []
            used_usvs = set()
            
            for j in range(n_requests):
                best_usv_idx = -1
                best_benefit = -1
                
                for i in range(n_usvs):
                    if i not in used_usvs and benefit_matrix[i, j] > best_benefit:
                        best_benefit = benefit_matrix[i, j]
                        best_usv_idx = i
                
                if best_usv_idx >= 0 and best_benefit > 0:
                    assignments.append((best_usv_idx, j))
                    used_usvs.add(best_usv_idx)
        else:
            # 请求多于USV，选择总效益最高的组合
            # 使用简化的贪心方法
            assignments = []
            assigned_requests = set()
            
            for i in range(n_usvs):
                best_req_idx = -1
                best_benefit = -1
                
                for j in range(n_requests):
                    if j not in assigned_requests and benefit_matrix[i, j] > best_benefit:
                        best_benefit = benefit_matrix[i, j]
                        best_req_idx = j
                
                if best_req_idx >= 0 and best_benefit > 0:
                    assignments.append((i, best_req_idx))
                    assigned_requests.add(best_req_idx)
        
        return assignments
    
    def _create_support_mission(self, 
                              usv: USVState, 
                              request: SupportRequest,
                              current_time: float,
                              emergency: bool = False) -> Optional[SupportMission]:
        """创建支援任务"""
        # 计算任务参数
        distance = self._calculate_distance(usv.position, request.position)
        travel_energy, travel_time = self.energy_calculator.calculate_movement_energy(
            AgentType.USV, distance, 8.0 if emergency else 6.0
        )
        
        # 计算充电参数
        energy_to_transfer = min(
            request.energy_needed,
            usv.charging_capacity,
            usv.current_energy * (1 - self.min_reserve_energy_ratio)
        )
        
        if energy_to_transfer <= 0:
            return None
        
        charging_time = energy_to_transfer / usv.charging_rate * 3600
        
        # 计算效益和成本
        benefit = self._calculate_support_benefit(usv, request, current_time)
        cost = travel_energy + energy_to_transfer
        
        mission = SupportMission(
            usv_id=usv.usv_id,
            target_uav_id=request.uav_id,
            start_position=usv.position,
            target_position=request.position,
            energy_to_transfer=energy_to_transfer * self.charging_transfer_efficiency,
            expected_arrival_time=current_time + travel_time,
            expected_completion_time=current_time + travel_time + charging_time,
            mission_benefit=benefit,
            mission_cost=cost,
            status=USVStatus.MOVING_TO_SUPPORT
        )
        
        return mission
    
    def _needs_recharge(self, usv: USVState) -> bool:
        """检查USV是否需要返回充电"""
        # 计算返回基地所需能量
        return_distance = self._calculate_distance(usv.position, usv.home_base)
        return_energy, _ = self.energy_calculator.calculate_movement_energy(
            AgentType.USV, return_distance, 6.0
        )
        
        # 预留能量
        reserve_energy = usv.max_energy * self.min_reserve_energy_ratio
        
        # 如果当前能量不足以返回基地+预留，则需要充电
        return usv.current_energy < (return_energy + reserve_energy)
    
    def _has_sufficient_energy(self, usv: USVState, required_energy: float) -> bool:
        """检查USV是否有足够能量完成任务"""
        # 计算返回基地所需能量
        return_distance = self._calculate_distance(usv.position, usv.home_base)
        return_energy, _ = self.energy_calculator.calculate_movement_energy(
            AgentType.USV, return_distance, 6.0
        )
        
        # 总需求 = 任务能量 + 返回能量 + 预留
        total_required = required_energy + return_energy + (usv.max_energy * self.min_reserve_energy_ratio)
        
        return usv.current_energy >= total_required
    
    def _find_best_available_usv(self, 
                               request: SupportRequest, 
                               emergency: bool = False) -> Optional[USVState]:
        """找到最佳可用USV"""
        available_usvs = [usv for usv in self.usv_states.values() 
                         if (usv.status == USVStatus.IDLE or usv.status == USVStatus.PATROLLING) and 
                         request.uav_id not in usv.supported_uavs]
        
        if not available_usvs:
            return None
        
        # 计算每个USV的得分
        best_usv = None
        best_score = -float('inf')
        
        for usv in available_usvs:
            # 距离因素
            distance = self._calculate_distance(usv.position, request.position)
            distance_score = 1.0 / (1.0 + distance / 1000.0)  # 归一化
            
            # 能量因素
            energy_score = usv.current_energy / usv.max_energy
            
            # 响应时间
            response_time = distance / 6.0  # USV速度
            time_score = 1.0 / (1.0 + response_time / self.support_response_time)
            
            # 综合得分
            if emergency:
                # 紧急情况下更重视距离和响应时间
                score = distance_score * 0.5 + time_score * 0.4 + energy_score * 0.1
            else:
                score = distance_score * 0.3 + time_score * 0.3 + energy_score * 0.4
            
            if score > best_score:
                best_score = score
                best_usv = usv
        
        # 如果选中的USV正在巡逻，中断巡逻任务
        if best_usv and best_usv.status == USVStatus.PATROLLING:
            self.interrupt_patrol_for_support(best_usv.usv_id)
        
        return best_usv
    
    def _evaluate_at_risk_tasks(self, uav_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """评估UAV的风险任务"""
        at_risk_tasks = []
        
        # 获取UAV的任务列表
        tasks = uav_state.get('assigned_tasks', [])
        current_energy = uav_state.get('energy', 0)
        
        for task in tasks:
            # 估算完成任务所需能量
            task_energy = task.get('energy_cost', 10.0)
            
            # 如果当前能量不足，则任务有风险
            if current_energy < task_energy * 1.2:  # 20%安全余量
                at_risk_tasks.append(task)
        
        return at_risk_tasks
    
    def _calculate_energy_needed(self, 
                               uav_state: Dict[str, Any], 
                               at_risk_tasks: List[Dict[str, Any]]) -> float:
        """计算UAV需要的支援能量"""
        # 基础需求：恢复到50%电量
        max_energy = uav_state.get('max_energy', 300.0)
        current_energy = uav_state.get('energy', 0)
        base_need = max_energy * 0.5 - current_energy
        
        # 任务需求：完成所有风险任务
        task_need = sum(task.get('energy_cost', 10.0) for task in at_risk_tasks)
        
        # 安全余量
        safety_need = max_energy * 0.1
        
        return max(0, base_need + task_need + safety_need)
    
    def _calculate_request_priority(self, 
                                  uav_state: Dict[str, Any],
                                  at_risk_tasks: List[Dict[str, Any]],
                                  energy_percentage: float) -> float:
        """计算支援请求优先级"""
        # 基础优先级（基于能量水平）
        base_priority = (1.0 - energy_percentage) * 10.0
        
        # 任务优先级
        task_priority = sum(task.get('priority', 1.0) for task in at_risk_tasks)
        
        # 任务数量因素
        task_count_factor = min(len(at_risk_tasks) / 3.0, 2.0)
        
        # 综合优先级
        priority = base_priority + task_priority * task_count_factor
        
        return priority
    
    def _estimate_energy_depletion_time(self, uav_state: Dict[str, Any]) -> float:
        """估算UAV能量耗尽时间"""
        current_energy = uav_state.get('energy', 0)
        consumption_rate = uav_state.get('consumption_rate', 50.0 / 3600)  # W转换为Wh/s
        
        if consumption_rate > 0:
            depletion_time = current_energy / consumption_rate
        else:
            depletion_time = float('inf')
        
        return min(depletion_time, 3600.0)  # 最多1小时
    
    def _is_duplicate_request(self, request: SupportRequest) -> bool:
        """检查是否为重复请求"""
        for existing_req in self.support_requests:
            if existing_req.uav_id == request.uav_id:
                # 如果已有请求且状态变化不大，视为重复
                if abs(existing_req.current_energy - request.current_energy) < 10.0:
                    return True
        return False
    
    def _is_in_responsibility_zone(self, usv_id: str, position: Tuple[float, float]) -> bool:
        """检查位置是否在USV的责任区域内"""
        if usv_id not in self.responsibility_zones:
            return True  # 如果没有划分区域，默认都负责
        
        zone = self.responsibility_zones[usv_id]
        # 简单的区域检查（可以扩展为更复杂的多边形检查）
        # 这里假设zone是一个矩形区域 [(x_min, y_min), (x_max, y_max)]
        if len(zone) >= 2:
            x_min, y_min = zone[0]
            x_max, y_max = zone[1]
            x, y = position
            return x_min <= x <= x_max and y_min <= y <= y_max
        
        return True
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """计算两点间欧氏距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def update_mission_status(self, mission_id: str, new_status: USVStatus):
        """更新任务状态"""
        if mission_id in self.active_missions:
            mission = self.active_missions[mission_id]
            mission.status = new_status
            
            # 更新USV状态
            if mission.usv_id in self.usv_states:
                self.usv_states[mission.usv_id].status = new_status
                
                # 如果任务完成，更新统计
                if new_status == USVStatus.IDLE:
                    self.performance_stats['total_supports'] += 1
                    self.performance_stats['successful_supports'] += 1
                    self.performance_stats['total_energy_transferred'] += mission.energy_to_transfer
                    
                    # 将任务移到历史记录
                    self.usv_states[mission.usv_id].mission_history.append(mission)
                    del self.active_missions[mission_id]
    
    def set_responsibility_zones(self, zones: Dict[str, List[Tuple[float, float]]]):
        """设置USV责任区域"""
        self.responsibility_zones = zones
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if self.performance_stats['total_supports'] > 0:
            self.performance_stats['support_efficiency'] = (
                self.performance_stats['successful_supports'] / 
                self.performance_stats['total_supports']
            )
        
        return {
            'stats': self.performance_stats.copy(),
            'active_missions': len(self.active_missions),
            'pending_requests': len(self.support_requests),
            'usv_utilization': self._calculate_usv_utilization()
        }
    
    def _calculate_usv_utilization(self) -> float:
        """计算USV利用率"""
        if not self.usv_states:
            return 0.0
        
        busy_usvs = sum(1 for usv in self.usv_states.values() 
                       if usv.status != USVStatus.IDLE)
        
        return busy_usvs / len(self.usv_states)
    
    def evaluate(self, 
                 assignment: Dict[str, List[int]], 
                 agents_state: Dict[str, Dict[str, Any]], 
                 tasks: List[Dict[str, Any]]) -> float:
        """
        评估支援方案的质量
        
        Args:
            assignment: 任务分配方案
            agents_state: 智能体状态
            tasks: 任务列表
            
        Returns:
            评估分数（越高越好）
        """
        total_score = 0.0
        
        # 评估每个USV的支援效益
        for usv_id, task_ids in assignment.items():
            if usv_id in self.usv_states:
                usv = self.usv_states[usv_id]
                
                for task_id in task_ids:
                    # 这里的task_id实际上对应支援任务
                    # 需要从active_missions中获取实际的支援信息
                    for mission in self.active_missions.values():
                        if mission.usv_id == usv_id:
                            # 效益/成本比
                            if mission.mission_cost > 0:
                                efficiency = mission.mission_benefit / mission.mission_cost
                            else:
                                efficiency = mission.mission_benefit
                            
                            total_score += efficiency
        
        # 考虑响应时间
        avg_response_time = self._calculate_average_response_time()
        time_penalty = max(0, avg_response_time - self.support_response_time) / self.support_response_time
        total_score *= (1 - time_penalty * 0.2)
        
        # 考虑覆盖率
        coverage_rate = self._calculate_coverage_rate(agents_state)
        total_score *= (0.8 + coverage_rate * 0.2)
        
        return total_score
    
    def _calculate_average_response_time(self) -> float:
        """计算平均响应时间"""
        if not self.active_missions:
            return 0.0
        
        total_time = sum(mission.expected_arrival_time - mission.expected_arrival_time 
                        for mission in self.active_missions.values())
        
        return total_time / len(self.active_missions)
    
    def _calculate_coverage_rate(self, agents_state: Dict[str, Dict[str, Any]]) -> float:
        """计算UAV覆盖率（有多少UAV能得到及时支援）"""
        total_uavs = sum(1 for agent in agents_state.values() 
                        if agent.get('type') == 'uav' or 'uav' in str(agent.get('id', '')).lower())
        
        if total_uavs == 0:
            return 1.0
        
        covered_uavs = 0
        for agent_id, state in agents_state.items():
            if state.get('type') == 'uav' or 'uav' in agent_id.lower():
                # 检查是否有USV能够及时支援
                min_distance = float('inf')
                for usv in self.usv_states.values():
                    distance = self._calculate_distance(usv.position, tuple(state['position']))
                    min_distance = min(min_distance, distance)
                
                if min_distance <= self.max_support_distance:
                    covered_uavs += 1
        
        return covered_uavs / total_uavs
    
    def _generate_patrol_missions(self, 
                                uav_states: Dict[str, Dict[str, Any]], 
                                current_time: float) -> List[SupportMission]:
        """为空闲USV生成巡逻任务"""
        patrol_missions = []
        
        # 获取所有空闲的USV
        idle_usvs = [usv for usv in self.usv_states.values() 
                     if usv.status == USVStatus.IDLE and 
                     not self._needs_recharge(usv)]
        
        if not idle_usvs:
            return patrol_missions
        
        # 获取UAV活动区域中心点
        uav_positions = []
        for uav_id, state in uav_states.items():
            if state.get('type') == 'uav' or 'uav' in uav_id.lower():
                uav_positions.append(state.get('position', [0, 0]))
        
        if not uav_positions:
            # 如果没有UAV，使用默认巡逻区域
            center = [400, 400]  # 默认中心点
        else:
            # 计算UAV群的重心
            center = [
                sum(pos[0] for pos in uav_positions) / len(uav_positions),
                sum(pos[1] for pos in uav_positions) / len(uav_positions)
            ]
        
        # 为每个空闲USV生成巡逻路径
        for i, usv in enumerate(idle_usvs):
            # 如果USV还没有巡逻路径，生成一个
            if not usv.patrol_waypoints:
                waypoints = self._generate_patrol_waypoints(center, i, len(idle_usvs))
                usv.patrol_waypoints = waypoints
                usv.current_patrol_index = 0
            
            # 获取下一个巡逻点
            next_waypoint = usv.patrol_waypoints[usv.current_patrol_index]
            
            # 创建巡逻任务
            mission = SupportMission(
                usv_id=usv.usv_id,
                target_uav_id="patrol",  # 特殊标识
                start_position=usv.position,
                target_position=next_waypoint,
                energy_to_transfer=0.0,
                expected_arrival_time=current_time + self._calculate_patrol_time(usv.position, next_waypoint),
                expected_completion_time=current_time + self._calculate_patrol_time(usv.position, next_waypoint) + 30,
                mission_benefit=1.0,  # 巡逻的基础效益
                mission_cost=self._calculate_patrol_cost(usv.position, next_waypoint),
                status=USVStatus.PATROLLING,
                mission_type="patrol"
            )
            
            patrol_missions.append(mission)
            
            # 更新巡逻索引（循环）
            usv.current_patrol_index = (usv.current_patrol_index + 1) % len(usv.patrol_waypoints)
        
        return patrol_missions
    
    def _generate_patrol_waypoints(self, 
                                 center: List[float], 
                                 usv_index: int, 
                                 total_usvs: int) -> List[Tuple[float, float]]:
        """生成巡逻路径点"""
        waypoints = []
        
        # 根据USV数量和索引生成不同的巡逻路径
        angle_offset = (2 * math.pi * usv_index) / max(1, total_usvs)
        
        # 生成围绕中心的巡逻点
        for i in range(self.patrol_waypoint_count):
            angle = angle_offset + (2 * math.pi * i) / self.patrol_waypoint_count
            radius = self.patrol_radius * (0.5 + 0.5 * (i % 2))  # 交替内外圈
            
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            
            # 确保坐标在合理范围内
            x = max(50, min(950, x))  # 假设地图宽度1000
            y = max(50, min(750, y))  # 假设地图高度800
            
            waypoints.append((x, y))
        
        return waypoints
    
    def _calculate_patrol_time(self, 
                             start_pos: Tuple[float, float], 
                             end_pos: Tuple[float, float]) -> float:
        """计算巡逻时间"""
        distance = self._calculate_distance(start_pos, end_pos)
        patrol_speed = 5.0  # USV巡逻速度 m/s
        return distance / patrol_speed
    
    def _calculate_patrol_cost(self, 
                             start_pos: Tuple[float, float], 
                             end_pos: Tuple[float, float]) -> float:
        """计算巡逻成本"""
        distance = self._calculate_distance(start_pos, end_pos)
        energy_cost, _ = self.energy_calculator.calculate_movement_energy(
            AgentType.USV, distance, 5.0  # 巡逻速度
        )
        return energy_cost
    
    def interrupt_patrol_for_support(self, usv_id: str):
        """中断USV的巡逻任务以执行支援任务"""
        if usv_id in self.usv_states:
            usv = self.usv_states[usv_id]
            if usv.status == USVStatus.PATROLLING:
                # 保存当前巡逻状态
                print(f"  ⚠️ {usv_id} 中断巡逻任务")
                usv.status = USVStatus.IDLE
                
                # 记录巡逻统计
                if usv.current_mission:
                    patrol_distance = self._calculate_distance(
                        usv.current_mission.start_position,
                        usv.position
                    )
                    self.performance_stats['patrol_distance'] += patrol_distance
                    self.performance_stats['patrol_time'] += (
                        time.time() - (usv.current_mission.expected_arrival_time - 
                                     self._calculate_patrol_time(
                                         usv.current_mission.start_position,
                                         usv.current_mission.target_position
                                     ))
                    )