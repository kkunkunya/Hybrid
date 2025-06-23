"""
集成调度器
整合所有4层调度器的功能，提供统一的任务分配和动态调度接口
"""
import copy
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time

from .base_scheduler import BaseScheduler
from .energy_aware_scheduler import EnergyAwareScheduler
from .usv_logistics_scheduler import USVLogisticsScheduler
from .charging_decision_maker import ChargingDecisionMaker, ChargingOption
from .dynamic_reallocator import DynamicReallocator, TriggerType


@dataclass
class IntegratedSchedulingResult:
    """集成调度结果"""
    task_assignment: Dict[str, List[int]]  # UAV任务分配
    usv_support_missions: Dict[str, Any]  # USV支援任务
    charging_decisions: Dict[str, Any]  # 充电决策
    reallocation_plan: Optional[Any]  # 重调度计划


class IntegratedScheduler:
    """
    集成调度器 - 4层优化架构的统一接口
    
    Layer 1: 能源感知任务分配器 (EnergyAwareScheduler)
    Layer 2: USV后勤智能调度器 (USVLogisticsScheduler)  
    Layer 3: 充电决策优化器 (ChargingDecisionMaker)
    Layer 4: 动态重调度管理器 (DynamicReallocator)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化集成调度器
        
        Args:
            config: 包含所有子调度器配置的字典
        """
        self.config = config
        
        # 初始化各层调度器
        self.task_allocator = EnergyAwareScheduler(config)
        self.usv_scheduler = USVLogisticsScheduler(config)
        self.charging_decision_maker = ChargingDecisionMaker(config)
        self.dynamic_reallocator = DynamicReallocator(config)
        
        # 状态管理
        self.current_assignment = {}
        self.last_scheduling_time = time.time()
        self.scheduling_history = []
        
        # 性能统计
        self.performance_metrics = {
            'total_schedules': 0,
            'successful_schedules': 0,
            'average_response_time': 0.0,
            'reallocation_count': 0
        }
    
    def schedule(self, 
                agents_state: Dict[str, Dict[str, Any]], 
                tasks: List[Dict[str, Any]], 
                environment_state: Dict[str, Any],
                force_reallocation: bool = False) -> IntegratedSchedulingResult:
        """
        执行完整的4层调度流程
        
        Args:
            agents_state: 所有智能体状态
            tasks: 待分配任务列表
            environment_state: 环境状态
            force_reallocation: 是否强制重调度
            
        Returns:
            集成调度结果
        """
        start_time = time.time()
        
        # 分离UAV和USV状态
        uav_states, usv_states = self._separate_agent_states(agents_state)
        
        # Layer 4: 检查是否需要动态重调度
        reallocation_plan = None
        if self.current_assignment and not force_reallocation:
            # 使用动态重调度器检查并执行重调度
            new_assignment = self.dynamic_reallocator.plan(
                agents_state, tasks, environment_state
            )
            
            # 如果重调度器返回了新方案，说明触发了重调度
            if new_assignment != self.current_assignment:
                reallocation_plan = {
                    'trigger': 'dynamic_reallocator',
                    'old_assignment': copy.deepcopy(self.current_assignment),
                    'new_assignment': new_assignment
                }
                self.current_assignment = new_assignment
                self.performance_metrics['reallocation_count'] += 1
        else:
            # Layer 1: 能源感知任务分配
            self.current_assignment = self.task_allocator.plan(
                agents_state, tasks, environment_state
            )
        
        # 获取UAV任务分配
        uav_assignment = {
            agent_id: task_ids 
            for agent_id, task_ids in self.current_assignment.items()
            if agent_id in uav_states
        }
        
        # Layer 2: USV后勤支援调度
        # 创建虚拟的支援任务（基于UAV状态）
        support_tasks = self._create_support_tasks(uav_states, uav_assignment)
        # 传递完整的agents_state，让USV调度器能看到所有智能体状态
        usv_support_plan = self.usv_scheduler.plan(
            agents_state, support_tasks, environment_state
        )
        
        # Layer 3: 充电决策
        charging_decisions = {}
        
        # 为充电决策准备USV状态
        # 注意：这里使用原始的usv_states，因为充电决策应该能看到所有可用的USV
        # USV调度器会优先处理紧急支援，但充电决策可能会选择其他USV
        
        # 准备充电站信息
        charging_stations = []
        for station_info in environment_state.get('charging_stations', []):
            from .charging_decision_maker import ChargingStation
            station = ChargingStation(
                station_id=str(station_info.get('id', 0)),
                position=tuple(station_info.get('position', (0, 0))),
                capacity=station_info.get('capacity', 4),
                queue=[],  # 简化处理，实际应该追踪排队情况
            )
            # 模拟排队长度
            for _ in range(station_info.get('queue_length', 0)):
                station.queue.append(f"dummy_uav_{_}")
            charging_stations.append(station)
        
        # 系统状态
        system_state = {
            'current_time': time.time(),
            'total_uavs': len(uav_states),
            'active_tasks': sum(len(tasks) for tasks in uav_assignment.values()),
            'weather': environment_state.get('weather', 'clear')
        }
        
        for uav_id, uav_state in uav_states.items():
            # 检查是否需要充电决策
            if self._needs_charging_decision(uav_state):
                # 添加UAV ID到状态中，并规范化字段名
                uav_state_with_id = {
                    'id': uav_id,
                    'position': uav_state.get('position', (0, 0)),
                    'current_energy': uav_state.get('energy', 0),
                    'max_energy': uav_state.get('max_energy', 300.0),
                    'energy_consumption_rate': uav_state.get('consumption_rate', 10.0),
                    'current_task_priority': uav_state.get('task_priority', 1.0),
                    'status': uav_state.get('status', 'idle')
                }
                
                decision = self.charging_decision_maker.make_decision(
                    uav_state_with_id, charging_stations, usv_states, system_state
                )
                charging_decisions[uav_id] = decision
        
        # 调试：检查任务分配结果
        total_assigned_tasks = 0
        for agent_id, task_ids in self.current_assignment.items():
            if agent_id in uav_states and task_ids:
                total_assigned_tasks += len(task_ids)
        
        total_pending_tasks = sum(1 for task in tasks if task.get('status', 'pending') == 'pending')
        
        if total_assigned_tasks == 0 and total_pending_tasks > 0:
            print(f"⚠️ 集成调度器警告：没有分配任何任务，但有{total_pending_tasks}个待分配任务")
            print(f"  当前分配: {self.current_assignment}")
            
            # 尝试强制分配
            if not reallocation_plan:
                print("  强制触发任务分配...")
                # 重新运行任务分配
                self.current_assignment = self.task_allocator.plan(
                    agents_state, tasks, environment_state
                )
        
        # 调试：检查任务分配结果
        total_assigned_tasks = 0
        for agent_id, task_ids in self.current_assignment.items():
            if agent_id in uav_states and task_ids:
                total_assigned_tasks += len(task_ids)
        
        total_pending_tasks = sum(1 for task in tasks if task.get('status', 'pending') == 'pending')
        
        if total_assigned_tasks == 0 and total_pending_tasks > 0:
            print(f"⚠️ 集成调度器警告：没有分配任何任务，但有{total_pending_tasks}个待分配任务")
            print(f"  当前分配: {self.current_assignment}")
            
            # 尝试强制分配
            if not reallocation_plan:
                print("  强制触发任务分配...")
                # 重新运行任务分配
                self.current_assignment = self.task_allocator.plan(
                    agents_state, tasks, environment_state
                )
        
        # 构建集成结果
        result = IntegratedSchedulingResult(
            task_assignment=self.current_assignment,
            usv_support_missions=self._format_usv_missions(usv_support_plan, support_tasks),
            charging_decisions=charging_decisions,
            reallocation_plan=reallocation_plan
        )
        
        # 更新性能统计
        elapsed_time = time.time() - start_time
        self._update_performance_metrics(elapsed_time, result)
        
        return result
    
    def _separate_agent_states(self, agents_state: Dict[str, Dict[str, Any]]) -> Tuple[Dict, Dict]:
        """分离UAV和USV状态"""
        uav_states = {}
        usv_states = {}
        
        for agent_id, state in agents_state.items():
            agent_type = state.get('type', 'uav').lower()
            if agent_type == 'uav':
                uav_states[agent_id] = state
            elif agent_type == 'usv':
                usv_states[agent_id] = state
        
        return uav_states, usv_states
    
    def _create_support_tasks(self, 
                            uav_states: Dict[str, Dict[str, Any]], 
                            uav_assignment: Dict[str, List[int]]) -> List[Dict[str, Any]]:
        """基于UAV状态创建支援任务"""
        support_tasks = []
        
        for uav_id, uav_state in uav_states.items():
            # 检查UAV是否需要支援
            energy_ratio = uav_state.get('energy', 0) / uav_state.get('max_energy', 300.0)
            
            if energy_ratio < 0.4:  # 电量低于40%考虑支援
                # 计算任务价值
                task_count = len(uav_assignment.get(uav_id, []))
                priority = 2.0 if energy_ratio < 0.15 else 1.0
                
                support_task = {
                    'task_id': f"support_{uav_id}",
                    'uav_id': uav_id,
                    'position': uav_state['position'],
                    'priority': priority,
                    'energy_needed': (0.5 - energy_ratio) * uav_state.get('max_energy', 300.0),
                    'at_risk_tasks': task_count,
                    'urgency': 'high' if energy_ratio < 0.15 else 'medium'
                }
                support_tasks.append(support_task)
        
        return support_tasks
    
    def _needs_charging_decision(self, uav_state: Dict[str, Any]) -> bool:
        """判断UAV是否需要充电决策"""
        energy_ratio = uav_state.get('energy', 0) / uav_state.get('max_energy', 300.0)
        status = uav_state.get('status', 'idle')
        
        # 电量低于40%且不在充电状态时需要充电决策
        return energy_ratio < 0.4 and status not in ['charging', 'returning']
    
    def _format_usv_missions(self, 
                           usv_support_plan: Dict[str, List[int]], 
                           support_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """格式化USV支援任务"""
        usv_missions = {}
        
        # 直接从USV调度器获取详细的任务信息
        if hasattr(self.usv_scheduler, 'active_missions'):
            for usv_id, mission in self.usv_scheduler.active_missions.items():
                missions = []
                if mission.mission_type == "support":
                    missions.append({
                        'target_uav': mission.target_uav_id,
                        'position': mission.target_position,
                        'energy_to_transfer': mission.energy_to_transfer,
                        'priority': mission.mission_benefit,
                        'mission_type': 'support'
                    })
                elif mission.mission_type == "patrol":
                    missions.append({
                        'target_uav': 'patrol',
                        'target_position': mission.target_position,
                        'energy_to_transfer': 0,
                        'priority': 1.0,
                        'mission_type': 'patrol'
                    })
                
                if missions:
                    usv_missions[usv_id] = missions
        else:
            # 降级处理：使用原来的逻辑
            for usv_id, task_indices in usv_support_plan.items():
                if task_indices:
                    missions = []
                    for idx in task_indices:
                        if idx < len(support_tasks):
                            task = support_tasks[idx]
                            missions.append({
                                'target_uav': task['uav_id'],
                                'position': task['position'],
                                'energy_to_transfer': task['energy_needed'],
                                'priority': task['priority'],
                                'mission_type': 'support'
                            })
                    
                    if missions:
                        usv_missions[usv_id] = missions
        
        return usv_missions
    
    def _update_performance_metrics(self, elapsed_time: float, result: IntegratedSchedulingResult):
        """更新性能统计"""
        self.performance_metrics['total_schedules'] += 1
        
        # 检查是否成功（至少有任务分配）
        if any(result.task_assignment.values()):
            self.performance_metrics['successful_schedules'] += 1
        
        # 更新平均响应时间
        n = self.performance_metrics['total_schedules']
        self.performance_metrics['average_response_time'] = (
            (self.performance_metrics['average_response_time'] * (n - 1) + elapsed_time) / n
        )
        
        # 保存历史记录
        self.scheduling_history.append({
            'timestamp': time.time(),
            'elapsed_time': elapsed_time,
            'task_count': sum(len(tasks) for tasks in result.task_assignment.values()),
            'charging_decisions': len(result.charging_decisions),
            'usv_missions': len(result.usv_support_missions),
            'reallocation': result.reallocation_plan is not None
        })
        
        # 限制历史记录大小
        if len(self.scheduling_history) > 100:
            self.scheduling_history.pop(0)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {
            '总体统计': {
                '总调度次数': self.performance_metrics['total_schedules'],
                '成功调度次数': self.performance_metrics['successful_schedules'],
                '成功率': f"{self.performance_metrics['successful_schedules'] / max(self.performance_metrics['total_schedules'], 1):.2%}",
                '平均响应时间': f"{self.performance_metrics['average_response_time']:.3f}秒",
                '重调度次数': self.performance_metrics['reallocation_count']
            },
            '各层调度器统计': {
                'Layer1_任务分配': self.task_allocator.get_allocation_statistics(),
                'Layer2_USV支援': self.usv_scheduler.get_performance_report(),
                'Layer3_充电决策': self.charging_decision_maker.get_statistics() if hasattr(self.charging_decision_maker, 'get_statistics') else {},
                'Layer4_动态重调度': self.dynamic_reallocator.get_statistics() if hasattr(self.dynamic_reallocator, 'get_statistics') else {}
            }
        }
        
        return report
    
    def reset_statistics(self):
        """重置统计信息"""
        self.performance_metrics = {
            'total_schedules': 0,
            'successful_schedules': 0,
            'average_response_time': 0.0,
            'reallocation_count': 0
        }
        self.scheduling_history = []
        self.current_assignment = {}


# 演示用法
if __name__ == "__main__":
    # 测试配置
    config = {
        'energy_aware_allocator': {
            'cost_weights': {'distance': 0.3, 'energy': 0.4, 'time': 0.2, 'risk': 0.1},
            'safety_margin': 0.25
        },
        'usv_logistics': {
            'support_benefit_weights': {'task_value': 0.4, 'energy_cost': 0.3},
            'min_reserve_energy_ratio': 0.3
        },
        'charging_decision': {
            'decision_factors': {'distance': 0.3, 'queue_time': 0.25}
        },
        'dynamic_reallocator': {
            'trigger_thresholds': {'critical_energy': 0.15}
        }
    }
    
    # 创建集成调度器
    scheduler = IntegratedScheduler(config)
    
    # 测试数据
    agents_state = {
        'uav1': {'position': (100, 100), 'energy': 150, 'max_energy': 300, 'type': 'uav', 'status': 'idle'},
        'uav2': {'position': (200, 200), 'energy': 80, 'max_energy': 300, 'type': 'uav', 'status': 'idle'},
        'usv1': {'position': (150, 150), 'energy': 800, 'max_energy': 1000, 'type': 'usv', 'status': 'idle'}
    }
    
    tasks = [
        {'task_id': 0, 'position': (150, 150), 'priority': 2.0},
        {'task_id': 1, 'position': (250, 250), 'priority': 1.5}
    ]
    
    environment_state = {'weather': 'clear'}
    
    # 执行调度
    result = scheduler.schedule(agents_state, tasks, environment_state)
    
    print("=== 集成调度结果 ===")
    print(f"任务分配: {result.task_assignment}")
    print(f"USV支援: {result.usv_support_missions}")
    print(f"充电决策: {result.charging_decisions}")
    print(f"重调度: {'是' if result.reallocation_plan else '否'}")
    
    # 显示性能报告
    print("\n=== 性能报告 ===")
    report = scheduler.get_performance_report()
    for category, stats in report.items():
        print(f"\n{category}:")
        if isinstance(stats, dict):
            for key, value in stats.items():
                print(f"  {key}: {value}")