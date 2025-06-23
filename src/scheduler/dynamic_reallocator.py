"""
动态重调度管理器 (Layer 4)
实现事件驱动的任务重新分配，响应系统状态变化
"""
import logging
from typing import Dict, List, Any, Tuple, Optional, Set
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

from src.scheduler.base_scheduler import BaseScheduler
from src.utils.energy import EnergyCalculator, AgentType


class TriggerLevel(Enum):
    """触发级别枚举"""
    IMMEDIATE = 1  # 立即触发
    DELAYED = 2    # 延迟触发
    PERIODIC = 3   # 周期性触发


class TriggerType(Enum):
    """触发类型枚举"""
    CRITICAL_ENERGY = "critical_energy"  # 能量危机
    TASK_FAILURE = "task_failure"        # 任务失败
    DEVICE_FAILURE = "device_failure"    # 设备故障
    HIGH_PRIORITY_TASK = "high_priority_task"  # 高优先级任务
    ENERGY_DEVIATION = "energy_deviation"  # 能耗偏差
    USV_BACKLOG = "usv_backlog"          # USV支援积压
    PERFORMANCE_DROP = "performance_drop"  # 性能下降
    LOAD_IMBALANCE = "load_imbalance"    # 负载不均衡
    ENVIRONMENT_CHANGE = "environment_change"  # 环境变化


@dataclass
class TriggerEvent:
    """触发事件数据结构"""
    event_type: TriggerType
    level: TriggerLevel
    timestamp: datetime
    agent_id: Optional[str] = None
    task_id: Optional[int] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReallocationPlan:
    """重调度计划"""
    affected_agents: Set[str]
    task_transfers: Dict[str, List[Tuple[int, str]]]  # {from_agent: [(task_id, to_agent)]}
    new_assignments: Dict[str, List[int]]
    is_global: bool
    transition_steps: int = 3


class DynamicReallocator(BaseScheduler):
    """动态重调度管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化动态重调度器
        
        Args:
            config: 配置参数字典
        """
        super().__init__(config)
        
        # 从配置中提取参数
        realloc_config = config.get('dynamic_reallocator', {})
        
        # 触发阈值
        self.trigger_thresholds = realloc_config.get('trigger_thresholds', {
            'critical_energy': 0.15,
            'performance_degradation': 0.2,
            'load_imbalance': 0.3,
            'task_failure_rate': 0.1
        })
        
        # 重调度参数
        self.reallocation_interval = realloc_config.get('reallocation_interval', 900)  # 15分钟
        self.smooth_transition_steps = realloc_config.get('smooth_transition_steps', 3)
        self.max_reallocation_frequency = realloc_config.get('max_reallocation_frequency', 4)
        self.local_adjustment_threshold = realloc_config.get('local_adjustment_threshold', 0.3)
        
        # 状态管理
        self.event_queue: List[TriggerEvent] = []
        self.last_reallocation_time = datetime.now()
        self.reallocation_count = 0
        self.last_periodic_check = datetime.now()
        
        # 历史记录
        self.assignment_history: List[Dict[str, List[int]]] = []
        self.performance_history: List[float] = []
        
        # 当前分配方案
        self.current_assignment: Dict[str, List[int]] = {}
        
        # 能源计算器
        self.energy_calculator = EnergyCalculator()
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
    def plan(self, 
             agents_state: Dict[str, Dict[str, Any]], 
             tasks: List[Dict[str, Any]], 
             environment_state: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        制定任务分配方案（包含动态重调度逻辑）
        
        Args:
            agents_state: 所有智能体的当前状态
            tasks: 待分配的任务列表
            environment_state: 环境状态
            
        Returns:
            任务分配结果
        """
        # 检查触发条件
        triggers = self._check_triggers(agents_state, tasks, environment_state)
        
        # 处理触发事件
        if triggers:
            self._process_triggers(triggers, agents_state, tasks, environment_state)
        
        # 周期性检查
        if self._should_perform_periodic_check():
            self._perform_periodic_check(agents_state, tasks, environment_state)
        
        # 如果需要重调度
        if self._should_reallocate():
            reallocation_plan = self._create_reallocation_plan(
                agents_state, tasks, environment_state
            )
            
            if reallocation_plan:
                self.current_assignment = self._execute_reallocation(
                    reallocation_plan, agents_state, tasks
                )
        
        # 返回当前分配方案（可能已更新）
        return self.current_assignment
    
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
            评估分数（越高越好）
        """
        score = 0.0
        
        # 1. 任务覆盖率
        total_tasks = len(tasks)
        assigned_tasks = sum(len(task_list) for task_list in assignment.values())
        coverage_score = assigned_tasks / total_tasks if total_tasks > 0 else 0
        score += coverage_score * 30
        
        # 2. 能源效率
        energy_efficiency = self._calculate_energy_efficiency(
            assignment, agents_state, tasks
        )
        score += energy_efficiency * 25
        
        # 3. 负载均衡
        load_balance = self._calculate_load_balance(assignment, agents_state)
        score += load_balance * 20
        
        # 4. 任务优先级满足度
        priority_satisfaction = self._calculate_priority_satisfaction(
            assignment, tasks
        )
        score += priority_satisfaction * 15
        
        # 5. 稳定性（减少频繁重调度）
        stability_score = self._calculate_stability_score(assignment)
        score += stability_score * 10
        
        return score
    
    def _check_triggers(self, 
                       agents_state: Dict[str, Dict[str, Any]], 
                       tasks: List[Dict[str, Any]], 
                       environment_state: Dict[str, Any]) -> List[TriggerEvent]:
        """检查各类触发条件"""
        triggers = []
        current_time = datetime.now()
        
        # 级别1 - 立即触发
        # 检查能量危机
        for agent_id, state in agents_state.items():
            if state.get('type') == 'uav':
                energy_percentage = state.get('energy', 100) / state.get('max_energy', 100)
                if energy_percentage < self.trigger_thresholds['critical_energy']:
                    triggers.append(TriggerEvent(
                        event_type=TriggerType.CRITICAL_ENERGY,
                        level=TriggerLevel.IMMEDIATE,
                        timestamp=current_time,
                        agent_id=agent_id,
                        data={'energy_percentage': energy_percentage}
                    ))
        
        # 检查任务失败
        failed_tasks = [t for t in tasks if t.get('status') == 'failed']
        if failed_tasks:
            for task in failed_tasks:
                triggers.append(TriggerEvent(
                    event_type=TriggerType.TASK_FAILURE,
                    level=TriggerLevel.IMMEDIATE,
                    timestamp=current_time,
                    task_id=task.get('task_id'),
                    data={'task': task}
                ))
        
        # 检查设备故障
        failed_agents = [a for a, s in agents_state.items() if s.get('status') == 'failed']
        if failed_agents:
            for agent_id in failed_agents:
                triggers.append(TriggerEvent(
                    event_type=TriggerType.DEVICE_FAILURE,
                    level=TriggerLevel.IMMEDIATE,
                    timestamp=current_time,
                    agent_id=agent_id,
                    data={'agent_state': agents_state[agent_id]}
                ))
        
        # 级别2 - 延迟触发
        # 检查高优先级新任务
        new_high_priority_tasks = [
            t for t in tasks 
            if t.get('status') == 'pending' and t.get('priority', 1.0) > 2.5
        ]
        if new_high_priority_tasks:
            triggers.append(TriggerEvent(
                event_type=TriggerType.HIGH_PRIORITY_TASK,
                level=TriggerLevel.DELAYED,
                timestamp=current_time,
                data={'tasks': new_high_priority_tasks}
            ))
        
        # 检查能耗偏差
        energy_deviation = self._calculate_energy_deviation(agents_state)
        if energy_deviation > self.trigger_thresholds['performance_degradation']:
            triggers.append(TriggerEvent(
                event_type=TriggerType.ENERGY_DEVIATION,
                level=TriggerLevel.DELAYED,
                timestamp=current_time,
                data={'deviation': energy_deviation}
            ))
        
        # 级别3 - 周期性触发
        # 检查性能下降
        if self.performance_history:
            recent_performance = np.mean(self.performance_history[-5:])
            if recent_performance < 0.8:  # 性能低于80%
                triggers.append(TriggerEvent(
                    event_type=TriggerType.PERFORMANCE_DROP,
                    level=TriggerLevel.PERIODIC,
                    timestamp=current_time,
                    data={'performance': recent_performance}
                ))
        
        # 检查负载不均衡
        load_imbalance = self._calculate_load_imbalance(agents_state)
        if load_imbalance > self.trigger_thresholds['load_imbalance']:
            triggers.append(TriggerEvent(
                event_type=TriggerType.LOAD_IMBALANCE,
                level=TriggerLevel.PERIODIC,
                timestamp=current_time,
                data={'imbalance': load_imbalance}
            ))
        
        return triggers
    
    def _process_triggers(self, 
                         triggers: List[TriggerEvent], 
                         agents_state: Dict[str, Dict[str, Any]], 
                         tasks: List[Dict[str, Any]], 
                         environment_state: Dict[str, Any]):
        """处理触发事件"""
        # 按优先级排序触发事件
        immediate_triggers = [t for t in triggers if t.level == TriggerLevel.IMMEDIATE]
        delayed_triggers = [t for t in triggers if t.level == TriggerLevel.DELAYED]
        periodic_triggers = [t for t in triggers if t.level == TriggerLevel.PERIODIC]
        
        # 处理立即触发事件
        if immediate_triggers:
            self.logger.warning(f"立即触发事件: {[t.event_type.value for t in immediate_triggers]}")
            self.event_queue.extend(immediate_triggers)
        
        # 处理延迟触发事件（5分钟内处理）
        if delayed_triggers:
            self.logger.info(f"延迟触发事件: {[t.event_type.value for t in delayed_triggers]}")
            self.event_queue.extend(delayed_triggers)
        
        # 处理周期性触发事件
        if periodic_triggers:
            self.logger.info(f"周期性触发事件: {[t.event_type.value for t in periodic_triggers]}")
            self.event_queue.extend(periodic_triggers)
    
    def _should_perform_periodic_check(self) -> bool:
        """判断是否应该执行周期性检查"""
        current_time = datetime.now()
        time_since_last_check = (current_time - self.last_periodic_check).total_seconds()
        return time_since_last_check >= self.reallocation_interval
    
    def _perform_periodic_check(self, 
                               agents_state: Dict[str, Dict[str, Any]], 
                               tasks: List[Dict[str, Any]], 
                               environment_state: Dict[str, Any]):
        """执行周期性检查"""
        self.logger.info("执行周期性检查")
        self.last_periodic_check = datetime.now()
        
        # 更新性能历史
        current_performance = self._calculate_system_performance(
            agents_state, tasks
        )
        self.performance_history.append(current_performance)
        
        # 限制历史记录长度
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def _should_reallocate(self) -> bool:
        """判断是否应该执行重调度"""
        # 检查是否有待处理的事件
        if not self.event_queue:
            return False
        
        # 检查重调度频率限制
        current_time = datetime.now()
        time_since_last = (current_time - self.last_reallocation_time).total_seconds()
        
        # 如果最近一小时内已达到最大重调度次数
        if time_since_last < 3600 and self.reallocation_count >= self.max_reallocation_frequency:
            self.logger.warning("达到重调度频率限制")
            return False
        
        # 如果有立即触发事件，必须重调度
        immediate_events = [e for e in self.event_queue if e.level == TriggerLevel.IMMEDIATE]
        if immediate_events:
            return True
        
        # 延迟触发事件，等待5分钟
        delayed_events = [e for e in self.event_queue if e.level == TriggerLevel.DELAYED]
        if delayed_events:
            oldest_delayed = min(delayed_events, key=lambda e: e.timestamp)
            if (current_time - oldest_delayed.timestamp).total_seconds() >= 300:
                return True
        
        # 周期性事件，在下一个周期处理
        return False
    
    def _create_reallocation_plan(self, 
                                 agents_state: Dict[str, Dict[str, Any]], 
                                 tasks: List[Dict[str, Any]], 
                                 environment_state: Dict[str, Any]) -> Optional[ReallocationPlan]:
        """创建重调度计划"""
        # 分析当前事件
        critical_agents = set()
        failed_tasks = []
        
        for event in self.event_queue:
            if event.event_type == TriggerType.CRITICAL_ENERGY:
                critical_agents.add(event.agent_id)
            elif event.event_type == TriggerType.TASK_FAILURE:
                failed_tasks.append(event.task_id)
            elif event.event_type == TriggerType.DEVICE_FAILURE:
                critical_agents.add(event.agent_id)
        
        # 计算影响范围
        total_agents = len([a for a in agents_state if agents_state[a].get('type') == 'uav'])
        affected_ratio = len(critical_agents) / total_agents if total_agents > 0 else 0
        
        # 决定局部还是全局重调度
        is_global = affected_ratio >= self.local_adjustment_threshold
        
        if is_global:
            self.logger.info(f"执行全局重调度，影响 {affected_ratio*100:.1f}% 的智能体")
            return self._create_global_reallocation_plan(
                agents_state, tasks, environment_state
            )
        else:
            self.logger.info(f"执行局部调整，影响 {len(critical_agents)} 个智能体")
            return self._create_local_reallocation_plan(
                critical_agents, failed_tasks, agents_state, tasks
            )
    
    def _create_local_reallocation_plan(self, 
                                       critical_agents: Set[str], 
                                       failed_tasks: List[int],
                                       agents_state: Dict[str, Dict[str, Any]], 
                                       tasks: List[Dict[str, Any]]) -> ReallocationPlan:
        """创建局部重调度计划"""
        task_transfers = defaultdict(list)
        affected_agents = critical_agents.copy()
        
        # 为能量危机的UAV转移任务
        for agent_id in critical_agents:
            if agent_id in self.current_assignment:
                agent_tasks = self.current_assignment[agent_id]
                
                # 找到最近的健康UAV
                healthy_uavs = [
                    a for a, s in agents_state.items() 
                    if a != agent_id and s.get('type') == 'uav' 
                    and s.get('energy', 0) / s.get('max_energy', 1) > 0.5
                ]
                
                if healthy_uavs:
                    # 按距离排序
                    agent_pos = agents_state[agent_id]['position']
                    healthy_uavs.sort(
                        key=lambda u: np.linalg.norm(
                            np.array(agents_state[u]['position']) - np.array(agent_pos)
                        )
                    )
                    
                    # 转移任务给最近的健康UAV
                    target_uav = healthy_uavs[0]
                    for task_id in agent_tasks:
                        task_transfers[agent_id].append((task_id, target_uav))
                        affected_agents.add(target_uav)
        
        # 重新分配失败的任务
        for task_id in failed_tasks:
            # 找到可用的UAV
            available_uavs = [
                a for a, s in agents_state.items()
                if s.get('type') == 'uav' and s.get('status') == 'active'
                and s.get('energy', 0) / s.get('max_energy', 1) > 0.3
            ]
            
            if available_uavs:
                # 选择负载最轻的UAV
                target_uav = min(
                    available_uavs, 
                    key=lambda u: len(self.current_assignment.get(u, []))
                )
                task_transfers['failed'].append((task_id, target_uav))
                affected_agents.add(target_uav)
        
        # 生成新的分配方案
        new_assignments = self.current_assignment.copy()
        
        # 执行任务转移
        for from_agent, transfers in task_transfers.items():
            if from_agent != 'failed' and from_agent in new_assignments:
                # 移除原分配
                for task_id, _ in transfers:
                    if task_id in new_assignments[from_agent]:
                        new_assignments[from_agent].remove(task_id)
            
            # 添加新分配
            for task_id, to_agent in transfers:
                if to_agent not in new_assignments:
                    new_assignments[to_agent] = []
                if task_id not in new_assignments[to_agent]:
                    new_assignments[to_agent].append(task_id)
        
        return ReallocationPlan(
            affected_agents=affected_agents,
            task_transfers=dict(task_transfers),
            new_assignments=new_assignments,
            is_global=False,
            transition_steps=self.smooth_transition_steps
        )
    
    def _create_global_reallocation_plan(self, 
                                        agents_state: Dict[str, Dict[str, Any]], 
                                        tasks: List[Dict[str, Any]], 
                                        environment_state: Dict[str, Any]) -> ReallocationPlan:
        """创建全局重调度计划"""
        # 收集所有活跃的UAV
        active_uavs = {
            a: s for a, s in agents_state.items() 
            if s.get('type') == 'uav' and s.get('status') == 'active'
        }
        
        # 收集所有待分配的任务
        pending_tasks = [
            t for t in tasks 
            if t.get('status') in ['pending', 'assigned', 'failed']
        ]
        
        # 使用贪心算法重新分配
        new_assignments = defaultdict(list)
        task_assignments = {}
        
        # 按优先级排序任务
        pending_tasks.sort(key=lambda t: t.get('priority', 1.0), reverse=True)
        
        for task in pending_tasks:
            best_uav = None
            best_score = float('inf')
            
            for uav_id, uav_state in active_uavs.items():
                # 计算分配成本
                cost = self._calculate_assignment_cost(
                    uav_id, uav_state, task, new_assignments[uav_id]
                )
                
                if cost < best_score:
                    best_score = cost
                    best_uav = uav_id
            
            if best_uav:
                new_assignments[best_uav].append(task['task_id'])
                task_assignments[task['task_id']] = best_uav
        
        # 计算任务转移
        task_transfers = defaultdict(list)
        affected_agents = set()
        
        for agent_id, old_tasks in self.current_assignment.items():
            for task_id in old_tasks:
                new_agent = task_assignments.get(task_id)
                if new_agent and new_agent != agent_id:
                    task_transfers[agent_id].append((task_id, new_agent))
                    affected_agents.add(agent_id)
                    affected_agents.add(new_agent)
        
        return ReallocationPlan(
            affected_agents=affected_agents,
            task_transfers=dict(task_transfers),
            new_assignments=dict(new_assignments),
            is_global=True,
            transition_steps=self.smooth_transition_steps
        )
    
    def _execute_reallocation(self, 
                             plan: ReallocationPlan,
                             agents_state: Dict[str, Dict[str, Any]], 
                             tasks: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """执行重调度计划（平滑过渡）"""
        self.logger.info(f"执行重调度计划，影响 {len(plan.affected_agents)} 个智能体")
        
        # 更新重调度记录
        self.last_reallocation_time = datetime.now()
        self.reallocation_count += 1
        
        # 保存历史
        self.assignment_history.append(self.current_assignment.copy())
        if len(self.assignment_history) > 10:
            self.assignment_history = self.assignment_history[-10:]
        
        # 清空事件队列
        self.event_queue.clear()
        
        # 返回新的分配方案
        return plan.new_assignments
    
    def _calculate_assignment_cost(self, 
                                  uav_id: str,
                                  uav_state: Dict[str, Any],
                                  task: Dict[str, Any],
                                  current_tasks: List[int]) -> float:
        """计算分配成本"""
        # 距离成本
        uav_pos = np.array(uav_state['position'])
        task_pos = np.array(task['position'])
        distance = np.linalg.norm(task_pos - uav_pos)
        
        # 能耗成本
        energy_cost = self.energy_calculator.calculate_movement_energy(
            AgentType.UAV, distance, uav_state.get('cruise_speed', 10.0)
        )[0]
        
        # 当前负载
        load_factor = len(current_tasks) + 1
        
        # 能量约束检查
        current_energy = uav_state.get('energy', 0)
        energy_percentage = current_energy / uav_state.get('max_energy', 1)
        
        if energy_percentage < 0.3:
            # 能量不足，增加惩罚
            energy_penalty = 1000
        else:
            energy_penalty = 0
        
        # 综合成本
        total_cost = (
            distance * 0.3 +
            energy_cost * 0.4 +
            load_factor * 100 * 0.2 +
            (1 / task.get('priority', 1.0)) * 100 * 0.1 +
            energy_penalty
        )
        
        return total_cost
    
    def _calculate_energy_efficiency(self,
                                   assignment: Dict[str, List[int]],
                                   agents_state: Dict[str, Dict[str, Any]],
                                   tasks: List[Dict[str, Any]]) -> float:
        """计算能源效率"""
        total_energy_used = 0
        total_tasks_value = 0
        
        task_dict = {t['task_id']: t for t in tasks}
        
        for agent_id, task_ids in assignment.items():
            if agent_id not in agents_state:
                continue
                
            agent_state = agents_state[agent_id]
            agent_pos = np.array(agent_state['position'])
            
            # 计算任务链能耗
            for i, task_id in enumerate(task_ids):
                if task_id not in task_dict:
                    continue
                    
                task = task_dict[task_id]
                task_pos = np.array(task['position'])
                
                # 移动能耗
                if i == 0:
                    distance = np.linalg.norm(task_pos - agent_pos)
                else:
                    prev_task = task_dict.get(task_ids[i-1])
                    if prev_task:
                        prev_pos = np.array(prev_task['position'])
                        distance = np.linalg.norm(task_pos - prev_pos)
                    else:
                        distance = 0
                
                energy_cost = self.energy_calculator.calculate_movement_energy(
                    AgentType.UAV if agent_state.get('type') == 'uav' else AgentType.USV,
                    distance,
                    agent_state.get('cruise_speed', 10.0)
                )[0]
                
                total_energy_used += energy_cost
                total_tasks_value += task.get('priority', 1.0)
        
        # 效率 = 任务价值 / 能耗
        if total_energy_used > 0:
            return total_tasks_value / total_energy_used
        else:
            return 0
    
    def _calculate_load_balance(self,
                               assignment: Dict[str, List[int]],
                               agents_state: Dict[str, Dict[str, Any]]) -> float:
        """计算负载均衡度"""
        uav_loads = []
        
        for agent_id, state in agents_state.items():
            if state.get('type') == 'uav':
                load = len(assignment.get(agent_id, []))
                uav_loads.append(load)
        
        if not uav_loads:
            return 1.0
        
        # 计算标准差
        mean_load = np.mean(uav_loads)
        std_load = np.std(uav_loads)
        
        # 均衡度 = 1 / (1 + 标准差)
        balance_score = 1 / (1 + std_load)
        
        return balance_score
    
    def _calculate_priority_satisfaction(self,
                                       assignment: Dict[str, List[int]],
                                       tasks: List[Dict[str, Any]]) -> float:
        """计算任务优先级满足度"""
        assigned_tasks = set()
        for task_list in assignment.values():
            assigned_tasks.update(task_list)
        
        total_priority = 0
        assigned_priority = 0
        
        for task in tasks:
            task_priority = task.get('priority', 1.0)
            total_priority += task_priority
            
            if task['task_id'] in assigned_tasks:
                assigned_priority += task_priority
        
        if total_priority > 0:
            return assigned_priority / total_priority
        else:
            return 0
    
    def _calculate_stability_score(self, assignment: Dict[str, List[int]]) -> float:
        """计算稳定性分数（减少频繁变动）"""
        if not self.assignment_history:
            return 1.0
        
        # 与上一次分配的相似度
        last_assignment = self.assignment_history[-1]
        
        total_changes = 0
        total_tasks = 0
        
        all_agents = set(assignment.keys()) | set(last_assignment.keys())
        
        for agent_id in all_agents:
            current_tasks = set(assignment.get(agent_id, []))
            previous_tasks = set(last_assignment.get(agent_id, []))
            
            changes = len(current_tasks.symmetric_difference(previous_tasks))
            total_changes += changes
            total_tasks += max(len(current_tasks), len(previous_tasks))
        
        if total_tasks > 0:
            stability = 1 - (total_changes / total_tasks)
        else:
            stability = 1.0
        
        return max(0, stability)
    
    def _calculate_energy_deviation(self, agents_state: Dict[str, Dict[str, Any]]) -> float:
        """计算能耗偏差"""
        uav_energies = []
        
        for agent_id, state in agents_state.items():
            if state.get('type') == 'uav':
                energy_percentage = state.get('energy', 0) / state.get('max_energy', 1)
                uav_energies.append(energy_percentage)
        
        if not uav_energies:
            return 0
        
        # 计算变异系数（标准差/平均值）
        mean_energy = np.mean(uav_energies)
        std_energy = np.std(uav_energies)
        
        if mean_energy > 0:
            return std_energy / mean_energy
        else:
            return 0
    
    def _calculate_load_imbalance(self, agents_state: Dict[str, Dict[str, Any]]) -> float:
        """计算负载不均衡度"""
        uav_task_counts = []
        
        for agent_id, state in agents_state.items():
            if state.get('type') == 'uav':
                task_count = len(self.current_assignment.get(agent_id, []))
                uav_task_counts.append(task_count)
        
        if not uav_task_counts or len(uav_task_counts) < 2:
            return 0
        
        # 计算最大差异比例
        max_tasks = max(uav_task_counts)
        min_tasks = min(uav_task_counts)
        avg_tasks = np.mean(uav_task_counts)
        
        if avg_tasks > 0:
            imbalance = (max_tasks - min_tasks) / avg_tasks
        else:
            imbalance = 0
        
        return imbalance
    
    def _calculate_system_performance(self,
                                    agents_state: Dict[str, Dict[str, Any]],
                                    tasks: List[Dict[str, Any]]) -> float:
        """计算系统整体性能"""
        # 任务完成率
        completed_tasks = len([t for t in tasks if t.get('status') == 'completed'])
        total_tasks = len(tasks)
        completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
        
        # 平均能量水平
        uav_energies = []
        for agent_id, state in agents_state.items():
            if state.get('type') == 'uav':
                energy_percentage = state.get('energy', 0) / state.get('max_energy', 1)
                uav_energies.append(energy_percentage)
        
        avg_energy = np.mean(uav_energies) if uav_energies else 0
        
        # 活跃UAV比例
        active_uavs = len([
            a for a, s in agents_state.items() 
            if s.get('type') == 'uav' and s.get('status') == 'active'
        ])
        total_uavs = len([
            a for a, s in agents_state.items() 
            if s.get('type') == 'uav'
        ])
        active_ratio = active_uavs / total_uavs if total_uavs > 0 else 0
        
        # 综合性能指标
        performance = (
            completion_rate * 0.4 +
            avg_energy * 0.3 +
            active_ratio * 0.3
        )
        
        return performance