"""
动态重调度管理器单元测试
"""
import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from src.scheduler.dynamic_reallocator import (
    DynamicReallocator, TriggerLevel, TriggerType, 
    TriggerEvent, ReallocationPlan
)


@pytest.fixture
def sample_config():
    """示例配置"""
    return {
        'dynamic_reallocator': {
            'trigger_thresholds': {
                'critical_energy': 0.15,
                'performance_degradation': 0.2,
                'load_imbalance': 0.3,
                'task_failure_rate': 0.1
            },
            'reallocation_interval': 900,  # 15分钟
            'smooth_transition_steps': 3,
            'max_reallocation_frequency': 4,
            'local_adjustment_threshold': 0.3
        }
    }


@pytest.fixture
def sample_agents_state():
    """示例智能体状态"""
    return {
        'uav_1': {
            'type': 'uav',
            'position': (100, 100),
            'energy': 250.0,
            'max_energy': 300.0,
            'status': 'active',
            'cruise_speed': 10.0
        },
        'uav_2': {
            'type': 'uav',
            'position': (200, 200),
            'energy': 40.0,  # 低电量
            'max_energy': 300.0,
            'status': 'active',
            'cruise_speed': 10.0
        },
        'uav_3': {
            'type': 'uav',
            'position': (300, 300),
            'energy': 200.0,
            'max_energy': 300.0,
            'status': 'active',
            'cruise_speed': 10.0
        },
        'usv_1': {
            'type': 'usv',
            'position': (150, 150),
            'energy': 800.0,
            'max_energy': 1000.0,
            'status': 'active',
            'cruise_speed': 6.0
        }
    }


@pytest.fixture
def sample_tasks():
    """示例任务列表"""
    return [
        {
            'task_id': 1,
            'position': (120, 120),
            'priority': 1.5,
            'status': 'assigned',
            'duration': 60,
            'energy_cost': 10.0
        },
        {
            'task_id': 2,
            'position': (180, 180),
            'priority': 2.0,
            'status': 'assigned',
            'duration': 90,
            'energy_cost': 15.0
        },
        {
            'task_id': 3,
            'position': (250, 250),
            'priority': 3.0,  # 高优先级
            'status': 'pending',
            'duration': 120,
            'energy_cost': 20.0
        },
        {
            'task_id': 4,
            'position': (320, 320),
            'priority': 1.0,
            'status': 'failed',  # 失败任务
            'duration': 60,
            'energy_cost': 10.0
        }
    ]


@pytest.fixture
def environment_state():
    """示例环境状态"""
    return {
        'wind_speed': 5.0,
        'wind_direction': 45.0,
        'obstacles': []
    }


@pytest.fixture
def reallocator(sample_config):
    """创建重调度器实例"""
    return DynamicReallocator(sample_config)


class TestDynamicReallocator:
    """动态重调度器测试类"""
    
    def test_init(self, reallocator, sample_config):
        """测试初始化"""
        assert reallocator.trigger_thresholds['critical_energy'] == 0.15
        assert reallocator.reallocation_interval == 900
        assert reallocator.smooth_transition_steps == 3
        assert reallocator.max_reallocation_frequency == 4
        assert reallocator.local_adjustment_threshold == 0.3
        assert len(reallocator.event_queue) == 0
        assert len(reallocator.assignment_history) == 0
    
    def test_check_triggers_critical_energy(self, reallocator, sample_agents_state, sample_tasks, environment_state):
        """测试能量危机触发检查"""
        triggers = reallocator._check_triggers(
            sample_agents_state, sample_tasks, environment_state
        )
        
        # 应该检测到 uav_2 的能量危机
        energy_triggers = [t for t in triggers if t.event_type == TriggerType.CRITICAL_ENERGY]
        assert len(energy_triggers) == 1
        assert energy_triggers[0].agent_id == 'uav_2'
        assert energy_triggers[0].level == TriggerLevel.IMMEDIATE
    
    def test_check_triggers_task_failure(self, reallocator, sample_agents_state, sample_tasks, environment_state):
        """测试任务失败触发检查"""
        triggers = reallocator._check_triggers(
            sample_agents_state, sample_tasks, environment_state
        )
        
        # 应该检测到任务4的失败
        failure_triggers = [t for t in triggers if t.event_type == TriggerType.TASK_FAILURE]
        assert len(failure_triggers) == 1
        assert failure_triggers[0].task_id == 4
        assert failure_triggers[0].level == TriggerLevel.IMMEDIATE
    
    def test_check_triggers_high_priority_task(self, reallocator, sample_agents_state, sample_tasks, environment_state):
        """测试高优先级任务触发检查"""
        triggers = reallocator._check_triggers(
            sample_agents_state, sample_tasks, environment_state
        )
        
        # 应该检测到任务3的高优先级
        priority_triggers = [t for t in triggers if t.event_type == TriggerType.HIGH_PRIORITY_TASK]
        assert len(priority_triggers) == 1
        assert priority_triggers[0].level == TriggerLevel.DELAYED
        assert len(priority_triggers[0].data['tasks']) == 1
        assert priority_triggers[0].data['tasks'][0]['task_id'] == 3
    
    def test_process_triggers(self, reallocator, sample_agents_state, sample_tasks, environment_state):
        """测试触发事件处理"""
        triggers = reallocator._check_triggers(
            sample_agents_state, sample_tasks, environment_state
        )
        
        # 处理触发事件
        reallocator._process_triggers(
            triggers, sample_agents_state, sample_tasks, environment_state
        )
        
        # 检查事件队列
        assert len(reallocator.event_queue) > 0
        
        # 验证事件被正确分类
        immediate_events = [e for e in reallocator.event_queue if e.level == TriggerLevel.IMMEDIATE]
        delayed_events = [e for e in reallocator.event_queue if e.level == TriggerLevel.DELAYED]
        
        assert len(immediate_events) >= 2  # 能量危机和任务失败
        assert len(delayed_events) >= 1    # 高优先级任务
    
    def test_should_reallocate(self, reallocator):
        """测试重调度判断逻辑"""
        # 初始状态，没有事件，不应该重调度
        assert not reallocator._should_reallocate()
        
        # 添加立即触发事件
        event = TriggerEvent(
            event_type=TriggerType.CRITICAL_ENERGY,
            level=TriggerLevel.IMMEDIATE,
            timestamp=datetime.now(),
            agent_id='uav_1'
        )
        reallocator.event_queue.append(event)
        
        # 应该触发重调度
        assert reallocator._should_reallocate()
    
    def test_should_reallocate_frequency_limit(self, reallocator):
        """测试重调度频率限制"""
        # 模拟已达到频率限制
        reallocator.reallocation_count = 4
        reallocator.last_reallocation_time = datetime.now()
        
        # 添加事件
        event = TriggerEvent(
            event_type=TriggerType.CRITICAL_ENERGY,
            level=TriggerLevel.IMMEDIATE,
            timestamp=datetime.now(),
            agent_id='uav_1'
        )
        reallocator.event_queue.append(event)
        
        # 不应该重调度（频率限制）
        assert not reallocator._should_reallocate()
    
    def test_create_local_reallocation_plan(self, reallocator, sample_agents_state, sample_tasks):
        """测试局部重调度计划创建"""
        # 设置当前分配
        reallocator.current_assignment = {
            'uav_1': [1],
            'uav_2': [2],  # 低电量UAV
            'uav_3': [3]
        }
        
        # 创建局部重调度计划
        critical_agents = {'uav_2'}
        failed_tasks = [4]
        
        plan = reallocator._create_local_reallocation_plan(
            critical_agents, failed_tasks, sample_agents_state, sample_tasks
        )
        
        assert isinstance(plan, ReallocationPlan)
        assert not plan.is_global
        assert 'uav_2' in plan.affected_agents
        assert len(plan.task_transfers) > 0
        
        # 验证任务2被转移
        if 'uav_2' in plan.task_transfers:
            transfers = plan.task_transfers['uav_2']
            assert len(transfers) > 0
            assert transfers[0][0] == 2  # 任务ID
    
    def test_create_global_reallocation_plan(self, reallocator, sample_agents_state, sample_tasks, environment_state):
        """测试全局重调度计划创建"""
        # 设置当前分配
        reallocator.current_assignment = {
            'uav_1': [1],
            'uav_2': [2],
            'uav_3': []
        }
        
        plan = reallocator._create_global_reallocation_plan(
            sample_agents_state, sample_tasks, environment_state
        )
        
        assert isinstance(plan, ReallocationPlan)
        assert plan.is_global
        assert len(plan.new_assignments) > 0
        
        # 验证所有待分配任务都被分配
        assigned_tasks = set()
        for tasks in plan.new_assignments.values():
            assigned_tasks.update(tasks)
        
        pending_tasks = {t['task_id'] for t in sample_tasks if t['status'] in ['pending', 'assigned', 'failed']}
        assert len(assigned_tasks) == len(pending_tasks)
    
    def test_calculate_assignment_cost(self, reallocator, sample_agents_state, sample_tasks):
        """测试分配成本计算"""
        uav_state = sample_agents_state['uav_1']
        task = sample_tasks[0]
        current_tasks = []
        
        cost = reallocator._calculate_assignment_cost(
            'uav_1', uav_state, task, current_tasks
        )
        
        assert cost > 0
        assert isinstance(cost, float)
        
        # 测试低电量UAV的成本更高
        low_energy_uav = sample_agents_state['uav_2']
        low_energy_cost = reallocator._calculate_assignment_cost(
            'uav_2', low_energy_uav, task, current_tasks
        )
        
        assert low_energy_cost > cost  # 低电量UAV成本应该更高
    
    def test_calculate_energy_efficiency(self, reallocator, sample_agents_state, sample_tasks):
        """测试能源效率计算"""
        assignment = {
            'uav_1': [1],
            'uav_2': [2],
            'uav_3': [3]
        }
        
        efficiency = reallocator._calculate_energy_efficiency(
            assignment, sample_agents_state, sample_tasks
        )
        
        assert efficiency >= 0
        assert isinstance(efficiency, float)
    
    def test_calculate_load_balance(self, reallocator, sample_agents_state):
        """测试负载均衡计算"""
        # 均衡分配
        balanced_assignment = {
            'uav_1': [1, 2],
            'uav_2': [3, 4],
            'uav_3': [5, 6]
        }
        
        balance_score = reallocator._calculate_load_balance(
            balanced_assignment, sample_agents_state
        )
        
        assert balance_score > 0.8  # 均衡分配应该有高分
        
        # 不均衡分配
        imbalanced_assignment = {
            'uav_1': [1, 2, 3, 4, 5],
            'uav_2': [],
            'uav_3': [6]
        }
        
        imbalance_score = reallocator._calculate_load_balance(
            imbalanced_assignment, sample_agents_state
        )
        
        assert imbalance_score < balance_score  # 不均衡分配分数应该更低
    
    def test_calculate_priority_satisfaction(self, reallocator, sample_tasks):
        """测试优先级满足度计算"""
        # 分配了高优先级任务
        good_assignment = {
            'uav_1': [3],  # 高优先级任务
            'uav_2': [1, 2]
        }
        
        satisfaction = reallocator._calculate_priority_satisfaction(
            good_assignment, sample_tasks
        )
        
        assert satisfaction > 0.5  # 应该有较高的满足度
    
    def test_calculate_stability_score(self, reallocator):
        """测试稳定性分数计算"""
        # 第一次分配，没有历史
        assignment1 = {'uav_1': [1, 2], 'uav_2': [3]}
        score1 = reallocator._calculate_stability_score(assignment1)
        assert score1 == 1.0  # 没有历史时返回1.0
        
        # 添加历史
        reallocator.assignment_history.append(assignment1)
        
        # 相同的分配
        score2 = reallocator._calculate_stability_score(assignment1)
        assert score2 == 1.0  # 完全相同应该是1.0
        
        # 略有变化的分配
        assignment2 = {'uav_1': [1, 3], 'uav_2': [2]}
        score3 = reallocator._calculate_stability_score(assignment2)
        assert 0 < score3 < 1.0  # 有变化但不是完全不同
    
    def test_plan_integration(self, reallocator, sample_agents_state, sample_tasks, environment_state):
        """测试完整的plan方法"""
        # 设置初始分配
        reallocator.current_assignment = {
            'uav_1': [1],
            'uav_2': [2],
            'uav_3': []
        }
        
        # 执行计划
        result = reallocator.plan(
            sample_agents_state, sample_tasks, environment_state
        )
        
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # 验证低电量UAV的任务被重新分配
        assert 'uav_2' not in result or len(result.get('uav_2', [])) == 0
    
    def test_evaluate(self, reallocator, sample_agents_state, sample_tasks):
        """测试评估方法"""
        assignment = {
            'uav_1': [1, 3],
            'uav_2': [],  # 低电量UAV没有任务
            'uav_3': [2]
        }
        
        score = reallocator.evaluate(
            assignment, sample_agents_state, sample_tasks
        )
        
        assert score > 0
        assert isinstance(score, float)
        assert score <= 100  # 最大分数是100
    
    def test_periodic_check(self, reallocator, sample_agents_state, sample_tasks, environment_state):
        """测试周期性检查"""
        # 初始时应该执行周期性检查
        assert reallocator._should_perform_periodic_check()
        
        # 执行周期性检查
        reallocator._perform_periodic_check(
            sample_agents_state, sample_tasks, environment_state
        )
        
        # 立即再次检查应该返回False
        assert not reallocator._should_perform_periodic_check()
        
        # 模拟时间流逝
        reallocator.last_periodic_check = datetime.now() - timedelta(seconds=1000)
        assert reallocator._should_perform_periodic_check()
    
    def test_execute_reallocation(self, reallocator, sample_agents_state, sample_tasks):
        """测试重调度执行"""
        # 创建重调度计划
        plan = ReallocationPlan(
            affected_agents={'uav_1', 'uav_3'},
            task_transfers={'uav_2': [(2, 'uav_1')]},
            new_assignments={'uav_1': [1, 2], 'uav_3': [3]},
            is_global=False,
            transition_steps=3
        )
        
        # 执行重调度
        result = reallocator._execute_reallocation(
            plan, sample_agents_state, sample_tasks
        )
        
        assert result == plan.new_assignments
        assert reallocator.reallocation_count == 1
        assert len(reallocator.event_queue) == 0  # 事件队列应该被清空
        assert len(reallocator.assignment_history) == 1  # 历史记录增加


class TestTriggerEvent:
    """触发事件测试"""
    
    def test_trigger_event_creation(self):
        """测试触发事件创建"""
        event = TriggerEvent(
            event_type=TriggerType.CRITICAL_ENERGY,
            level=TriggerLevel.IMMEDIATE,
            timestamp=datetime.now(),
            agent_id='uav_1',
            data={'energy_percentage': 0.12}
        )
        
        assert event.event_type == TriggerType.CRITICAL_ENERGY
        assert event.level == TriggerLevel.IMMEDIATE
        assert event.agent_id == 'uav_1'
        assert event.data['energy_percentage'] == 0.12


class TestReallocationPlan:
    """重调度计划测试"""
    
    def test_reallocation_plan_creation(self):
        """测试重调度计划创建"""
        plan = ReallocationPlan(
            affected_agents={'uav_1', 'uav_2'},
            task_transfers={'uav_1': [(1, 'uav_2')]},
            new_assignments={'uav_1': [], 'uav_2': [1]},
            is_global=False,
            transition_steps=3
        )
        
        assert len(plan.affected_agents) == 2
        assert 'uav_1' in plan.task_transfers
        assert plan.is_global is False
        assert plan.transition_steps == 3