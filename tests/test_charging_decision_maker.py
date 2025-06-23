"""
充电决策优化器测试
"""
import pytest
import numpy as np
from typing import Dict, List
import time

from src.scheduler.charging_decision_maker import (
    ChargingDecisionMaker, ChargingOption, ChargingStation, ChargingDecision
)


class TestChargingDecisionMaker:
    """充电决策器测试类"""
    
    @pytest.fixture
    def config(self):
        """测试配置"""
        return {
            'charging_decision': {
                'decision_factors': {
                    'distance': 0.3,
                    'queue_time': 0.25,
                    'availability': 0.25,
                    'urgency': 0.2
                },
                'queue_prediction_window': 900,
                'emergency_threshold': 0.1,
                'charging_station_capacity': 2,
                'usv_charging_rate': 50,
                'station_charging_power': 100
            }
        }
    
    @pytest.fixture
    def decision_maker(self, config):
        """创建决策器实例"""
        return ChargingDecisionMaker(config)
    
    @pytest.fixture
    def charging_stations(self):
        """测试用充电桩"""
        return [
            ChargingStation(
                station_id='station_01',
                position=(300, 400),
                capacity=2,
                queue=['uav_002', 'uav_003']  # 已有排队
            ),
            ChargingStation(
                station_id='station_02',
                position=(700, 600),
                capacity=3,
                queue=[]  # 空闲
            ),
            ChargingStation(
                station_id='station_03',
                position=(100, 100),
                capacity=1,
                queue=['uav_004', 'uav_005', 'uav_006']  # 排队严重
            )
        ]
    
    @pytest.fixture
    def usv_states(self):
        """测试用USV状态"""
        return {
            'usv_001': {
                'position': (600, 450),
                'available_energy': 200,
                'max_energy': 1000,
                'status': 'available'
            },
            'usv_002': {
                'position': (400, 700),
                'available_energy': 50,
                'max_energy': 1000,
                'status': 'available'
            },
            'usv_003': {
                'position': (200, 300),
                'available_energy': 500,
                'max_energy': 1000,
                'status': 'busy'  # 不可用
            }
        }
    
    def test_emergency_decision(self, decision_maker, charging_stations, usv_states):
        """测试紧急情况下的决策"""
        # 极低电量的UAV
        uav_state = {
            'id': 'uav_001',
            'position': (500, 500),
            'current_energy': 15,  # 5%电量
            'max_energy': 300,
            'energy_consumption_rate': 20.0,  # 高能耗
            'current_task_priority': 3.0  # 高优先级任务
        }
        
        system_state = {
            'timestamp': time.time(),
            'total_uavs': 10,
            'charging_uavs': 2,
            'active_tasks': 15,
            'avg_energy_level': 0.5,
            'avg_station_queue': 2.0,
            'available_usvs': 2
        }
        
        decision = decision_maker.make_decision(
            uav_state, charging_stations, usv_states, system_state
        )
        
        # 紧急情况应该选择最快的选项
        assert decision.option in [ChargingOption.CHARGING_STATION, ChargingOption.USV_SUPPORT]
        assert decision.priority_score > 0.5  # 高优先级
        assert decision.factors['urgency'] > 0.8  # 高紧急度
    
    def test_normal_decision(self, decision_maker, charging_stations, usv_states):
        """测试正常情况下的决策"""
        # 中等电量的UAV
        uav_state = {
            'id': 'uav_007',
            'position': (600, 500),
            'current_energy': 120,  # 40%电量
            'max_energy': 300,
            'energy_consumption_rate': 10.0,  # 正常能耗
            'current_task_priority': 1.5  # 中等优先级
        }
        
        system_state = {
            'timestamp': time.time(),
            'total_uavs': 10,
            'charging_uavs': 2,
            'active_tasks': 10,
            'avg_energy_level': 0.6,
            'avg_station_queue': 1.0,
            'available_usvs': 2
        }
        
        decision = decision_maker.make_decision(
            uav_state, charging_stations, usv_states, system_state
        )
        
        # 非紧急情况可能选择不充电
        assert decision.option in [ChargingOption.NONE, ChargingOption.CHARGING_STATION]
        if decision.option != ChargingOption.NONE:
            # 如果决定充电，应该选择空闲的充电桩
            assert decision.target_id == 'station_02'  # 空闲充电桩
    
    def test_distance_factor(self, decision_maker, charging_stations, usv_states):
        """测试距离因子的影响"""
        # UAV靠近某个充电桩
        uav_state = {
            'id': 'uav_008',
            'position': (720, 620),  # 非常靠近station_02
            'current_energy': 60,  # 20%电量
            'max_energy': 300,
            'energy_consumption_rate': 15.0,
            'current_task_priority': 2.0
        }
        
        system_state = {
            'timestamp': time.time(),
            'total_uavs': 10,
            'charging_uavs': 3,
            'active_tasks': 12,
            'avg_energy_level': 0.5,
            'avg_station_queue': 1.5,
            'available_usvs': 2
        }
        
        decision = decision_maker.make_decision(
            uav_state, charging_stations, usv_states, system_state
        )
        
        # 应该选择最近的充电桩
        if decision.option == ChargingOption.CHARGING_STATION:
            assert decision.target_id == 'station_02'
            assert decision.factors['distance'] > 0.9  # 距离很近，因子应该很高
    
    def test_queue_prediction(self, decision_maker, charging_stations, usv_states):
        """测试排队预测功能"""
        # 先添加一些历史数据
        for i in range(10):
            decision_maker.update_history(f'uav_{i:03d}', ChargingOption.CHARGING_STATION, 1800 + i * 60)
        
        # UAV需要充电
        uav_state = {
            'id': 'uav_009',
            'position': (400, 400),
            'current_energy': 45,  # 15%电量
            'max_energy': 300,
            'energy_consumption_rate': 12.0,
            'current_task_priority': 1.8
        }
        
        system_state = {
            'timestamp': time.time(),
            'total_uavs': 10,
            'charging_uavs': 5,
            'active_tasks': 8,
            'avg_energy_level': 0.4,
            'avg_station_queue': 3.0,  # 排队严重
            'available_usvs': 2
        }
        
        decision = decision_maker.make_decision(
            uav_state, charging_stations, usv_states, system_state
        )
        
        # 排队严重时，可能选择USV支援
        if system_state['avg_station_queue'] > 2:
            # 验证队列因子的影响
            if decision.option == ChargingOption.CHARGING_STATION:
                # 选择排队最短的
                assert decision.target_id in ['station_02']  # 空闲充电桩
    
    def test_system_load_adjustment(self, decision_maker, charging_stations, usv_states):
        """测试系统负载动态调整"""
        # 高系统负载场景
        uav_state = {
            'id': 'uav_010',
            'position': (500, 500),
            'current_energy': 75,  # 25%电量
            'max_energy': 300,
            'energy_consumption_rate': 10.0,
            'current_task_priority': 1.5
        }
        
        # 高负载系统状态
        high_load_state = {
            'timestamp': time.time(),
            'total_uavs': 10,
            'charging_uavs': 8,  # 80%在充电
            'active_tasks': 20,
            'avg_energy_level': 0.3,
            'avg_station_queue': 4.0,
            'available_usvs': 1
        }
        
        decision_high = decision_maker.make_decision(
            uav_state, charging_stations, usv_states, high_load_state
        )
        
        # 低负载系统状态
        low_load_state = {
            'timestamp': time.time(),
            'total_uavs': 10,
            'charging_uavs': 2,  # 20%在充电
            'active_tasks': 5,
            'avg_energy_level': 0.7,
            'avg_station_queue': 0.5,
            'available_usvs': 3
        }
        
        decision_low = decision_maker.make_decision(
            uav_state, charging_stations, usv_states, low_load_state
        )
        
        # 高负载时应该更谨慎
        if decision_high.option != ChargingOption.NONE:
            assert decision_high.priority_score != decision_low.priority_score
    
    def test_usv_availability(self, decision_maker, charging_stations, usv_states):
        """测试USV可用性判断"""
        # UAV位置靠近USV
        uav_state = {
            'id': 'uav_011',
            'position': (590, 440),  # 靠近usv_001
            'current_energy': 30,  # 10%电量
            'max_energy': 300,
            'energy_consumption_rate': 18.0,
            'current_task_priority': 2.5
        }
        
        system_state = {
            'timestamp': time.time(),
            'total_uavs': 10,
            'charging_uavs': 4,
            'active_tasks': 10,
            'avg_energy_level': 0.5,
            'avg_station_queue': 2.5,
            'available_usvs': 2
        }
        
        decision = decision_maker.make_decision(
            uav_state, charging_stations, usv_states, system_state
        )
        
        # 不应该选择busy的USV
        if decision.option == ChargingOption.USV_SUPPORT:
            assert decision.target_id != 'usv_003'  # busy状态
            assert decision.target_id in ['usv_001', 'usv_002']  # 可用的USV
    
    def test_statistics_tracking(self, decision_maker):
        """测试统计信息追踪"""
        # 添加一些历史数据
        for i in range(20):
            decision_maker.update_history(
                f'uav_{i:03d}',
                ChargingOption.CHARGING_STATION if i % 2 == 0 else ChargingOption.USV_SUPPORT,
                1500 + i * 100
            )
        
        stats = decision_maker.get_system_statistics()
        
        assert 'system_load' in stats
        assert 'avg_charging_time' in stats
        assert 'queue_history_size' in stats
        assert stats['queue_history_size'] <= 100  # 最大历史记录数
        assert stats['avg_charging_time'] > 0
    
    def test_energy_calculation(self, decision_maker):
        """测试能量计算功能"""
        uav_state = {
            'id': 'uav_012',
            'position': (400, 400),
            'current_energy': 50,
            'max_energy': 300,
            'energy_consumption_rate': 10.0,
            'current_task_priority': 1.0
        }
        
        # 测试紧急度计算
        urgency = decision_maker._calculate_urgency(50/300, uav_state)
        assert 0 <= urgency <= 1
        
        # 低电量应该有高紧急度
        uav_state['current_energy'] = 15
        high_urgency = decision_maker._calculate_urgency(15/300, uav_state)
        assert high_urgency > 0.8
        
        # 高电量应该有低紧急度
        uav_state['current_energy'] = 250
        low_urgency = decision_maker._calculate_urgency(250/300, uav_state)
        assert low_urgency < 0.3


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])