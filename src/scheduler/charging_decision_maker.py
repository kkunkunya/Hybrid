"""
充电决策优化器（Layer 3）
多因子启发式决策树，用于UAV充电方式选择
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import time

from src.utils.energy import AgentType, EnergyCalculator


class ChargingOption(Enum):
    """充电选项枚举"""
    CHARGING_STATION = "charging_station"  # 充电桩
    USV_SUPPORT = "usv_support"            # USV支援
    NONE = "none"                          # 暂不充电


@dataclass
class ChargingStation:
    """充电桩信息"""
    station_id: str
    position: Tuple[float, float]
    capacity: int = 2  # 同时可充电的UAV数量
    charging_power: float = 100.0  # 充电功率 (W)
    queue: List[str] = None  # 排队的UAV ID列表
    
    def __post_init__(self):
        if self.queue is None:
            self.queue = []


@dataclass
class ChargingDecision:
    """充电决策结果"""
    uav_id: str
    option: ChargingOption
    target_id: Optional[str] = None  # 充电桩ID或USV ID
    estimated_time: float = 0.0       # 预计总时间（到达+排队+充电）
    priority_score: float = 0.0       # 决策优先级分数
    score: float = 0.0               # 综合评分
    factors: Dict[str, float] = None  # 各因子得分明细


class ChargingDecisionMaker:
    """充电决策优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化充电决策器
        
        Args:
            config: 配置参数字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 决策因子权重
        decision_config = config.get('charging_decision', {})
        self.decision_factors = decision_config.get('decision_factors', {
            'distance': 0.3,
            'queue_time': 0.25,
            'availability': 0.25,
            'urgency': 0.2
        })
        
        # 预测参数
        self.queue_prediction_window = decision_config.get('queue_prediction_window', 900)  # 15分钟
        self.emergency_threshold = decision_config.get('emergency_threshold', 0.1)  # 10%
        self.charging_station_capacity = decision_config.get('charging_station_capacity', 2)
        self.usv_charging_rate = decision_config.get('usv_charging_rate', 50)  # W
        
        # 充电桩默认功率
        self.station_charging_power = decision_config.get('station_charging_power', 100)  # W
        
        # 能源计算器
        self.energy_calculator = EnergyCalculator()
        
        # 历史数据队列（用于排队时间预测）
        self.queue_history = deque(maxlen=100)  # 保存最近100条记录
        self.charging_time_history = deque(maxlen=50)  # 充电时间历史
        
        # 系统状态追踪
        self.system_load = 0.0  # 系统负载
        self.avg_queue_length = 0.0  # 平均排队长度
        
    def make_decision(self,
                     uav_state: Dict[str, Any],
                     charging_stations: List[ChargingStation],
                     usv_states: Dict[str, Dict[str, Any]],
                     system_state: Dict[str, Any]) -> ChargingDecision:
        """
        为UAV做出充电决策
        
        Args:
            uav_state: UAV状态信息
                {id, position, current_energy, max_energy, energy_consumption_rate}
            charging_stations: 充电桩列表
            usv_states: USV状态字典
                {usv_id: {position, available_energy, status, current_support}}
            system_state: 系统状态信息
                {timestamp, total_uavs, active_tasks, avg_energy_level}
                
        Returns:
            充电决策结果
        """
        uav_id = uav_state['id']
        uav_pos = uav_state['position']
        current_energy = uav_state['current_energy']
        max_energy = uav_state['max_energy']
        
        # 计算能量百分比和紧急度
        energy_percentage = current_energy / max_energy
        urgency = self._calculate_urgency(energy_percentage, uav_state)
        
        # 决策开始日志
        self.logger.info(
            f"===== 充电决策开始 ====="
            f"\nUAV ID: {uav_id}"
            f"\n当前电量: {current_energy:.1f}Wh ({energy_percentage:.1%})"
            f"\n位置: {uav_pos}"
            f"\n紧急度: {urgency:.2f}"
        )
        
        # 调整决策阈值：40%以下就需要考虑充电
        if urgency < 0.3 and energy_percentage > 0.4:
            self.logger.info(f"UAV {uav_id} 电量充足 ({energy_percentage:.1%})，暂不需要充电")
            return ChargingDecision(
                uav_id=uav_id,
                option=ChargingOption.NONE,
                priority_score=0.0
            )
        
        # 评估所有充电选项
        station_options = self._evaluate_charging_stations(
            uav_state, charging_stations, urgency
        )
        usv_options = self._evaluate_usv_support(
            uav_state, usv_states, urgency
        )
        
        # 选择最佳选项
        best_decision = self._select_best_option(
            station_options, usv_options, urgency
        )
        
        # 动态阈值调整
        best_decision = self._apply_dynamic_threshold(
            best_decision, system_state, urgency
        )
        
        self.logger.info("===== 充电决策结束 =====\n")
        
        return best_decision
    
    def _calculate_urgency(self, 
                          energy_percentage: float,
                          uav_state: Dict[str, Any]) -> float:
        """
        计算充电紧急度
        
        Args:
            energy_percentage: 当前电量百分比
            uav_state: UAV状态
            
        Returns:
            紧急度分数 (0-1)
        """
        # 基础紧急度（基于电量）
        if energy_percentage <= self.emergency_threshold:
            base_urgency = 1.0  # 紧急状态
        elif energy_percentage <= 0.2:
            base_urgency = 0.8
        elif energy_percentage <= 0.3:
            base_urgency = 0.5
        else:
            base_urgency = max(0, (0.5 - energy_percentage) * 2)
        
        # 考虑能耗速率
        consumption_rate = uav_state.get('energy_consumption_rate', 10.0)  # Wh/min
        time_to_empty = (uav_state['current_energy'] / consumption_rate) * 60  # 秒
        
        if time_to_empty < 600:  # 10分钟内耗尽
            rate_factor = 1.0
        elif time_to_empty < 1200:  # 20分钟内耗尽
            rate_factor = 0.7
        else:
            rate_factor = 0.4
        
        # 考虑任务重要性
        task_importance = uav_state.get('current_task_priority', 1.0)
        importance_factor = min(1.0, task_importance / 3.0)  # 假设最高优先级为3
        
        # 综合紧急度
        urgency = base_urgency * 0.5 + rate_factor * 0.3 + importance_factor * 0.2
        
        return min(1.0, urgency)
    
    def _evaluate_charging_stations(self,
                                  uav_state: Dict[str, Any],
                                  charging_stations: List[ChargingStation],
                                  urgency: float) -> List[ChargingDecision]:
        """
        评估所有充电桩选项
        
        Returns:
            充电决策列表
        """
        decisions = []
        uav_pos = uav_state['position']
        
        # 获取当前电量百分比
        energy_percentage = uav_state['current_energy'] / uav_state['max_energy']
        
        for station in charging_stations:
            # 计算距离因子
            distance = np.linalg.norm(
                np.array(uav_pos) - np.array(station.position)
            )
            distance_factor = 1.0 / (1.0 + distance / 1000.0)  # 归一化
            
            # 预测排队时间
            queue_time = self._predict_queue_time(station)
            queue_factor = 1.0 / (1.0 + queue_time / 600.0)  # 10分钟为基准
            
            # 计算可用性因子
            availability = self._calculate_station_availability(station)
            
            # 综合评分
            factors = {
                'distance': distance_factor,
                'queue_time': queue_factor,
                'availability': availability,
                'urgency': urgency
            }
            
            score = sum(
                factors[key] * self.decision_factors[key]
                for key in self.decision_factors
            )
            
            # 根据电量区间调整充电站评分
            if 0.15 < energy_percentage <= 0.25:
                # 15-25%电量：大幅降低充电站评分
                score *= 0.4
                self.logger.info(f"UAV {uav_state['id']} 电量 {energy_percentage:.1%}，降低充电站 {station.station_id} 评分至 {score:.3f}")
            elif 0.25 < energy_percentage <= 0.40:
                # 25-40%电量：适度降低充电站评分
                score *= 0.7
                self.logger.info(f"UAV {uav_state['id']} 电量 {energy_percentage:.1%}，适度降低充电站 {station.station_id} 评分至 {score:.3f}")
            
            # 估算总时间
            travel_time = distance / self.energy_calculator.params[AgentType.UAV].max_speed
            charging_time = self._estimate_charging_time(
                uav_state, self.station_charging_power
            )
            total_time = travel_time + queue_time + charging_time
            
            decisions.append(ChargingDecision(
                uav_id=uav_state['id'],
                option=ChargingOption.CHARGING_STATION,
                target_id=station.station_id,
                estimated_time=total_time,
                priority_score=score,
                score=score,
                factors=factors
            ))
        
        return decisions
    
    def _evaluate_usv_support(self,
                            uav_state: Dict[str, Any],
                            usv_states: Dict[str, Dict[str, Any]],
                            urgency: float) -> List[ChargingDecision]:
        """
        评估所有USV支援选项
        
        Returns:
            充电决策列表
        """
        decisions = []
        uav_pos = uav_state['position']
        
        # 获取当前电量百分比
        energy_percentage = uav_state['current_energy'] / uav_state['max_energy']
        
        for usv_id, usv_state in usv_states.items():
            # 检查USV是否可用（idle状态的USV可以接受新任务）
            usv_status = usv_state.get('status', 'idle')
            if usv_status not in ['idle', 'available']:
                self.logger.debug(f"USV {usv_id} 不可用 (状态: {usv_status})")
                continue
                
            # 检查USV可用充电能量（使用charging_capacity或current_energy）
            available_energy = usv_state.get('charging_capacity', usv_state.get('current_energy', 0))
            if available_energy < 50:  # 最少需要50Wh
                self.logger.debug(f"USV {usv_id} 能量不足 (可用: {available_energy}Wh)")
                continue
            
            # 计算距离因子
            distance = np.linalg.norm(
                np.array(uav_pos) - np.array(usv_state['position'])
            )
            distance_factor = 1.0 / (1.0 + distance / 1000.0)
            
            # USV不需要排队，但可能需要等待到达
            travel_time = distance / self.energy_calculator.params[AgentType.USV].max_speed
            queue_factor = 1.0  # 无排队 - USV的重要优势
            
            # 计算可用性因子（基于USV剩余能量）
            usv_energy_ratio = available_energy / usv_state.get('max_energy', 1000)
            availability = min(1.0, usv_energy_ratio * 2)  # 能量充足时为1
            
            # 综合评分
            factors = {
                'distance': distance_factor,
                'queue_time': queue_factor,
                'availability': availability,
                'urgency': urgency
            }
            
            score = sum(
                factors[key] * self.decision_factors[key]
                for key in self.decision_factors
            )
            
            # 根据电量区间增加USV支援偏好
            if 0.15 < energy_percentage <= 0.25:
                # 15-25%电量：大幅提高USV评分
                score *= 2.5
                self.logger.info(f"UAV {uav_state['id']} 电量 {energy_percentage:.1%}，大幅提高USV {usv_id} 评分至 {score:.3f}")
            elif 0.25 < energy_percentage <= 0.40:
                # 25-40%电量：适度提高USV评分
                score *= 1.5
                self.logger.info(f"UAV {uav_state['id']} 电量 {energy_percentage:.1%}，适度提高USV {usv_id} 评分至 {score:.3f}")
            
            # 额外的USV优势加成（无需排队）
            if queue_factor == 1.0 and 0.15 < energy_percentage <= 0.40:
                score *= 1.2  # 无排队优势额外加成
                self.logger.debug(f"USV {usv_id} 无排队优势，额外加成20%")
            
            # 估算总时间
            charging_time = self._estimate_charging_time(
                uav_state, self.usv_charging_rate
            )
            total_time = travel_time + charging_time
            
            decisions.append(ChargingDecision(
                uav_id=uav_state['id'],
                option=ChargingOption.USV_SUPPORT,
                target_id=usv_id,
                estimated_time=total_time,
                priority_score=score,
                score=score,
                factors=factors
            ))
        
        return decisions
    
    def _predict_queue_time(self, station: ChargingStation) -> float:
        """
        预测充电桩排队时间
        
        Args:
            station: 充电桩
            
        Returns:
            预计排队时间（秒）
        """
        current_queue_length = len(station.queue)
        
        # 基于当前队列长度的简单预测
        if current_queue_length < station.capacity:
            return 0.0  # 无需排队
        
        # 基于历史数据的预测
        if len(self.charging_time_history) > 0:
            avg_charging_time = np.mean(self.charging_time_history)
        else:
            avg_charging_time = 1800  # 默认30分钟
        
        # 计算等待时间
        wait_position = current_queue_length - station.capacity + 1
        predicted_time = wait_position * avg_charging_time / station.capacity
        
        # 考虑趋势（系统负载）
        if self.system_load > 0.8:
            predicted_time *= 1.2  # 高负载时增加20%
        
        return predicted_time
    
    def _calculate_station_availability(self, station: ChargingStation) -> float:
        """
        计算充电桩可用性
        
        Returns:
            可用性分数 (0-1)
        """
        # 基于队列长度
        queue_ratio = len(station.queue) / (station.capacity * 3)  # 3倍容量为满
        availability = max(0, 1 - queue_ratio)
        
        # 考虑充电桩容量
        if station.capacity > 2:
            availability *= 1.1  # 大容量充电桩加分
        
        return min(1.0, availability)
    
    def _estimate_charging_time(self,
                              uav_state: Dict[str, Any],
                              charging_power: float) -> float:
        """
        估算充电时间
        
        Args:
            uav_state: UAV状态
            charging_power: 充电功率 (W)
            
        Returns:
            充电时间（秒）
        """
        energy_needed = uav_state['max_energy'] - uav_state['current_energy']
        charging_time = self.energy_calculator.get_charging_time(
            AgentType.UAV,
            uav_state['current_energy'],
            uav_state['max_energy'],
            charging_power
        )
        
        return charging_time
    
    def _select_best_option(self,
                          station_options: List[ChargingDecision],
                          usv_options: List[ChargingDecision],
                          urgency: float) -> ChargingDecision:
        """
        选择最佳充电选项
        
        Returns:
            最佳充电决策
        """
        all_options = station_options + usv_options
        
        if not all_options:
            # 无可用选项，返回不充电
            self.logger.warning("无可用充电选项")
            return ChargingDecision(
                uav_id="",
                option=ChargingOption.NONE,
                priority_score=0.0
            )
        
        # 记录所有选项
        self.logger.info(f"评估 {len(station_options)} 个充电站和 {len(usv_options)} 个USV选项")
        for option in all_options:
            self.logger.debug(
                f"选项: {option.option.value} - {option.target_id}, "
                f"评分: {option.priority_score:.3f}, 预计时间: {option.estimated_time:.1f}s"
            )
        
        # 按综合得分排序
        all_options.sort(key=lambda x: x.priority_score, reverse=True)
        
        # 紧急情况下（电量低于15%），优先选择时间最短的
        if urgency > 0.8:
            self.logger.warning(f"紧急情况（urgency={urgency:.2f}），优先选择最快到达的选项")
            all_options.sort(key=lambda x: x.estimated_time)
        
        best_option = all_options[0]
        self.logger.info(
            f"最终决策: {best_option.option.value} - {best_option.target_id}, "
            f"评分: {best_option.priority_score:.3f}, 预计时间: {best_option.estimated_time:.1f}s"
        )
        
        return best_option
    
    def _apply_dynamic_threshold(self,
                               decision: ChargingDecision,
                               system_state: Dict[str, Any],
                               urgency: float) -> ChargingDecision:
        """
        应用动态阈值调整
        
        Args:
            decision: 初始决策
            system_state: 系统状态
            urgency: 紧急度
            
        Returns:
            调整后的决策
        """
        # 更新系统负载
        total_uavs = system_state.get('total_uavs', 1)
        charging_uavs = system_state.get('charging_uavs', 0)
        self.system_load = charging_uavs / total_uavs if total_uavs > 0 else 0
        
        # 系统繁忙时的调整（减少对USV的惩罚）
        if self.system_load > 0.7 and decision.option == ChargingOption.USV_SUPPORT:
            # 仅在非常不紧急的情况下才轻微降低USV优先级
            if urgency < 0.3:  # 只有非常不紧急时
                decision.priority_score *= 0.95  # 轻微降低（原来是0.8）
                self.logger.debug(f"系统负载高 ({self.system_load:.2f})，轻微降低USV优先级")
        
        # 充电桩排队严重时的调整（增强USV优势）
        avg_queue = system_state.get('avg_station_queue', 0)
        if avg_queue > 2 and decision.option == ChargingOption.USV_SUPPORT:
            # 充电桩排队严重时，大幅提高USV权重
            decision.priority_score *= 1.3
            self.logger.info(f"充电桩平均排队 {avg_queue:.1f}，提高USV {decision.target_id} 优先级")
        elif avg_queue > 3 and decision.option == ChargingOption.CHARGING_STATION:
            # 降低充电桩权重
            decision.priority_score *= 0.7
            self.logger.info(f"充电桩排队严重，降低充电站 {decision.target_id} 优先级")
        
        # USV数量不足时的调整（只在极端情况下调整）
        available_usvs = system_state.get('available_usvs', 0)
        if available_usvs < 1 and decision.option == ChargingOption.USV_SUPPORT:
            # 只有在完全没有可用USV时才显著降低优先级
            decision.priority_score *= 0.8  # 原来是0.7
            self.logger.warning("无可用USV，降低USV支援优先级")
        
        return decision
    
    def update_history(self,
                      uav_id: str,
                      charging_option: ChargingOption,
                      actual_time: float):
        """
        更新历史数据（用于改进预测）
        
        Args:
            uav_id: UAV ID
            charging_option: 实际选择的充电方式
            actual_time: 实际耗时
        """
        if charging_option == ChargingOption.CHARGING_STATION:
            self.charging_time_history.append(actual_time)
        
        # 更新排队历史
        self.queue_history.append({
            'timestamp': time.time(),
            'option': charging_option,
            'time': actual_time
        })
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        获取系统统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'system_load': self.system_load,
            'avg_queue_length': self.avg_queue_length,
            'avg_charging_time': np.mean(self.charging_time_history) if self.charging_time_history else 0,
            'queue_history_size': len(self.queue_history),
            'decision_factors': self.decision_factors
        }
        
        return stats


# 测试和演示
if __name__ == "__main__":
    # 创建测试配置
    test_config = {
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
    
    # 创建决策器
    decision_maker = ChargingDecisionMaker(test_config)
    
    # 测试UAV状态
    uav_state = {
        'id': 'uav_001',
        'position': (500, 500),
        'current_energy': 45,  # 15%电量
        'max_energy': 300,
        'energy_consumption_rate': 15.0,  # Wh/min
        'current_task_priority': 2.5
    }
    
    # 测试充电桩
    charging_stations = [
        ChargingStation(
            station_id='station_01',
            position=(300, 400),
            capacity=2,
            queue=['uav_002', 'uav_003']
        ),
        ChargingStation(
            station_id='station_02',
            position=(700, 600),
            capacity=3,
            queue=[]
        )
    ]
    
    # 测试USV状态
    usv_states = {
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
        }
    }
    
    # 系统状态
    system_state = {
        'timestamp': time.time(),
        'total_uavs': 10,
        'charging_uavs': 3,
        'active_tasks': 15,
        'avg_energy_level': 0.45,
        'avg_station_queue': 1.5,
        'available_usvs': 2
    }
    
    # 做出决策
    decision = decision_maker.make_decision(
        uav_state, charging_stations, usv_states, system_state
    )
    
    # 打印结果
    print("=== 充电决策结果 ===")
    print(f"UAV ID: {decision.uav_id}")
    print(f"充电选项: {decision.option.value}")
    print(f"目标ID: {decision.target_id}")
    print(f"预计总时间: {decision.estimated_time:.1f} 秒")
    print(f"优先级分数: {decision.priority_score:.3f}")
    print("\n决策因子得分:")
    if decision.factors:
        for factor, score in decision.factors.items():
            print(f"  {factor}: {score:.3f}")
    
    # 获取系统统计
    stats = decision_maker.get_system_statistics()
    print("\n=== 系统统计信息 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")