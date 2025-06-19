"""
能源管理工具箱
统一的UAV/USV电量动力学计算模块
"""
import numpy as np
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum


class AgentType(Enum):
    """智能体类型枚举"""
    UAV = "uav"
    USV = "usv"


@dataclass
class EnergyParams:
    """能源参数配置"""
    # 基础能耗参数
    base_power: float  # 基础功率消耗 (W)
    movement_efficiency: float  # 移动效率系数
    max_speed: float  # 最大速度 (m/s)
    
    # 电池参数
    battery_capacity: float  # 电池容量 (Wh)
    voltage: float  # 工作电压 (V)
    
    # 负载系数
    load_factor: float  # 负载系数
    
    # 任务相关
    task_power_consumption: float  # 任务执行时额外功耗 (W)


class EnergyCalculator:
    """能源计算器"""
    
    # 预定义的UAV和USV能源参数
    DEFAULT_PARAMS = {
        AgentType.UAV: EnergyParams(
            base_power=50.0,  # W
            movement_efficiency=0.7,
            max_speed=15.0,  # m/s
            battery_capacity=300.0,  # Wh
            voltage=24.0,  # V
            load_factor=1.0,
            task_power_consumption=20.0  # 拍照、数据采集等
        ),
        AgentType.USV: EnergyParams(
            base_power=100.0,  # W
            movement_efficiency=0.8,
            max_speed=8.0,  # m/s
            battery_capacity=1000.0,  # Wh
            voltage=48.0,  # V
            load_factor=1.2,  # 负载较重
            task_power_consumption=30.0  # 传感器、通信等
        )
    }
    
    def __init__(self, custom_params: Optional[Dict[AgentType, EnergyParams]] = None):
        """
        初始化能源计算器
        
        Args:
            custom_params: 自定义能源参数，覆盖默认值
        """
        self.params = self.DEFAULT_PARAMS.copy()
        if custom_params:
            self.params.update(custom_params)
    
    def calculate_movement_energy(self, 
                                agent_type: AgentType,
                                distance: float,
                                speed: float) -> Tuple[float, float]:
        """
        计算移动消耗的能量和时间
        
        Args:
            agent_type: 智能体类型
            distance: 移动距离 (m)
            speed: 移动速度 (m/s)
            
        Returns:
            (能量消耗 Wh, 移动时间 s)
        """
        params = self.params[agent_type]
        
        # 限制速度不超过最大值
        actual_speed = min(speed, params.max_speed)
        
        # 计算移动时间
        travel_time = distance / actual_speed
        
        # 能耗计算：基础功耗 + 移动功耗
        base_consumption = params.base_power * travel_time / 3600  # 转换为Wh
        
        # 移动功耗与速度平方成正比（克服阻力）
        speed_factor = (actual_speed / params.max_speed) ** 2
        movement_consumption = (params.base_power * speed_factor * 
                              params.load_factor * travel_time) / 3600
        
        total_energy = (base_consumption + movement_consumption) / params.movement_efficiency
        
        return total_energy, travel_time
    
    def calculate_task_energy(self, 
                            agent_type: AgentType,
                            task_duration: float,
                            task_intensity: float = 1.0) -> float:
        """
        计算任务执行的能量消耗
        
        Args:
            agent_type: 智能体类型
            task_duration: 任务持续时间 (s)
            task_intensity: 任务强度系数 (1.0为标准强度)
            
        Returns:
            能量消耗 (Wh)
        """
        params = self.params[agent_type]
        
        # 基础悬停/停留功耗
        base_consumption = params.base_power * task_duration / 3600
        
        # 任务相关功耗
        task_consumption = (params.task_power_consumption * 
                          task_intensity * task_duration) / 3600
        
        return base_consumption + task_consumption
    
    def calculate_remaining_energy(self,
                                 agent_type: AgentType,
                                 current_energy: float,
                                 energy_consumed: float) -> float:
        """
        计算剩余能量
        
        Args:
            agent_type: 智能体类型
            current_energy: 当前能量 (Wh)
            energy_consumed: 消耗的能量 (Wh)
            
        Returns:
            剩余能量 (Wh)
        """
        return max(0.0, current_energy - energy_consumed)
    
    def estimate_max_range(self, 
                          agent_type: AgentType,
                          current_energy: float,
                          speed: float = None) -> float:
        """
        估算最大续航距离
        
        Args:
            agent_type: 智能体类型
            current_energy: 当前能量 (Wh)
            speed: 飞行速度，None则使用经济速度
            
        Returns:
            最大续航距离 (m)
        """
        params = self.params[agent_type]
        
        if speed is None:
            # 使用经济巡航速度（约70%最大速度）
            speed = params.max_speed * 0.7
        
        # 简化计算：假设匀速直线飞行
        speed_factor = (speed / params.max_speed) ** 2
        base_power_per_hour = params.base_power
        movement_power_per_hour = base_power_per_hour * speed_factor * params.load_factor
        
        total_power_per_hour = (base_power_per_hour + movement_power_per_hour) / params.movement_efficiency
        
        # 续航时间（小时）
        flight_time_hours = current_energy / total_power_per_hour
        
        # 最大续航距离
        max_range = speed * flight_time_hours * 3600  # 转换为秒
        
        return max_range
    
    
    def is_energy_sufficient(self,
                           agent_type: AgentType,
                           current_energy: float,
                           planned_energy_consumption: float,
                           safety_margin: float = 0.2) -> bool:
        """
        检查能量是否足够执行计划任务
        
        Args:
            agent_type: 智能体类型
            current_energy: 当前能量 (Wh)
            planned_energy_consumption: 计划消耗能量 (Wh)
            safety_margin: 安全余量比例
            
        Returns:
            能量是否充足
        """
        required_energy = planned_energy_consumption * (1 + safety_margin)
        return current_energy >= required_energy
    
    def get_energy_percentage(self, 
                            agent_type: AgentType,
                            current_energy: float) -> float:
        """
        获取当前能量百分比
        
        Args:
            agent_type: 智能体类型
            current_energy: 当前能量 (Wh)
            
        Returns:
            能量百分比 (0-100)
        """
        max_capacity = self.params[agent_type].battery_capacity
        return (current_energy / max_capacity) * 100
    
    def get_charging_time(self,
                        agent_type: AgentType,
                        current_energy: float,
                        target_energy: float = None,
                        charging_power: float = None) -> float:
        """
        计算充电时间
        
        Args:
            agent_type: 智能体类型
            current_energy: 当前能量 (Wh)
            target_energy: 目标能量，None则充满
            charging_power: 充电功率 (W)，None则使用默认值
            
        Returns:
            充电时间 (s)
        """
        params = self.params[agent_type]
        
        if target_energy is None:
            target_energy = params.battery_capacity
            
        if charging_power is None:
            # 默认充电功率为电池容量的20%（5小时充满）
            charging_power = params.battery_capacity * 0.2
            
        energy_needed = max(0, target_energy - current_energy)
        charging_time = (energy_needed / charging_power) * 3600  # 转换为秒
        
        return charging_time


# 演示用法和测试
if __name__ == "__main__":
    # 创建能源计算器
    energy_calc = EnergyCalculator()
    
    # 测试UAV移动能耗
    print("=== UAV 能耗测试 ===")
    distance = 1000  # 1km
    speed = 10  # 10m/s
    
    energy, time = energy_calc.calculate_movement_energy(
        AgentType.UAV, distance, speed
    )
    print(f"移动1km消耗: {energy:.2f} Wh, 用时: {time:.1f} 秒")
    
    # 测试任务能耗
    task_energy = energy_calc.calculate_task_energy(AgentType.UAV, 60.0)  # 1分钟任务
    print(f"1分钟任务消耗: {task_energy:.2f} Wh")
    
    # 测试续航距离
    current_energy = 250  # Wh
    max_range = energy_calc.estimate_max_range(AgentType.UAV, current_energy)
    print(f"当前能量 {current_energy} Wh 最大续航: {max_range/1000:.1f} km")
    
    print("\\n=== USV 能耗测试 ===")
    energy, time = energy_calc.calculate_movement_energy(
        AgentType.USV, distance, 6.0  # USV较慢
    )
    print(f"USV移动1km消耗: {energy:.2f} Wh, 用时: {time:.1f} 秒")