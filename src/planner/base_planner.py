"""
路径规划器抽象基类
定义统一的规划接口
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class BasePlanner(ABC):
    """路径规划器抽象基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化规划器
        
        Args:
            config: 配置参数
        """
        self.config = config
    
    @abstractmethod
    def plan_path(self, 
                  start_pos: Tuple[float, float], 
                  target_pos: Tuple[float, float],
                  agent_type: str,
                  environment: 'Environment',
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
        pass
    
    @abstractmethod
    def plan_multi_target(self, 
                         start_pos: Tuple[float, float],
                         targets: List[Tuple[float, float]], 
                         agent_type: str,
                         environment: 'Environment') -> Tuple[List[Tuple[float, float]], float, float]:
        """
        规划多目标访问路径
        
        Args:
            start_pos: 起始位置
            targets: 目标位置列表
            agent_type: 智能体类型
            environment: 环境对象
            
        Returns:
            完整路径, 总时间, 总能量
        """
        pass


class BaseOptimizer(ABC):
    """路径优化器抽象基类"""
    
    @abstractmethod
    def optimize_sequence(self, 
                         positions: List[Tuple[float, float]], 
                         start_pos: Tuple[float, float],
                         agent_type: str,
                         environment: 'Environment') -> Tuple[List[int], float]:
        """
        优化访问序列
        
        Args:
            positions: 待访问位置列表
            start_pos: 起始位置
            agent_type: 智能体类型
            environment: 环境对象
            
        Returns:
            优化后的访问顺序索引, 总成本
        """
        pass