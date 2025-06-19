"""
调度器抽象基类
定义统一的调度接口，便于算法替换
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np


class BaseScheduler(ABC):
    """调度器抽象基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化调度器
        
        Args:
            config: 配置参数字典
        """
        self.config = config
        
    @abstractmethod
    def plan(self, 
             agents_state: Dict[str, Dict[str, Any]], 
             tasks: List[Dict[str, Any]], 
             environment_state: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        制定任务分配方案
        
        Args:
            agents_state: 所有智能体（UAV/USV）的当前状态
                格式: {agent_id: {position: (x, y), energy: float, status: str}}
            tasks: 待分配的任务列表
                格式: [{task_id: int, position: (x, y), priority: float, energy_cost: float}]
            environment_state: 环境状态（风力、障碍物等）
                
        Returns:
            任务分配结果: {agent_id: [task_id_list]}
        """
        pass
        
    @abstractmethod
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
        pass