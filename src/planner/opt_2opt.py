"""
2-opt 路径优化器
消除路径交叉，优化访问序列，缩短总路程
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import random
from dataclasses import dataclass

from ..utils.energy import EnergyCalculator, AgentType
from .base_planner import BaseOptimizer
from .hca_star import Environment


@dataclass
class OptimizationResult:
    """优化结果"""
    optimized_sequence: List[int]  # 优化后的访问顺序
    total_cost: float  # 总成本
    improvement: float  # 改进程度
    iterations: int  # 迭代次数


class TwoOptOptimizer(BaseOptimizer):
    """2-opt路径优化器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化2-opt优化器
        
        Args:
            config: 配置参数
        """
        config = config or {}
        self.energy_calculator = EnergyCalculator()
        
        # 优化参数
        self.max_iterations = config.get('max_iterations', 1000)
        self.improvement_threshold = config.get('improvement_threshold', 0.01)  # 1%改进阈值
        self.time_weight = config.get('time_weight', 1.0)
        self.energy_weight = config.get('energy_weight', 1.0)
        self.random_seed = config.get('random_seed', None)
        
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
    
    def optimize_sequence(self, 
                         positions: List[Tuple[float, float]], 
                         start_pos: Tuple[float, float],
                         agent_type: str,
                         environment: Environment) -> Tuple[List[int], float]:
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
        if len(positions) <= 2:
            # 位置太少，无需优化
            return list(range(len(positions))), self._calculate_total_cost(
                positions, start_pos, list(range(len(positions))), agent_type, environment
            )
        
        # 执行2-opt优化
        result = self._two_opt_optimize(positions, start_pos, agent_type, environment)
        
        return result.optimized_sequence, result.total_cost
    
    def optimize_with_details(self, 
                            positions: List[Tuple[float, float]], 
                            start_pos: Tuple[float, float],
                            agent_type: str,
                            environment: Environment,
                            initial_sequence: Optional[List[int]] = None) -> OptimizationResult:
        """
        详细优化过程，返回完整结果
        
        Args:
            positions: 待访问位置列表
            start_pos: 起始位置
            agent_type: 智能体类型
            environment: 环境对象
            initial_sequence: 初始访问序列，None则使用顺序访问
            
        Returns:
            优化结果详情
        """
        return self._two_opt_optimize(positions, start_pos, agent_type, environment, initial_sequence)
    
    def _two_opt_optimize(self, 
                         positions: List[Tuple[float, float]], 
                         start_pos: Tuple[float, float],
                         agent_type: str,
                         environment: Environment,
                         initial_sequence: Optional[List[int]] = None) -> OptimizationResult:
        """
        执行2-opt优化算法
        
        Args:
            positions: 位置列表
            start_pos: 起始位置
            agent_type: 智能体类型
            environment: 环境对象
            initial_sequence: 初始序列
            
        Returns:
            优化结果
        """
        n = len(positions)
        
        # 初始化序列
        if initial_sequence is None:
            current_sequence = list(range(n))
        else:
            current_sequence = initial_sequence.copy()
        
        # 计算初始成本
        initial_cost = self._calculate_total_cost(positions, start_pos, current_sequence, agent_type, environment)
        current_cost = initial_cost
        best_sequence = current_sequence.copy()
        best_cost = current_cost
        
        improved = True
        iterations = 0
        
        while improved and iterations < self.max_iterations:
            improved = False
            iterations += 1
            
            # 尝试所有可能的2-opt交换
            for i in range(n - 1):
                for j in range(i + 2, n):  # j必须至少比i大2，以确保有效的2-opt交换
                    # 执行2-opt交换
                    new_sequence = self._two_opt_swap(current_sequence, i, j)
                    
                    # 计算新序列的成本
                    new_cost = self._calculate_total_cost(positions, start_pos, new_sequence, agent_type, environment)
                    
                    # 如果找到更好的解
                    if new_cost < best_cost:
                        best_sequence = new_sequence.copy()
                        best_cost = new_cost
                        improved = True
                        
                        # 如果改进显著，更新当前解
                        improvement_ratio = (current_cost - new_cost) / current_cost
                        if improvement_ratio > self.improvement_threshold:
                            current_sequence = new_sequence.copy()
                            current_cost = new_cost
        
        # 计算总体改进
        total_improvement = (initial_cost - best_cost) / initial_cost if initial_cost > 0 else 0.0
        
        return OptimizationResult(
            optimized_sequence=best_sequence,
            total_cost=best_cost,
            improvement=total_improvement,
            iterations=iterations
        )
    
    def _two_opt_swap(self, sequence: List[int], i: int, j: int) -> List[int]:
        """
        执行2-opt交换操作
        
        Args:
            sequence: 原序列
            i, j: 交换的边界索引
            
        Returns:
            交换后的新序列
        """
        # 2-opt交换：反转序列中从i+1到j的部分
        new_sequence = sequence.copy()
        new_sequence[i+1:j+1] = reversed(new_sequence[i+1:j+1])
        return new_sequence
    
    def _calculate_total_cost(self, 
                            positions: List[Tuple[float, float]], 
                            start_pos: Tuple[float, float],
                            sequence: List[int],
                            agent_type: str,
                            environment: Environment) -> float:
        """
        计算访问序列的总成本
        
        Args:
            positions: 位置列表
            start_pos: 起始位置
            sequence: 访问序列
            agent_type: 智能体类型
            environment: 环境对象
            
        Returns:
            总成本
        """
        if not sequence:
            return 0.0
        
        total_time = 0.0
        total_energy = 0.0
        
        agent_type_enum = AgentType.UAV if agent_type == 'uav' else AgentType.USV
        max_speed = self.energy_calculator.params[agent_type_enum].max_speed
        cruise_speed = max_speed * 0.7  # 经济巡航速度
        
        current_pos = start_pos
        
        # 计算按序列访问的总成本
        for idx in sequence:
            target_pos = positions[idx]
            
            # 计算移动成本
            distance = np.sqrt((target_pos[0] - current_pos[0])**2 + 
                             (target_pos[1] - current_pos[1])**2)
            
            if distance > 0:
                # 计算能量和时间消耗
                segment_energy, segment_time = self.energy_calculator.calculate_movement_energy(
                    agent_type_enum, distance, cruise_speed
                )
                
                total_time += segment_time
                total_energy += segment_energy
            
            current_pos = target_pos
        
        # 综合成本
        combined_cost = self.time_weight * total_time + self.energy_weight * total_energy
        
        return combined_cost
    
    def visualize_improvement(self, 
                            positions: List[Tuple[float, float]], 
                            start_pos: Tuple[float, float],
                            original_sequence: List[int],
                            optimized_sequence: List[int]) -> Dict[str, Any]:
        """
        可视化优化效果
        
        Args:
            positions: 位置列表
            start_pos: 起始位置
            original_sequence: 原始序列
            optimized_sequence: 优化后序列
            
        Returns:
            可视化数据
        """
        def get_path_points(sequence):
            path = [start_pos]
            for idx in sequence:
                path.append(positions[idx])
            return path
        
        original_path = get_path_points(original_sequence)
        optimized_path = get_path_points(optimized_sequence)
        
        def calculate_path_length(path):
            total_length = 0.0
            for i in range(len(path) - 1):
                total_length += np.sqrt((path[i+1][0] - path[i][0])**2 + 
                                      (path[i+1][1] - path[i][1])**2)
            return total_length
        
        original_length = calculate_path_length(original_path)
        optimized_length = calculate_path_length(optimized_path)
        
        return {
            'original_path': original_path,
            'optimized_path': optimized_path,
            'original_length': original_length,
            'optimized_length': optimized_length,
            'improvement': (original_length - optimized_length) / original_length if original_length > 0 else 0.0,
            'original_sequence': original_sequence,
            'optimized_sequence': optimized_sequence
        }
    
    def batch_optimize(self, 
                      position_sets: List[List[Tuple[float, float]]], 
                      start_positions: List[Tuple[float, float]],
                      agent_types: List[str],
                      environment: Environment) -> List[OptimizationResult]:
        """
        批量优化多个路径
        
        Args:
            position_sets: 位置集合列表
            start_positions: 起始位置列表
            agent_types: 智能体类型列表
            environment: 环境对象
            
        Returns:
            优化结果列表
        """
        results = []
        
        for positions, start_pos, agent_type in zip(position_sets, start_positions, agent_types):
            result = self._two_opt_optimize(positions, start_pos, agent_type, environment)
            results.append(result)
        
        return results


# 高级2-opt变体
class AdaptiveTwoOptOptimizer(TwoOptOptimizer):
    """自适应2-opt优化器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        config = config or {}
        self.adaptive_threshold = config.get('adaptive_threshold', True)
        self.min_improvement = config.get('min_improvement', 0.001)  # 最小改进阈值
    
    def _two_opt_optimize(self, 
                         positions: List[Tuple[float, float]], 
                         start_pos: Tuple[float, float],
                         agent_type: str,
                         environment: Environment,
                         initial_sequence: Optional[List[int]] = None) -> OptimizationResult:
        """
        自适应2-opt优化
        根据问题规模和改进情况动态调整策略
        """
        n = len(positions)
        
        # 根据问题规模调整迭代次数
        if n <= 10:
            max_iter = min(self.max_iterations, 500)
        elif n <= 20:
            max_iter = min(self.max_iterations, 1000)
        else:
            max_iter = self.max_iterations
        
        # 初始化
        if initial_sequence is None:
            current_sequence = list(range(n))
        else:
            current_sequence = initial_sequence.copy()
        
        initial_cost = self._calculate_total_cost(positions, start_pos, current_sequence, agent_type, environment)
        best_sequence = current_sequence.copy()
        best_cost = initial_cost
        
        iterations = 0
        no_improvement_count = 0
        
        while iterations < max_iter and no_improvement_count < 50:  # 连续50次无改进则停止
            found_improvement = False
            iterations += 1
            
            # 随机化搜索顺序，避免局部最优
            indices = [(i, j) for i in range(n-1) for j in range(i+2, n)]
            random.shuffle(indices)
            
            for i, j in indices:
                new_sequence = self._two_opt_swap(current_sequence, i, j)
                new_cost = self._calculate_total_cost(positions, start_pos, new_sequence, agent_type, environment)
                
                if new_cost < best_cost - self.min_improvement:
                    best_sequence = new_sequence.copy()
                    best_cost = new_cost
                    current_sequence = new_sequence.copy()
                    found_improvement = True
                    no_improvement_count = 0
                    break  # 找到改进后立即开始下一轮
            
            if not found_improvement:
                no_improvement_count += 1
        
        total_improvement = (initial_cost - best_cost) / initial_cost if initial_cost > 0 else 0.0
        
        return OptimizationResult(
            optimized_sequence=best_sequence,
            total_cost=best_cost,
            improvement=total_improvement,
            iterations=iterations
        )


# 演示用法
if __name__ == "__main__":
    from .hca_star import SimpleEnvironment
    
    # 创建测试环境
    environment = SimpleEnvironment(1000, 1000)
    
    # 创建测试位置
    np.random.seed(42)
    positions = [(np.random.uniform(100, 900), np.random.uniform(100, 900)) for _ in range(10)]
    start_pos = (50.0, 50.0)
    
    print("=== 2-opt路径优化测试 ===")
    print(f"位置数量: {len(positions)}")
    print(f"起始位置: {start_pos}")
    
    # 基础2-opt优化器
    optimizer = TwoOptOptimizer({'max_iterations': 500})
    
    # 优化序列
    result = optimizer.optimize_with_details(positions, start_pos, 'uav', environment)
    
    print(f"\\n基础2-opt优化结果:")
    print(f"优化序列: {result.optimized_sequence}")
    print(f"总成本: {result.total_cost:.2f}")
    print(f"改进程度: {result.improvement:.2%}")
    print(f"迭代次数: {result.iterations}")
    
    # 自适应2-opt优化器
    adaptive_optimizer = AdaptiveTwoOptOptimizer({'max_iterations': 500})
    adaptive_result = adaptive_optimizer.optimize_with_details(positions, start_pos, 'uav', environment)
    
    print(f"\\n自适应2-opt优化结果:")
    print(f"优化序列: {adaptive_result.optimized_sequence}")
    print(f"总成本: {adaptive_result.total_cost:.2f}")
    print(f"改进程度: {adaptive_result.improvement:.2%}")
    print(f"迭代次数: {adaptive_result.iterations}")
    
    # 可视化比较
    original_sequence = list(range(len(positions)))
    vis_data = optimizer.visualize_improvement(positions, start_pos, original_sequence, result.optimized_sequence)
    
    print(f"\\n路径长度比较:")
    print(f"原始路径长度: {vis_data['original_length']:.1f}m")
    print(f"优化路径长度: {vis_data['optimized_length']:.1f}m")
    print(f"长度改进: {vis_data['improvement']:.2%}")