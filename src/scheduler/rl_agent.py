"""
强化学习调度器
基于深度强化学习的多智能体任务分配决策器
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import random
from collections import deque, namedtuple
import copy

from .base_scheduler import BaseScheduler
from ..utils.energy import EnergyCalculator, AgentType
from ..planner.hca_star import HCAStarPlanner, SimpleEnvironment
from ..planner.opt_2opt import TwoOptOptimizer


# 经验回放缓存
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


@dataclass
class AgentState:
    """智能体状态"""
    agent_id: str
    position: Tuple[float, float]
    energy: float
    max_energy: float
    agent_type: str  # 'uav' or 'usv'
    status: str  # 'idle', 'moving', 'working', 'charging'
    current_task: Optional[int] = None
    task_progress: float = 0.0


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: int
    position: Tuple[float, float]
    priority: float
    estimated_duration: float
    energy_requirement: float
    deadline: Optional[float] = None
    assigned_agent: Optional[str] = None
    status: str = 'pending'  # 'pending', 'assigned', 'in_progress', 'completed'


class DQNNetwork(nn.Module):
    """深度Q网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        """
        初始化DQN网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度  
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
        
        layers = []
        input_dim = state_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(x)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        """
        初始化回放缓冲区
        
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        """添加经验"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """采样经验批次"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)


class RLScheduler(BaseScheduler):
    """强化学习调度器"""
    
    def _ensure_float(self, value):
        """确保值是浮点类型"""
        try:
            return float(value)
        except (ValueError, TypeError):
            raise ValueError(f"无法将 {value} 转换为 float 类型")
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化RL调度器
        
        Args:
            config: 配置参数
        """
        super().__init__(config)
        
        # 网络参数 - 确保类型正确
        self.state_dim = int(config.get('state_dim', 64))
        self.action_dim = int(config.get('action_dim', 100))  # 根据最大任务数调整
        self.hidden_dims = config.get('hidden_dims', [256, 256, 128])
        
        # 训练参数 - 确保类型正确，使用更严格的转换
        self.learning_rate = self._ensure_float(config.get('learning_rate', 1e-4))
        self.gamma = self._ensure_float(config.get('gamma', 0.99))  # 折扣因子
        self.epsilon = self._ensure_float(config.get('epsilon', 0.1))  # 探索概率
        self.epsilon_min = self._ensure_float(config.get('epsilon_min', 0.01))
        self.epsilon_decay = self._ensure_float(config.get('epsilon_decay', 0.995))
        self.target_update_freq = int(config.get('target_update_freq', 100))
        
        # 回放缓冲区
        self.buffer_size = config.get('buffer_size', 10000)
        self.batch_size = config.get('batch_size', 64)
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化网络
        self.q_network = DQNNetwork(self.state_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.target_network = DQNNetwork(self.state_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # 同步目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 辅助工具
        self.energy_calculator = EnergyCalculator()
        self.path_planner = HCAStarPlanner(config.get('planner_config', {}))
        self.path_optimizer = TwoOptOptimizer(config.get('optimizer_config', {}))
        
        # 训练统计
        self.training_step = 0
        self.episode_rewards = []
        
        # 训练模式标志
        self.training = False  # 默认为评估模式
    
    def plan(self, 
             agents_state: Dict[str, Dict[str, Any]], 
             tasks: List[Dict[str, Any]], 
             environment_state: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        制定任务分配方案
        
        Args:
            agents_state: 智能体状态字典
            tasks: 任务列表
            environment_state: 环境状态
            
        Returns:
            任务分配结果: {agent_id: [task_id_list]}
        """
        if not tasks:
            return {agent_id: [] for agent_id in agents_state.keys()}
        
        # 转换为内部数据结构
        agent_states = self._convert_agent_states(agents_state)
        task_infos = self._convert_task_infos(tasks)
        
        # 编码状态
        state_vector = self._encode_state(agent_states, task_infos, environment_state)
        
        # 选择动作（分配策略）
        if self.training and random.random() < self.epsilon:
            # 探索：随机分配
            assignment = self._random_assignment(agent_states, task_infos)
        else:
            # 利用：基于Q网络的贪心分配
            assignment = self._greedy_assignment(state_vector, agent_states, task_infos, environment_state)
        
        # 优化任务序列
        optimized_assignment = self._optimize_task_sequences(assignment, agent_states, task_infos, environment_state)
        
        return optimized_assignment
    
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
        if not assignment or not tasks:
            return 0.0
        
        # 转换数据结构
        agent_states = self._convert_agent_states(agents_state)
        task_infos = self._convert_task_infos(tasks)
        
        total_score = 0.0
        total_tasks = len(tasks)
        
        for agent_id, task_ids in assignment.items():
            if not task_ids or agent_id not in agent_states:
                continue
                
            agent = agent_states[agent_id]
            agent_tasks = [task_infos[tid] for tid in task_ids if tid < len(task_infos)]
            
            if not agent_tasks:
                continue
            
            # 计算执行成本
            execution_cost = self._calculate_execution_cost(agent, agent_tasks)
            
            # 计算任务价值（基于优先级和及时性）
            task_value = sum(task.priority for task in agent_tasks)
            
            # 能源效率评分
            energy_efficiency = self._calculate_energy_efficiency(agent, agent_tasks)
            
            # 负载均衡评分
            load_balance_score = min(len(task_ids) / (total_tasks / len(agent_states)), 1.0)
            
            # 综合评分
            agent_score = (task_value * 0.4 + 
                          energy_efficiency * 0.3 + 
                          load_balance_score * 0.2 - 
                          execution_cost * 0.1)
            
            total_score += agent_score
        
        return total_score / len(agent_states) if agent_states else 0.0
    
    def _convert_agent_states(self, agents_state: Dict[str, Dict[str, Any]]) -> Dict[str, AgentState]:
        """转换智能体状态格式"""
        converted = {}
        for agent_id, state in agents_state.items():
            converted[agent_id] = AgentState(
                agent_id=agent_id,
                position=state['position'],
                energy=state['energy'],
                max_energy=state.get('max_energy', 300.0),
                agent_type=state.get('type', 'uav'),
                status=state.get('status', 'idle'),
                current_task=state.get('current_task'),
                task_progress=state.get('task_progress', 0.0)
            )
        return converted
    
    def _convert_task_infos(self, tasks: List[Dict[str, Any]]) -> List[TaskInfo]:
        """转换任务信息格式"""
        converted = []
        for i, task in enumerate(tasks):
            converted.append(TaskInfo(
                task_id=task.get('task_id', i),
                position=task['position'],
                priority=task.get('priority', 1.0),
                estimated_duration=task.get('estimated_duration', 60.0),
                energy_requirement=task.get('energy_requirement', 10.0),
                deadline=task.get('deadline'),
                status=task.get('status', 'pending')
            ))
        return converted
    
    def _encode_state(self, 
                     agents: Dict[str, AgentState], 
                     tasks: List[TaskInfo], 
                     environment: Dict[str, Any]) -> torch.Tensor:
        """
        编码状态为神经网络输入
        
        Args:
            agents: 智能体状态
            tasks: 任务信息
            environment: 环境状态
            
        Returns:
            状态向量
        """
        state_features = []
        
        # 智能体特征
        for agent in agents.values():
            # 位置特征（归一化）
            norm_x = agent.position[0] / 1000.0  # 假设地图1km范围
            norm_y = agent.position[1] / 1000.0
            
            # 能源特征
            energy_ratio = agent.energy / agent.max_energy
            
            # 类型特征（one-hot）
            is_uav = 1.0 if agent.agent_type == 'uav' else 0.0
            is_usv = 1.0 if agent.agent_type == 'usv' else 0.0
            
            # 状态特征
            is_idle = 1.0 if agent.status == 'idle' else 0.0
            is_working = 1.0 if agent.status == 'working' else 0.0
            
            agent_features = [norm_x, norm_y, energy_ratio, is_uav, is_usv, is_idle, is_working]
            state_features.extend(agent_features)
        
        # 任务特征
        for task in tasks[:10]:  # 限制任务数量避免状态空间过大
            # 位置特征
            norm_x = task.position[0] / 1000.0
            norm_y = task.position[1] / 1000.0
            
            # 优先级和需求
            norm_priority = task.priority / 10.0  # 假设最大优先级为10
            norm_energy = task.energy_requirement / 50.0  # 假设最大需求50Wh
            
            # 状态特征
            is_pending = 1.0 if task.status == 'pending' else 0.0
            is_assigned = 1.0 if task.status == 'assigned' else 0.0
            
            task_features = [norm_x, norm_y, norm_priority, norm_energy, is_pending, is_assigned]
            state_features.extend(task_features)
        
        # 填充到固定长度
        while len(state_features) < self.state_dim:
            state_features.append(0.0)
        
        # 截断到固定长度
        state_features = state_features[:self.state_dim]
        
        return torch.FloatTensor(state_features).to(self.device)
    
    def _greedy_assignment(self, 
                          state: torch.Tensor,
                          agents: Dict[str, AgentState], 
                          tasks: List[TaskInfo],
                          environment: Dict[str, Any]) -> Dict[str, List[int]]:
        """基于Q网络的贪心任务分配"""
        
        # 获取Q值
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0)).squeeze(0)
        
        # 初始化分配结果
        assignment = {agent_id: [] for agent_id in agents.keys()}
        assigned_tasks = set()
        
        agent_list = list(agents.keys())
        
        # 按Q值排序选择动作
        sorted_actions = torch.argsort(q_values, descending=True)
        
        for action_idx in sorted_actions:
            action = action_idx.item()
            
            # 解码动作：action = task_id * num_agents + agent_idx
            if action >= len(tasks) * len(agent_list):
                continue
                
            task_id = action // len(agent_list)
            agent_idx = action % len(agent_list)
            
            if task_id >= len(tasks) or task_id in assigned_tasks:
                continue
                
            agent_id = agent_list[agent_idx]
            agent = agents[agent_id]
            task = tasks[task_id]
            
            # 检查分配可行性
            if self._is_assignment_feasible(agent, task, environment):
                assignment[agent_id].append(task_id)
                assigned_tasks.add(task_id)
        
        return assignment
    
    def _random_assignment(self, 
                          agents: Dict[str, AgentState], 
                          tasks: List[TaskInfo]) -> Dict[str, List[int]]:
        """随机任务分配（用于探索）"""
        assignment = {agent_id: [] for agent_id in agents.keys()}
        agent_list = list(agents.keys())
        
        for task_idx in range(len(tasks)):
            # 随机选择一个智能体
            agent_id = random.choice(agent_list)
            assignment[agent_id].append(task_idx)
        
        return assignment
    
    def _is_assignment_feasible(self, 
                               agent: AgentState, 
                               task: TaskInfo,
                               environment: Dict[str, Any]) -> bool:
        """检查任务分配是否可行"""
        
        # 检查能源是否足够
        distance = np.sqrt((task.position[0] - agent.position[0])**2 + 
                          (task.position[1] - agent.position[1])**2)
        
        agent_type_enum = AgentType.UAV if agent.agent_type == 'uav' else AgentType.USV
        
        # 估算移动能耗
        estimated_energy, _ = self.energy_calculator.calculate_movement_energy(
            agent_type_enum, distance, 10.0  # 假设10m/s速度
        )
        
        # 总能耗需求
        total_energy_needed = estimated_energy + task.energy_requirement
        
        return agent.energy >= total_energy_needed * 1.2  # 20%安全余量
    
    def _optimize_task_sequences(self, 
                                assignment: Dict[str, List[int]],
                                agents: Dict[str, AgentState], 
                                tasks: List[TaskInfo],
                                environment: Dict[str, Any]) -> Dict[str, List[int]]:
        """优化每个智能体的任务执行序列"""
        
        optimized_assignment = {}
        
        # 创建简单环境用于路径规划
        simple_env = SimpleEnvironment(1000, 1000)  # 1km x 1km
        
        for agent_id, task_ids in assignment.items():
            if not task_ids or agent_id not in agents:
                optimized_assignment[agent_id] = task_ids
                continue
                
            agent = agents[agent_id]
            task_positions = [tasks[tid].position for tid in task_ids if tid < len(tasks)]
            
            if len(task_positions) <= 1:
                optimized_assignment[agent_id] = task_ids
                continue
            
            # 使用2-opt优化任务访问顺序
            try:
                optimized_sequence, _ = self.path_optimizer.optimize_sequence(
                    task_positions, agent.position, agent.agent_type, simple_env
                )
                
                # 重新排列任务ID
                optimized_task_ids = [task_ids[i] for i in optimized_sequence]
                optimized_assignment[agent_id] = optimized_task_ids
                
            except Exception as e:
                print(f"序列优化失败 {agent_id}: {e}")
                optimized_assignment[agent_id] = task_ids
        
        return optimized_assignment
    
    def _calculate_execution_cost(self, agent: AgentState, tasks: List[TaskInfo]) -> float:
        """计算执行成本"""
        if not tasks:
            return 0.0
        
        total_distance = 0.0
        current_pos = agent.position
        
        for task in tasks:
            distance = np.sqrt((task.position[0] - current_pos[0])**2 + 
                             (task.position[1] - current_pos[1])**2)
            total_distance += distance
            current_pos = task.position
        
        # 归一化成本
        return total_distance / 1000.0  # 假设最大距离1km
    
    def _calculate_energy_efficiency(self, agent: AgentState, tasks: List[TaskInfo]) -> float:
        """计算能源效率"""
        if not tasks:
            return 1.0
        
        total_energy_needed = sum(task.energy_requirement for task in tasks)
        energy_ratio = agent.energy / agent.max_energy
        
        # 能源充足程度评分
        if total_energy_needed <= agent.energy * 0.8:
            return 1.0
        elif total_energy_needed <= agent.energy:
            return 0.5
        else:
            return 0.0
    
    def train_step(self):
        """执行一步训练"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 采样经验批次
        experiences = self.replay_buffer.sample(self.batch_size)
        
        states = torch.stack([exp.state for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        next_states = torch.stack([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in experiences]).to(self.device)
        
        # 当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新目标网络
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减探索率 - 确保所有参数都是数值类型
        try:
            epsilon_min = self._ensure_float(self.epsilon_min)
            epsilon = self._ensure_float(self.epsilon)
            epsilon_decay = self._ensure_float(self.epsilon_decay)
            self.epsilon = max(epsilon_min, epsilon * epsilon_decay)
        except (ValueError, TypeError) as e:
            print(f"警告：epsilon衰减计算失败: {e}")
            # 使用默认值确保程序继续运行
            self.epsilon = max(0.01, self.epsilon * 0.995)
    
    def set_training_mode(self, training: bool = True):
        """设置训练模式"""
        self.training = training
        if training:
            self.q_network.train()
            self.target_network.train()
        else:
            self.q_network.eval()
            self.target_network.eval()
    
    def eval(self):
        """设置为评估模式"""
        self.set_training_mode(False)
    
    def train(self):
        """设置为训练模式"""
        self.set_training_mode(True)
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'config': self.config
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.epsilon = checkpoint['epsilon']


# 演示用法
if __name__ == "__main__":
    # 配置RL调度器
    config = {
        'state_dim': 64,
        'action_dim': 50,  # 5个智能体 × 10个任务
        'learning_rate': 1e-4,
        'epsilon': 0.1,
        'buffer_size': 5000,
        'batch_size': 32
    }
    
    # 创建调度器
    scheduler = RLScheduler(config)
    
    # 测试数据
    agents_state = {
        'uav1': {'position': (100, 100), 'energy': 250, 'type': 'uav', 'status': 'idle'},
        'uav2': {'position': (200, 200), 'energy': 200, 'type': 'uav', 'status': 'idle'},
        'usv1': {'position': (300, 300), 'energy': 800, 'type': 'usv', 'status': 'idle'}
    }
    
    tasks = [
        {'task_id': 0, 'position': (150, 150), 'priority': 2.0, 'energy_requirement': 15.0},
        {'task_id': 1, 'position': (250, 250), 'priority': 1.5, 'energy_requirement': 12.0},
        {'task_id': 2, 'position': (350, 350), 'priority': 1.0, 'energy_requirement': 20.0},
        {'task_id': 3, 'position': (400, 200), 'priority': 1.8, 'energy_requirement': 10.0}
    ]
    
    environment_state = {'weather': 'clear', 'visibility': 10000.0}
    
    print("=== RL调度器测试 ===")
    
    # 任务分配
    assignment = scheduler.plan(agents_state, tasks, environment_state)
    print(f"任务分配结果: {assignment}")
    
    # 评估分配质量
    score = scheduler.evaluate(assignment, agents_state, tasks)
    print(f"分配质量评分: {score:.2f}")
    
    print("\\nRL调度器框架搭建完成！")