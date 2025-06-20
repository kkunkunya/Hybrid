"""
可视化实验平台
集成多UAV-USV系统的完整实验环境
"""
import time
import threading
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt

from .enhanced_scene_parser import EnhancedSceneParser as SceneParser
from .visualizer import RealTimeVisualizer, Agent, Task
from ..scheduler.rl_agent import RLScheduler
from ..planner.hca_star import HCAStarPlanner, Environment
from ..planner.opt_2opt import TwoOptOptimizer
from ..utils.energy import EnergyCalculator, AgentType
from ..config.config_loader import load_default_config


class ExperimentPlatform:
    """可视化实验平台"""
    
    def __init__(self, scene_name: str = "兴化湾海上风电场", config: Dict[str, Any] = None):
        """
        初始化实验平台
        
        Args:
            scene_name: 场景名称
            config: 配置参数
        """
        self.scene_name = scene_name
        self.config = config or load_default_config()
        
        # 核心组件
        self.scene_parser = SceneParser(scene_name)
        self.visualizer = None
        self.scheduler = None
        self.path_planner = None
        self.path_optimizer = None
        self.energy_calculator = None
        
        # 实验数据
        self.experiment_data = {
            'agents': {},
            'tasks': {},
            'assignments': {},
            'trajectories': {},
            'performance_metrics': {},
            'timeline': []
        }
        
        # 实验控制
        self.is_running = False
        self.simulation_speed = 1.0
        self.current_step = 0
        self.total_steps = 0
        
    def initialize_platform(self) -> bool:
        """
        初始化实验平台
        
        Returns:
            是否初始化成功
        """
        try:
            # 获取数据目录
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            data_dir = project_root / "data"
            
            print(f"🔧 初始化实验平台...")
            
            # 1. 加载场景
            if not self.scene_parser.load_scene(data_dir):
                print("❌ 场景加载失败")
                return False
            
            # 2. 初始化可视化器
            self.visualizer = RealTimeVisualizer(self.scene_parser)
            if not self.visualizer.setup_visualization():
                print("❌ 可视化器初始化失败")
                return False
            
            # 3. 初始化算法组件
            self._initialize_algorithms()
            
            print("✅ 实验平台初始化完成")
            return True
            
        except Exception as e:
            print(f"❌ 实验平台初始化失败: {e}")
            return False
    
    def _initialize_algorithms(self):
        """初始化算法组件"""
        try:
            # 强化学习调度器
            scheduler_config = self.config.get('scheduler', {})
            self.scheduler = RLScheduler(scheduler_config)
            
            # HCA-A*路径规划器
            planner_config = self.config.get('planner', {})
            self.path_planner = HCAStarPlanner(planner_config)
            
            # 2-opt路径优化器
            optimizer_config = self.config.get('optimizer', {})
            self.path_optimizer = TwoOptOptimizer(optimizer_config)
            
            # 能源计算器
            self.energy_calculator = EnergyCalculator()
            
            print("✅ 算法组件初始化完成")
            
        except Exception as e:
            print(f"❌ 算法组件初始化失败: {e}")
    
    def setup_scenario(self, scenario_type: str = "standard") -> bool:
        """
        设置实验场景
        
        Args:
            scenario_type: 场景类型 ("standard", "large", "complex")
            
        Returns:
            是否设置成功
        """
        try:
            print(f"🌍 设置 {scenario_type} 实验场景...")
            
            # 获取安全位置
            safe_positions = self.scene_parser.get_safe_spawn_positions(10)
            task_positions = self.scene_parser.get_task_positions()
            
            # 根据场景类型配置智能体和任务
            if scenario_type == "standard":
                # 标准场景：3 UAV + 1 USV, 12个任务
                self._setup_agents_standard(safe_positions)
                self._setup_tasks_standard(task_positions[:12])
                
            elif scenario_type == "large":
                # 大规模场景：5 UAV + 2 USV, 25个任务
                self._setup_agents_large(safe_positions)
                self._setup_tasks_large(task_positions[:25])
                
            elif scenario_type == "complex":
                # 复杂场景：4 UAV + 2 USV, 20个任务，包含动态任务
                self._setup_agents_complex(safe_positions)
                self._setup_tasks_complex(task_positions[:20])
            
            print(f"✅ {scenario_type} 场景设置完成")
            print(f"   智能体数量: {len(self.experiment_data['agents'])}")
            print(f"   任务数量: {len(self.experiment_data['tasks'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ 场景设置失败: {e}")
            return False
    
    def _setup_agents_standard(self, safe_positions: List[Tuple[int, int]]):
        """设置标准场景的智能体"""
        agents_config = self.config.get('agents', {})
        
        # 3个UAV
        for i in range(3):
            agent_id = f"uav{i+1}"
            self.visualizer.add_agent(agent_id, "UAV", safe_positions[i])
            
            self.experiment_data['agents'][agent_id] = {
                'type': 'UAV',
                'position': safe_positions[i],
                'battery_capacity': agents_config.get('uav', {}).get('battery_capacity', 300.0),
                'max_speed': agents_config.get('uav', {}).get('max_speed', 15.0),
                'status': 'idle'
            }
        
        # 1个USV  
        agent_id = "usv1"
        self.visualizer.add_agent(agent_id, "USV", safe_positions[3])
        
        self.experiment_data['agents'][agent_id] = {
            'type': 'USV',
            'position': safe_positions[3],
            'battery_capacity': agents_config.get('usv', {}).get('battery_capacity', 1000.0),
            'max_speed': agents_config.get('usv', {}).get('max_speed', 8.0),
            'status': 'idle'
        }
    
    def _setup_agents_large(self, safe_positions: List[Tuple[int, int]]):
        """设置大规模场景的智能体"""
        agents_config = self.config.get('agents', {})
        
        # 5个UAV
        for i in range(5):
            agent_id = f"uav{i+1}"
            self.visualizer.add_agent(agent_id, "UAV", safe_positions[i])
            
            self.experiment_data['agents'][agent_id] = {
                'type': 'UAV',
                'position': safe_positions[i],
                'battery_capacity': agents_config.get('uav', {}).get('battery_capacity', 300.0),
                'max_speed': agents_config.get('uav', {}).get('max_speed', 15.0),
                'status': 'idle'
            }
        
        # 2个USV
        for i in range(2):
            agent_id = f"usv{i+1}"
            self.visualizer.add_agent(agent_id, "USV", safe_positions[5+i])
            
            self.experiment_data['agents'][agent_id] = {
                'type': 'USV',
                'position': safe_positions[5+i],
                'battery_capacity': agents_config.get('usv', {}).get('battery_capacity', 1000.0),
                'max_speed': agents_config.get('usv', {}).get('max_speed', 8.0),
                'status': 'idle'
            }
    
    def _setup_agents_complex(self, safe_positions: List[Tuple[int, int]]):
        """设置复杂场景的智能体"""
        agents_config = self.config.get('agents', {})
        
        # 4个UAV
        for i in range(4):
            agent_id = f"uav{i+1}"
            self.visualizer.add_agent(agent_id, "UAV", safe_positions[i])
            
            self.experiment_data['agents'][agent_id] = {
                'type': 'UAV',
                'position': safe_positions[i],
                'battery_capacity': agents_config.get('uav', {}).get('battery_capacity', 300.0),
                'max_speed': agents_config.get('uav', {}).get('max_speed', 15.0),
                'status': 'idle'
            }
        
        # 2个USV
        for i in range(2):
            agent_id = f"usv{i+1}"
            self.visualizer.add_agent(agent_id, "USV", safe_positions[4+i])
            
            self.experiment_data['agents'][agent_id] = {
                'type': 'USV',
                'position': safe_positions[4+i],
                'battery_capacity': agents_config.get('usv', {}).get('battery_capacity', 1000.0),
                'max_speed': agents_config.get('usv', {}).get('max_speed', 8.0),
                'status': 'idle'
            }
    
    def _setup_tasks_standard(self, task_positions: List[Tuple[int, int]]):
        """设置标准场景的任务"""
        for i, pos in enumerate(task_positions):
            task_id = i
            self.visualizer.add_task(task_id, pos, "wind_turbine_inspection")
            
            self.experiment_data['tasks'][task_id] = {
                'position': pos,
                'type': 'wind_turbine_inspection',
                'priority': np.random.uniform(1.0, 3.0),
                'estimated_duration': np.random.uniform(30.0, 120.0),
                'energy_requirement': np.random.uniform(5.0, 15.0),
                'status': 'pending'
            }
    
    def _setup_tasks_large(self, task_positions: List[Tuple[int, int]]):
        """设置大规模场景的任务"""
        for i, pos in enumerate(task_positions):
            task_id = i
            task_type = "wind_turbine_inspection" if i < 20 else "sea_area_patrol"
            self.visualizer.add_task(task_id, pos, task_type)
            
            self.experiment_data['tasks'][task_id] = {
                'position': pos,
                'type': task_type,
                'priority': np.random.uniform(0.5, 3.0),
                'estimated_duration': np.random.uniform(30.0, 180.0),
                'energy_requirement': np.random.uniform(5.0, 25.0),
                'status': 'pending'
            }
    
    def _setup_tasks_complex(self, task_positions: List[Tuple[int, int]]):
        """设置复杂场景的任务"""
        for i, pos in enumerate(task_positions):
            task_id = i
            
            # 多种任务类型
            if i < 15:
                task_type = "wind_turbine_inspection"
            elif i < 18:
                task_type = "maintenance_check"
            else:
                task_type = "emergency_response"
            
            self.visualizer.add_task(task_id, pos, task_type)
            
            # 紧急任务有更高优先级
            priority = np.random.uniform(2.5, 3.0) if task_type == "emergency_response" else np.random.uniform(1.0, 2.5)
            
            self.experiment_data['tasks'][task_id] = {
                'position': pos,
                'type': task_type,
                'priority': priority,
                'estimated_duration': np.random.uniform(20.0, 150.0),
                'energy_requirement': np.random.uniform(3.0, 20.0),
                'status': 'pending'
            }
    
    def run_task_assignment(self) -> Dict[str, List[int]]:
        """
        运行任务分配
        
        Returns:
            任务分配结果
        """
        try:
            print("🧠 执行智能任务分配...")
            
            # 构建智能体状态
            agents_state = {}
            for agent_id, agent_data in self.experiment_data['agents'].items():
                agents_state[agent_id] = {
                    'position': agent_data['position'],
                    'energy': agent_data['battery_capacity'],  # 假设满电
                    'status': agent_data['status'],
                    'type': agent_data['type']
                }
            
            # 构建任务列表
            tasks = []
            for task_id, task_data in self.experiment_data['tasks'].items():
                tasks.append({
                    'task_id': task_id,
                    'position': task_data['position'],
                    'priority': task_data['priority'],
                    'energy_cost': task_data['energy_requirement']
                })
            
            # 环境状态
            environment_state = {
                'weather': 'clear',
                'visibility': 10000.0,
                'temperature': 25.0
            }
            
            # 执行任务分配
            assignments = self.scheduler.plan(agents_state, tasks, environment_state)
            
            # 更新可视化器
            self.visualizer.assign_tasks(assignments)
            
            # 保存结果
            self.experiment_data['assignments'] = assignments
            
            print(f"✅ 任务分配完成")
            for agent_id, task_ids in assignments.items():
                if task_ids:
                    print(f"   {agent_id}: 任务 {task_ids}")
            
            return assignments
            
        except Exception as e:
            print(f"❌ 任务分配失败: {e}")
            return {}
    
    def simulate_execution(self, duration: float = 300.0, time_step: float = 1.0):
        """
        仿真执行过程
        
        Args:
            duration: 仿真持续时间（秒）
            time_step: 时间步长（秒）
        """
        try:
            print(f"🎬 开始仿真执行 (持续{duration}秒)...")
            
            self.is_running = True
            self.total_steps = int(duration / time_step)
            self.current_step = 0
            
            start_time = time.time()
            
            while self.is_running and self.current_step < self.total_steps:
                current_time = self.current_step * time_step
                
                # 更新智能体状态
                self._update_agents_simulation(current_time)
                
                # 检查任务完成状态
                self._check_task_completion(current_time)
                
                # 记录时间线事件
                self._record_timeline_event(current_time)
                
                # 更新可视化
                if self.visualizer:
                    self.visualizer.current_time = current_time
                
                self.current_step += 1
                time.sleep(time_step / self.simulation_speed)
            
            elapsed_time = time.time() - start_time
            print(f"✅ 仿真执行完成 (实际用时: {elapsed_time:.1f}秒)")
            
            # 生成性能报告
            self._generate_performance_report()
            
        except Exception as e:
            print(f"❌ 仿真执行失败: {e}")
    
    def _update_agents_simulation(self, current_time: float):
        """更新智能体仿真状态"""
        for agent_id, agent_data in self.experiment_data['agents'].items():
            # 简化的移动模拟
            if agent_id in self.experiment_data['assignments']:
                assigned_tasks = self.experiment_data['assignments'][agent_id]
                if assigned_tasks:
                    # 移动到下一个任务位置
                    task_id = assigned_tasks[0]
                    task_pos = self.experiment_data['tasks'][task_id]['position']
                    
                    # 简单的线性移动
                    current_pos = agent_data['position']
                    dx = task_pos[0] - current_pos[0]
                    dy = task_pos[1] - current_pos[1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    if distance > 10:  # 还未到达
                        speed = agent_data['max_speed']
                        move_distance = speed * 1.0  # 1秒移动距离（像素）
                        
                        if move_distance < distance:
                            # 继续移动
                            ratio = move_distance / distance
                            new_x = current_pos[0] + dx * ratio
                            new_y = current_pos[1] + dy * ratio
                            new_pos = (int(new_x), int(new_y))
                            
                            agent_data['position'] = new_pos
                            if self.visualizer:
                                self.visualizer.update_agent_position(agent_id, new_pos, "moving")
    
    def _check_task_completion(self, current_time: float):
        """检查任务完成状态"""
        # 简化的任务完成逻辑
        for task_id, task_data in self.experiment_data['tasks'].items():
            if task_data['status'] == 'pending':
                # 随机完成一些任务（演示用）
                if np.random.random() < 0.01:  # 每秒1%概率完成
                    task_data['status'] = 'completed'
                    if self.visualizer:
                        self.visualizer.complete_task(task_id, current_time)
    
    def _record_timeline_event(self, current_time: float):
        """记录时间线事件"""
        event = {
            'time': current_time,
            'agents_status': {aid: adata['status'] for aid, adata in self.experiment_data['agents'].items()},
            'completed_tasks': sum(1 for t in self.experiment_data['tasks'].values() if t['status'] == 'completed'),
            'total_tasks': len(self.experiment_data['tasks'])
        }
        self.experiment_data['timeline'].append(event)
    
    def _generate_performance_report(self):
        """生成性能报告"""
        try:
            total_tasks = len(self.experiment_data['tasks'])
            completed_tasks = sum(1 for t in self.experiment_data['tasks'].values() if t['status'] == 'completed')
            completion_rate = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
            
            print("\\n📈 实验性能报告")
            print("=" * 50)
            print(f"总任务数: {total_tasks}")
            print(f"完成任务数: {completed_tasks}")
            print(f"完成率: {completion_rate:.1f}%")
            print(f"智能体数量: {len(self.experiment_data['agents'])}")
            print(f"仿真时长: {self.current_step}步")
            
            # 保存性能指标
            self.experiment_data['performance_metrics'] = {
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'completion_rate': completion_rate,
                'simulation_steps': self.current_step,
                'agent_count': len(self.experiment_data['agents'])
            }
            
        except Exception as e:
            print(f"❌ 性能报告生成失败: {e}")
    
    def start_experiment(self, scenario_type: str = "standard", duration: float = 300.0):
        """
        启动完整实验
        
        Args:
            scenario_type: 场景类型
            duration: 实验持续时间
        """
        try:
            print(f"🚀 启动可视化实验: {scenario_type}")
            
            # 1. 初始化平台
            if not self.initialize_platform():
                return False
            
            # 2. 设置场景
            if not self.setup_scenario(scenario_type):
                return False
            
            # 3. 执行任务分配
            assignments = self.run_task_assignment()
            if not assignments:
                print("❌ 任务分配失败，无法继续实验")
                return False
            
            # 4. 启动可视化
            if self.visualizer:
                # 在单独线程中运行仿真
                simulation_thread = threading.Thread(
                    target=self.simulate_execution, 
                    args=(duration,)
                )
                simulation_thread.daemon = True
                simulation_thread.start()
                
                # 启动可视化界面
                self.visualizer.start_animation(interval=100)
            
            return True
            
        except Exception as e:
            print(f"❌ 实验启动失败: {e}")
            return False


def run_demo_experiment():
    """运行演示实验"""
    try:
        # 创建实验平台
        platform = ExperimentPlatform("兴化湾海上风电场")
        
        # 启动标准场景实验
        platform.start_experiment("standard", duration=180.0)
        
    except Exception as e:
        print(f"❌ 演示实验失败: {e}")


if __name__ == "__main__":
    run_demo_experiment()