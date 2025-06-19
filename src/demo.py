"""
系统演示脚本
展示多UAV-USV协同巡检系统的完整功能
"""
import numpy as np
import time
from typing import Dict, List, Any
import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.env.satellite_scene import SatelliteScene
from src.utils.energy import EnergyCalculator, AgentType
from src.planner.hca_star import HCAStarPlanner, SimpleEnvironment
from src.planner.opt_2opt import TwoOptOptimizer
from src.scheduler.rl_agent import RLScheduler
from src.config.config_loader import load_default_config


class HybridInspectionDemo:
    """混合智能巡检系统演示"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化演示系统
        
        Args:
            config: 系统配置
        """
        self.config = config
        self.setup_components()
        self.setup_scenario()
        
    def setup_components(self):
        """设置系统组件"""
        print("🔧 初始化系统组件...")
        
        # 能源计算器
        self.energy_calculator = EnergyCalculator()
        print("  ✅ 能源管理模块已加载")
        
        # 路径规划器
        self.path_planner = HCAStarPlanner(self.config['planner'])
        print("  ✅ HCA-A*路径规划器已加载")
        
        # 路径优化器
        self.path_optimizer = TwoOptOptimizer(self.config.get('optimizer', {}))
        print("  ✅ 2-opt路径优化器已加载")
        
        # RL调度器
        self.scheduler = RLScheduler(self.config['scheduler'])
        print("  ✅ 强化学习调度器已加载")
        
        # 环境设置
        self.environment = SimpleEnvironment(
            width=1000, 
            height=1000, 
            obstacles=[(200, 200, 300, 300), (600, 600, 700, 700)]
        )
        print("  ✅ 仿真环境已创建")
        
    def setup_scenario(self):
        """设置演示场景"""
        print("\\n🌍 设置演示场景...")
        
        # 智能体状态
        self.agents_state = {
            'uav1': {
                'position': (100.0, 100.0),
                'energy': 280.0,
                'max_energy': 300.0,
                'type': 'uav',
                'status': 'idle'
            },
            'uav2': {
                'position': (800.0, 200.0),
                'energy': 260.0,
                'max_energy': 300.0,
                'type': 'uav',
                'status': 'idle'
            },
            'uav3': {
                'position': (500.0, 100.0),
                'energy': 290.0,
                'max_energy': 300.0,
                'type': 'uav',
                'status': 'idle'
            },
            'usv1': {
                'position': (50.0, 500.0),
                'energy': 950.0,
                'max_energy': 1000.0,
                'type': 'usv',
                'status': 'idle'
            }
        }
        
        # 巡检任务
        self.tasks = [
            {
                'task_id': 0,
                'position': (150.0, 400.0),
                'priority': 2.5,
                'estimated_duration': 120.0,
                'energy_requirement': 25.0,
                'description': '风机A01检测'
            },
            {
                'task_id': 1,
                'position': (400.0, 150.0),
                'priority': 2.0,
                'estimated_duration': 90.0,
                'energy_requirement': 20.0,
                'description': '风机B03检测'
            },
            {
                'task_id': 2,
                'position': (750.0, 400.0),
                'priority': 1.8,
                'estimated_duration': 100.0,
                'energy_requirement': 22.0,
                'description': '风机C05检测'
            },
            {
                'task_id': 3,
                'position': (400.0, 800.0),
                'priority': 1.5,
                'estimated_duration': 80.0,
                'energy_requirement': 18.0,
                'description': '风机D02检测'
            },
            {
                'task_id': 4,
                'position': (100.0, 700.0),
                'priority': 3.0,
                'estimated_duration': 150.0,
                'energy_requirement': 30.0,
                'description': '海洋平台监测'
            },
            {
                'task_id': 5,
                'position': (850.0, 750.0),
                'priority': 1.2,
                'estimated_duration': 60.0,
                'energy_requirement': 15.0,
                'description': '海域巡查'
            }
        ]
        
        # 环境状态
        self.environment_state = {
            'weather': 'clear',
            'visibility': 10000.0,
            'temperature': 25.0
        }
        
        print(f"  📍 智能体数量: {len(self.agents_state)}")
        print(f"  📋 任务数量: {len(self.tasks)}")
        print(f"  🌤️ 天气: {self.environment_state['weather']}")
        
    def display_system_status(self):
        """显示系统状态"""
        print("\\n" + "="*60)
        print("🚁🚢 多UAV-USV协同巡检系统状态")
        print("="*60)
        
        print("\\n📊 智能体状态:")
        for agent_id, state in self.agents_state.items():
            energy_pct = (state['energy'] / state['max_energy']) * 100
            agent_type = "无人机" if state['type'] == 'uav' else "无人船"
            print(f"  {agent_id:6s} | {agent_type} | 位置: {state['position']} | "
                  f"电量: {energy_pct:.1f}% | 状态: {state['status']}")
        
        print("\\n📋 待执行任务:")
        for task in self.tasks:
            print(f"  任务{task['task_id']} | {task['description']} | "
                  f"位置: {task['position']} | 优先级: {task['priority']:.1f}")
        
        print(f"\\n🌍 环境条件:")
        print(f"  天气: {self.environment_state['weather']}")
        print(f"  能见度: {self.environment_state['visibility']}m")
        print(f"  温度: {self.environment_state.get('temperature', 25)}°C")
    
    def run_task_assignment(self):
        """运行任务分配演示"""
        print("\\n" + "="*60)
        print("🧠 智能任务分配")
        print("="*60)
        
        print("\\n正在进行任务分配...")
        start_time = time.time()
        
        # 使用RL调度器进行任务分配
        assignment = self.scheduler.plan(
            self.agents_state,
            self.tasks,
            self.environment_state
        )
        
        assignment_time = time.time() - start_time
        
        print(f"✅ 任务分配完成 (耗时: {assignment_time:.3f}s)")
        
        # 显示分配结果
        print("\\n📋 任务分配结果:")
        for agent_id, task_ids in assignment.items():
            if task_ids:
                agent_type = "无人机" if self.agents_state[agent_id]['type'] == 'uav' else "无人船"
                task_names = [f"任务{tid}" for tid in task_ids]
                print(f"  {agent_id} ({agent_type}): {', '.join(task_names)}")
            else:
                print(f"  {agent_id}: 无分配任务")
        
        # 评估分配质量
        score = self.scheduler.evaluate(assignment, self.agents_state, self.tasks)
        print(f"\\n📈 分配质量评分: {score:.2f}")
        
        return assignment
    
    def run_path_planning(self, assignment: Dict[str, List[int]]):
        """运行路径规划演示"""
        print("\\n" + "="*60)
        print("🗺️ 智能路径规划")
        print("="*60)
        
        path_results = {}
        
        for agent_id, task_ids in assignment.items():
            if not task_ids:
                continue
                
            agent_state = self.agents_state[agent_id]
            agent_type = agent_state['type']
            start_pos = agent_state['position']
            
            print(f"\\n为 {agent_id} 规划路径...")
            
            # 获取任务位置
            task_positions = [self.tasks[tid]['position'] for tid in task_ids]
            
            # 使用HCA-A*规划多目标路径
            start_time = time.time()
            path, total_time, total_energy = self.path_planner.plan_multi_target(
                start_pos, task_positions, agent_type, self.environment
            )
            planning_time = time.time() - start_time
            
            if path:
                print(f"  ✅ 路径规划成功 (耗时: {planning_time:.3f}s)")
                print(f"  📏 路径点数: {len(path)}")
                print(f"  ⏱️ 预计用时: {total_time:.1f}s")
                print(f"  🔋 能耗估算: {total_energy:.1f}Wh")
                
                # 使用2-opt优化路径序列
                if len(task_positions) > 2:
                    print(f"  🔧 正在优化访问序列...")
                    opt_start_time = time.time()
                    
                    optimized_sequence, optimized_cost = self.path_optimizer.optimize_sequence(
                        task_positions, start_pos, agent_type, self.environment
                    )
                    
                    opt_time = time.time() - opt_start_time
                    print(f"  ✅ 序列优化完成 (耗时: {opt_time:.3f}s)")
                    print(f"  📈 优化后任务顺序: {optimized_sequence}")
                
                # 能源充足性检查
                current_energy = agent_state['energy']
                energy_sufficient = self.energy_calculator.is_energy_sufficient(
                    AgentType.UAV if agent_type == 'uav' else AgentType.USV,
                    current_energy,
                    total_energy,
                    safety_margin=0.2
                )
                
                if energy_sufficient:
                    print(f"  ✅ 能源充足，可执行任务")
                else:
                    print(f"  ⚠️ 能源不足，建议充电或重新分配")
                
            else:
                print(f"  ❌ 路径规划失败")
                total_time = float('inf')
                total_energy = float('inf')
            
            path_results[agent_id] = {
                'path': path,
                'time': total_time,
                'energy': total_energy,
                'task_ids': task_ids
            }
        
        return path_results
    
    def run_mission_simulation(self, path_results: Dict[str, Any]):
        """运行任务仿真演示"""
        print("\\n" + "="*60)
        print("⚡ 任务执行仿真")
        print("="*60)
        
        print("\\n🎬 开始仿真执行...")
        
        total_mission_time = 0.0
        total_energy_consumed = 0.0
        completed_tasks = 0
        
        for agent_id, result in path_results.items():
            if result['time'] == float('inf'):
                continue
                
            agent_state = self.agents_state[agent_id]
            agent_type = "无人机" if agent_state['type'] == 'uav' else "无人船"
            
            print(f"\\n{agent_id} ({agent_type}) 执行任务:")
            print(f"  📍 起始位置: {agent_state['position']}")
            print(f"  🎯 任务数量: {len(result['task_ids'])}")
            print(f"  ⏱️ 预计用时: {result['time']:.1f}s ({result['time']/60:.1f}分钟)")
            print(f"  🔋 能耗估算: {result['energy']:.1f}Wh")
            
            # 模拟任务执行
            for task_id in result['task_ids']:
                task = self.tasks[task_id]
                print(f"    执行 {task['description']}...")
                completed_tasks += 1
            
            total_mission_time = max(total_mission_time, result['time'])
            total_energy_consumed += result['energy']
        
        print(f"\\n📊 仿真结果总结:")
        print(f"  ✅ 完成任务数: {completed_tasks}/{len(self.tasks)}")
        print(f"  ⏱️ 总任务时间: {total_mission_time:.1f}s ({total_mission_time/60:.1f}分钟)")
        print(f"  🔋 总能耗: {total_energy_consumed:.1f}Wh")
        print(f"  ⚡ 任务效率: {completed_tasks/max(total_mission_time/60, 1):.2f} 任务/分钟")
    
    def run_performance_analysis(self):
        """运行性能分析"""
        print("\\n" + "="*60)
        print("📈 系统性能分析")
        print("="*60)
        
        print("\\n🔍 分析各模块性能...")
        
        # 能源效率分析
        print("\\n1. 能源效率分析:")
        for agent_id, state in self.agents_state.items():
            agent_type_enum = AgentType.UAV if state['type'] == 'uav' else AgentType.USV
            
            # 计算最大续航
            max_range = self.energy_calculator.estimate_max_range(
                agent_type_enum, state['energy']
            )
            
            # 计算能量百分比
            energy_pct = self.energy_calculator.get_energy_percentage(
                agent_type_enum, state['energy']
            )
            
            print(f"  {agent_id}: 电量 {energy_pct:.1f}%, 最大续航 {max_range/1000:.1f}km")
        
        # 算法性能测试
        print("\\n2. 算法性能测试:")
        
        # 测试HCA-A*性能
        start_time = time.time()
        test_path, _, _ = self.path_planner.plan_path(
            (100, 100), (900, 900), 'uav', self.environment
        )
        hca_time = time.time() - start_time
        print(f"  HCA-A*规划(1km): {hca_time:.3f}s")
        
        # 测试2-opt性能
        test_positions = [(i*100, i*100) for i in range(5)]
        start_time = time.time()
        _, _ = self.path_optimizer.optimize_sequence(
            test_positions, (0, 0), 'uav', self.environment
        )
        opt_time = time.time() - start_time
        print(f"  2-opt优化(5点): {opt_time:.3f}s")
        
        # 系统负载分析
        print("\\n3. 系统负载分析:")
        uav_count = sum(1 for agent in self.agents_state.values() if agent['type'] == 'uav')
        usv_count = sum(1 for agent in self.agents_state.values() if agent['type'] == 'usv')
        
        task_per_agent = len(self.tasks) / len(self.agents_state)
        
        print(f"  UAV数量: {uav_count}, USV数量: {usv_count}")
        print(f"  平均任务负载: {task_per_agent:.1f} 任务/智能体")
        print(f"  系统规模评级: {'小型' if len(self.tasks) < 10 else '中型' if len(self.tasks) < 50 else '大型'}")
        
    def run_complete_demo(self):
        """运行完整演示"""
        print("🚀 启动多UAV-USV协同巡检系统演示")
        print("="*60)
        
        try:
            # 1. 显示初始状态
            self.display_system_status()
            
            # 2. 任务分配
            assignment = self.run_task_assignment()
            
            # 3. 路径规划
            path_results = self.run_path_planning(assignment)
            
            # 4. 任务仿真
            self.run_mission_simulation(path_results)
            
            # 5. 性能分析
            self.run_performance_analysis()
            
            # 6. 总结
            print("\\n" + "="*60)
            print("🎉 演示完成！")
            print("="*60)
            print("\\n系统各模块运行正常，具备以下能力：")
            print("✅ 智能任务分配 - RL调度器")
            print("✅ 代价感知路径规划 - HCA-A*算法")
            print("✅ 路径序列优化 - 2-opt算法")
            print("✅ 精确能耗建模 - 动力学计算")
            print("✅ 多智能体协同 - UAV+USV混合作业")
            print("\\n🔬 系统已准备就绪，可进行进一步的研究和开发！")
            
        except Exception as e:
            print(f"\\n❌ 演示过程中出现错误: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多UAV-USV协同巡检系统演示')
    parser.add_argument('--config', type=str, default='default', 
                       help='配置文件名（默认: default）')
    parser.add_argument('--verbose', action='store_true', 
                       help='详细输出模式')
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = load_default_config()
        
        if args.verbose:
            print(f"🔧 使用配置: {args.config}")
            print(f"📊 系统配置摘要:")
            print(f"  UAV数量: {config['agents']['uav']['count']}")
            print(f"  USV数量: {config['agents']['usv']['count']}")
            print(f"  网格分辨率: {config['planner']['grid_resolution']}m")
            print(f"  学习率: {config['scheduler']['learning_rate']}")
        
        # 创建并运行演示
        demo = HybridInspectionDemo(config)
        demo.run_complete_demo()
        
    except KeyboardInterrupt:
        print("\\n\\n⏹️ 用户中断演示")
    except Exception as e:
        print(f"\\n\\n❌ 演示失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()