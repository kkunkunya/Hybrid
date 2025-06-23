#!/usr/bin/env python3
"""
集成调度器可视化演示
展示新的4层优化架构调度系统的效果
"""
import sys
import time
from pathlib import Path
import threading
import queue
import logging
from datetime import datetime
import io
import math

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置日志系统
class TeeLogger:
    """同时输出到控制台和日志文件的日志类"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

# 创建logs文件夹（如果不存在）
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)

# 生成日志文件名（基于时间戳）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = logs_dir / f"integrated_scheduler_demo_{timestamp}.log"

# 设置双向输出
tee_logger = TeeLogger(log_filename)
sys.stdout = tee_logger
sys.stderr = tee_logger

# 配置logging模块
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.__stdout__)
    ]
)

print(f"📝 日志文件: {log_filename}")
print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

def main():
    """主函数 - 集成调度器演示"""
    print("🎮 多UAV-USV协同巡检系统 - 集成调度器演示")
    print("=" * 60)
    
    # 显示新特性
    print("\n✨ 集成调度器特性:")
    print("📋 Layer 1: 能源感知任务分配 - 多任务贪心算法")
    print("🚢 Layer 2: USV后勤智能调度 - 动态支援决策")
    print("🔋 Layer 3: 充电决策优化 - 多因子评分模型")
    print("🔄 Layer 4: 动态重调度管理 - 事件驱动机制")
    
    # 选择场景
    print("\n📍 选择场景:")
    print("1. xinghua_bay_wind_farm (默认)")
    print("2. fuqing_haitan_wind_farm")
    print("3. pingtan_wind_farm")
    print("4. putian_nanri_wind_farm_phase1_2")
    
    scene_map = {
        '1': 'xinghua_bay_wind_farm',
        '2': 'fuqing_haitan_wind_farm',
        '3': 'pingtan_wind_farm',
        '4': 'putian_nanri_wind_farm_phase1_2'
    }
    
    try:
        choice = input("\n请选择 (1-4，默认1): ").strip() or '1'
    except EOFError:
        choice = '1'  # 默认选择场景1
        print("1")  # 显示默认选择
    scene_name = scene_map.get(choice, 'xinghua_bay_wind_farm')
    
    try:
        # 检查pygame依赖
        try:
            import pygame
            print(f"\n✅ Pygame版本: {pygame.version.ver}")
        except ImportError:
            print("❌ Pygame未安装")
            print("请运行: pip install pygame>=2.5.0")
            return False
        
        # 导入模块
        from src.visualization.enhanced_scene_parser import EnhancedSceneParser
        from src.visualization.final_enhanced_visualizer import FinalEnhancedPygameVisualizer
        from src.scheduler import IntegratedScheduler  # 使用新的集成调度器
        from src.config.config_loader import load_default_config
        from src.visualization.charging_station import ChargingStation
        
        # 加载场景
        data_dir = project_root / "data"
        parser = EnhancedSceneParser(scene_name)
        
        if not parser.load_scene(data_dir):
            print("❌ 场景加载失败")
            return False
        
        print("✅ 场景加载成功")
        
        # 创建可视化器
        visualizer = FinalEnhancedPygameVisualizer(parser, (1400, 1000))
        
        # 手动设置必要组件
        print("🔧 初始化可视化器组件...")
        
        # 初始化pygame
        pygame.init()
        visualizer.screen = pygame.display.set_mode(visualizer.window_size, pygame.RESIZABLE)
        pygame.display.set_caption("多UAV-USV协同巡检系统 - 集成调度器演示")
        
        # 设置字体
        try:
            visualizer.font_small = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 18)
            visualizer.font_medium = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 22)
            visualizer.font_large = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 26)
        except:
            visualizer.font_small = pygame.font.Font(None, 18)
            visualizer.font_medium = pygame.font.Font(None, 22)
            visualizer.font_large = pygame.font.Font(None, 26)
        
        # 设置缩放因子
        visualizer.scale_x = visualizer.window_size[0] / parser.image_size[0]
        visualizer.scale_y = visualizer.window_size[1] / parser.image_size[1]
        
        # 设置背景
        if parser.image is not None:
            visualizer.background_surface = pygame.transform.scale(
                pygame.surfarray.make_surface(parser.image.swapaxes(0, 1)),
                visualizer.window_size
            )
        else:
            visualizer.background_surface = pygame.Surface(visualizer.window_size)
            visualizer.background_surface.fill((100, 149, 237))
        
        # 创建充电桩
        fixed_charging_positions = [(120, 400), (350, 50), (800, 650)]
        visualizer.charging_stations = []
        for i, pos in enumerate(fixed_charging_positions):
            station = ChargingStation(i, pos, "shore")
            visualizer.charging_stations.append(station)
        
        print(f"✅ 部署了 {len(visualizer.charging_stations)} 个充电桩")
        
        # 设置智能体位置
        fixed_agent_positions = [(150, 350), (200, 400), (250, 450), (300, 500), (350, 550)]
        
        print("\n📋 智能体配置:")
        # 添加UAV（所有UAV从100%电量开始）
        uav_configs = [
            {'id': 'uav1', 'battery': 100},  # 满电量开始
            {'id': 'uav2', 'battery': 100},  # 满电量开始  
            {'id': 'uav3', 'battery': 100},  # 满电量开始
        ]
        
        for i, config in enumerate(uav_configs):
            agent = visualizer.add_agent(config['id'], "UAV", fixed_agent_positions[i])
            agent.battery_level = config['battery']
            print(f"  {config['id']}: 电量 {config['battery']}%")
        
        # 添加USV
        usv_count = 2
        for i in range(usv_count):
            visualizer.add_agent(f"usv{i+1}", "USV", fixed_agent_positions[i + len(uav_configs)])
            print(f"  usv{i+1}: 后勤支援船")
        
        # 添加任务 - 为所有风机创建任务
        task_positions = parser.get_task_positions()
        num_tasks = len(task_positions)  # 使用所有风机，不限制数量
        
        print(f"\n📋 创建任务...")
        for i in range(num_tasks):
            visualizer.add_task(i, task_positions[i], "wind_turbine_inspection")
        
        print(f"✅ 添加巡检任务: {num_tasks}个（所有风机）")
        
        # 🧠 使用集成调度器
        print("\n🧠 启动4层集成调度系统...")
        
        # 加载配置
        config = load_default_config()
        
        # 调整配置以展示更多功能
        config['energy_aware_allocator']['energy_threshold'] = 0.10  # 10%以下不分配任务
        config['usv_logistics']['min_reserve_energy_ratio'] = 0.3
        config['usv_logistics']['emergency_energy_threshold'] = 0.15  # 15%触发紧急支援
        config['charging_decision']['emergency_threshold'] = 0.15
        config['dynamic_reallocator']['critical_energy'] = 0.15
        
        scheduler = IntegratedScheduler(config)
        
        # 构建智能体状态信息
        agents_state = {}
        for agent_id, agent in visualizer.agents.items():
            # 使用兼容性属性
            current_energy = agent.energy
            max_energy = agent.max_energy
            
            agents_state[agent_id] = {
                'position': agent.position,
                'energy': current_energy,
                'max_energy': max_energy,
                'type': agent.agent_type.lower(),
                'status': agent.status,
                'cruise_speed': agent.speed
            }
            
            # USV额外属性
            if agent.agent_type.lower() == 'usv':
                agents_state[agent_id]['charging_capacity'] = 200
                agents_state[agent_id]['charging_rate'] = 50.0
        
        # 构建任务列表 - 为所有风机创建任务数据
        tasks = []
        for i in range(num_tasks):
            # 根据位置设置优先级（离基地越远优先级越高）
            base_pos = (150, 350)  # UAV基地位置
            distance_from_base = ((task_positions[i][0] - base_pos[0])**2 + 
                                (task_positions[i][1] - base_pos[1])**2)**0.5
            priority = 1.0 + min(2.0, distance_from_base / 500.0)  # 距离越远优先级越高
            
            tasks.append({
                'task_id': i,
                'position': task_positions[i],
                'priority': priority,
                'estimated_duration': 60,  # 统一巡检时间60秒
                'energy_requirement': 10.0  # 统一能量需求
            })
        
        # 环境状态（包含充电站信息）
        environment_state = {
            'weather': 'clear',
            'visibility': 10000.0,
            'temperature': 25.0,
            'charging_stations': [
                {
                    'id': i,
                    'position': station.position,
                    'capacity': 4,
                    'queue_length': 0
                }
                for i, station in enumerate(visualizer.charging_stations)
            ]
        }
        
        # 执行集成调度
        print("\n🔍 开始执行集成调度...")
        print(f"  UAV数量: {len([a for a in agents_state.values() if a['type'] == 'uav'])}")
        print(f"  USV数量: {len([a for a in agents_state.values() if a['type'] == 'usv'])}")
        print(f"  任务数量: {len(tasks)}")
        print(f"  充电站数量: {len(environment_state['charging_stations'])}")
        
        # 显示每个智能体的初始状态
        print("\n📋 智能体初始状态:")
        for agent_id, state in agents_state.items():
            agent_type = state['type'].upper()
            energy_pct = (state['energy'] / state['max_energy']) * 100
            print(f"  {agent_id} ({agent_type}): 位置={state['position']}, "
                  f"电量={energy_pct:.1f}% ({state['energy']:.1f}/{state['max_energy']:.1f}Wh), "
                  f"状态={state['status']}")
        
        result = scheduler.schedule(agents_state, tasks, environment_state)
        
        print("\n📊 调度结果:")
        
        # 显示任务分配
        print("\n📋 Layer 1 - 任务分配:")
        for agent_id, task_ids in result.task_assignment.items():
            if visualizer.agents[agent_id].agent_type == "UAV":
                if task_ids:
                    print(f"  {agent_id}: {len(task_ids)}个任务 {task_ids}")
                else:
                    print(f"  {agent_id}: 无任务（电量不足）")
        
        # 显示和执行USV支援
        if result.usv_support_missions:
            print("\n🚢 Layer 2 - USV支援决策:")
            for usv_id, missions in result.usv_support_missions.items():
                print(f"  {usv_id}:")
                if usv_id in visualizer.agents:
                    usv_agent = visualizer.agents[usv_id]
                    for mission in missions:
                        target_uav_id = mission['target_uav']
                        print(f"    → 支援 {target_uav_id} "
                              f"(传输{mission['energy_to_transfer']:.1f}Wh)")
                        
                        # 初始执行：让USV移动到目标UAV位置
                        if target_uav_id in visualizer.agents:
                            target_uav = visualizer.agents[target_uav_id]
                            usv_agent.target_position = list(target_uav.position)
                            usv_agent.status = "supporting"
                            if hasattr(usv_agent, 'supported_uavs'):
                                usv_agent.supported_uavs = [target_uav_id]
                            print(f"    ✅ {usv_id} 开始移动支援 {target_uav_id}")
        else:
            print("\n🚢 Layer 2 - USV支援决策:")
            print("  当前没有USV支援任务")
            
            # 检查是否有巡逻任务
            has_patrol = False
            for usv_id, missions in result.usv_support_missions.items():
                for mission in missions:
                    if mission.get('mission_type') == 'patrol':
                        has_patrol = True
                        print(f"  🚢 {usv_id} 执行巡逻任务")
                        if usv_id in visualizer.agents:
                            usv_agent = visualizer.agents[usv_id]
                            usv_agent.target_position = list(mission['target_position'])
                            usv_agent.status = "patrolling"
            
            if not has_patrol:
                # 显示USV状态
                for agent_id, agent in visualizer.agents.items():
                    if agent.agent_type.lower() == 'usv':
                        print(f"  {agent_id}: 状态={agent.status}, 位置={agent.position}, "
                              f"电量={agent.energy:.1f}/{agent.max_energy:.1f}Wh")
        
        # 显示充电决策
        if result.charging_decisions:
            print("\n🔋 Layer 3 - 充电决策:")
            for uav_id, decision in result.charging_decisions.items():
                option_str = str(decision.option).split('.')[-1]
                if decision.target_id:
                    print(f"  {uav_id}: {option_str} → {decision.target_id} "
                          f"(评分:{decision.score:.2f})")
        
        # 显示重调度信息
        if result.reallocation_plan:
            print("\n🔄 Layer 4 - 动态重调度: 已触发")
        
        # 应用调度结果到可视化器
        visualizer.assign_tasks(result.task_assignment)
        
        # 创建调度更新线程
        def scheduler_update_loop():
            """定期更新调度"""
            update_interval = 10.0  # 10秒更新一次
            
            while True:
                time.sleep(update_interval)
                
                # 获取最新状态
                current_agents_state = {}
                for agent_id, agent in visualizer.agents.items():
                    current_agents_state[agent_id] = {
                        'position': agent.position,
                        'energy': agent.energy,
                        'max_energy': agent.max_energy,
                        'type': agent.agent_type.lower(),
                        'status': agent.status,
                        'cruise_speed': agent.speed
                    }
                    
                    if agent.agent_type.lower() == 'usv':
                        current_agents_state[agent_id]['charging_capacity'] = 200
                        current_agents_state[agent_id]['charging_rate'] = 50.0
                
                # 重新调度
                try:
                    new_result = scheduler.schedule(
                        current_agents_state, tasks, environment_state
                    )
                    
                    # 记录定期状态更新
                    logging.info(f"定期调度更新 - 时间: {time.strftime('%H:%M:%S')}")
                    
                    # 输出当前系统状态摘要和任务状态
                    uav_count = sum(1 for a in current_agents_state.values() if a['type'] == 'uav')
                    usv_count = sum(1 for a in current_agents_state.values() if a['type'] == 'usv')
                    low_energy_uavs = sum(1 for agent_id, state in current_agents_state.items() 
                                         if state['type'] == 'uav' and state['energy'] / state['max_energy'] < 0.3)
                    
                    # 检查任务完成状态
                    completed_tasks = sum(1 for task in visualizer.tasks.values() if task['status'] == 'completed')
                    total_tasks = len(visualizer.tasks)
                    in_progress_tasks = sum(1 for task in visualizer.tasks.values() if task['status'] == 'in_progress')
                    assigned_tasks = sum(1 for task in visualizer.tasks.values() if task['status'] == 'assigned')
                    pending_tasks = sum(1 for task in visualizer.tasks.values() if task['status'] == 'pending')
                    
                    print(f"\n⏱️ [{time.strftime('%H:%M:%S')}] 系统状态更新:")
                    print(f"  活跃UAV: {uav_count}, USV: {usv_count}")
                    print(f"  低电量UAV: {low_energy_uavs}个")
                    print(f"  📋 任务状态: 完成={completed_tasks}/{total_tasks}, 进行中={in_progress_tasks}, 已分配={assigned_tasks}, 待分配={pending_tasks}")
                    
                    # 监控任务分配情况
                    uav_with_tasks = 0
                    total_assigned_to_uavs = 0
                    for agent_id, agent in visualizer.agents.items():
                        if agent.agent_type == "UAV" and agent.assigned_tasks:
                            uav_with_tasks += 1
                            total_assigned_to_uavs += len(agent.assigned_tasks)
                    
                    if uav_with_tasks == 0 and pending_tasks > 0:
                        print(f"  ⚠️ 警告：没有UAV有任务，但还有{pending_tasks}个待分配任务！")
                        # 强制触发重新分配
                        print(f"  🔄 强制触发任务重新分配...")
                        
                        # 重建任务列表，包括所有未完成的任务
                        updated_tasks = []
                        for task_id, task in visualizer.tasks.items():
                            if task['status'] != 'completed':
                                task_copy = task.copy()
                                task_copy['task_id'] = task_id
                                # 重置待分配任务的状态
                                if task['status'] == 'pending':
                                    task_copy['status'] = 'pending'
                                updated_tasks.append(task_copy)
                        
                        if updated_tasks:
                            # 使用更新的任务列表重新调度
                            new_result = scheduler.schedule(
                                current_agents_state, updated_tasks, environment_state, 
                                force_reallocation=True
                            )
                            new_result.reallocation_plan = {'reason': 'no_tasks_assigned'}
                    
                    # 检查UAV状态分布
                    uav_status_count = {}
                    for agent_id, state in current_agents_state.items():
                        if state['type'] == 'uav':
                            status = visualizer.agents[agent_id].status
                            uav_status_count[status] = uav_status_count.get(status, 0) + 1
                    print(f"  🛩️ UAV状态分布: {uav_status_count}")
                    
                    # 检查USV状态
                    for agent_id, state in current_agents_state.items():
                        if state['type'] == 'usv':
                            usv_agent = visualizer.agents[agent_id]
                            print(f"  🚢 {agent_id}: 状态={usv_agent.status}, 位置={usv_agent.position}, 目标={usv_agent.target_position}")
                    
                    # 检查任务分配是否为空（防止所有UAV无任务）
                    empty_assignment = True
                    for agent_id, task_ids in new_result.task_assignment.items():
                        if agents_state[agent_id].get('type', 'uav') == 'uav' and task_ids:
                            empty_assignment = False
                            break
                    
                    if empty_assignment and pending_tasks > 0:
                        print(f"  ⚠️ 警告：所有UAV无任务但还有{pending_tasks}个待分配任务！")
                        # 强制触发重调度
                        new_result.reallocation_plan = {'reason': 'empty_assignment_fix'}
                    
                    # 检查任务分配是否为空（防止所有UAV无任务）
                    empty_assignment = True
                    for agent_id, task_ids in new_result.task_assignment.items():
                        if agents_state[agent_id].get('type', 'uav') == 'uav' and task_ids:
                            empty_assignment = False
                            break
                    
                    if empty_assignment and pending_tasks > 0:
                        print(f"  ⚠️ 警告：所有UAV无任务但还有{pending_tasks}个待分配任务！")
                        # 强制触发重调度
                        new_result.reallocation_plan = {'reason': 'empty_assignment_fix'}
                    
                    # 检查是否有变化或有新的USV支援任务
                    if new_result.reallocation_plan or new_result.usv_support_missions:
                        if new_result.reallocation_plan:
                            print(f"\n🔄 [{time.strftime('%H:%M:%S')}] 触发动态重调度!")
                            logging.info("触发动态重调度事件")
                        
                        # 显示重调度详情
                        if new_result.task_assignment:
                            print("  📋 新的任务分配:")
                            for agent_id, task_ids in new_result.task_assignment.items():
                                if current_agents_state[agent_id]['type'] == 'uav':
                                    if task_ids:
                                        print(f"    {agent_id}: {len(task_ids)}个任务 {task_ids}")
                                    else:
                                        print(f"    {agent_id}: 无任务")
                        
                        # 处理USV支援任务
                        if new_result.usv_support_missions:
                            print("  🚢 USV支援计划:")
                            for usv_id, missions in new_result.usv_support_missions.items():
                                if usv_id in visualizer.agents:
                                    usv_agent = visualizer.agents[usv_id]
                                    for mission in missions:
                                        target_uav_id = mission['target_uav']
                                        # 跳过巡逻任务，只处理真正的支援任务
                                        if target_uav_id == 'patrol':
                                            continue
                                            
                                        print(f"    {usv_id} → {target_uav_id} "
                                              f"(支援{mission['energy_to_transfer']:.1f}Wh)")
                                        
                                        # 让USV移动到目标UAV位置
                                        if target_uav_id in visualizer.agents:
                                            target_uav = visualizer.agents[target_uav_id]
                                            usv_agent.target_position = list(target_uav.position)
                                            usv_agent.status = "supporting"
                                            # 设置USV的支援目标
                                            if hasattr(usv_agent, 'supported_uavs'):
                                                usv_agent.supported_uavs = [target_uav_id]
                                            
                                            print(f"    ✅ {usv_id} 开始移动到 {target_uav_id} 位置: {target_uav.position}")
                        
                        # 应用新的任务分配
                        if new_result.task_assignment:
                            # 检查任务分配变化
                            task_assignment_changed = False
                            for agent_id, task_ids in new_result.task_assignment.items():
                                if agent_id in visualizer.agents:
                                    current_tasks = getattr(visualizer.agents[agent_id], 'assigned_tasks', [])
                                    if current_tasks != task_ids:
                                        task_assignment_changed = True
                                        print(f"📋 {agent_id} 任务分配变化: {current_tasks} → {task_ids}")
                            
                            if task_assignment_changed:
                                # 重置所有任务状态为assigned（防止任务卡在in_progress状态）
                                for task_id, task in visualizer.tasks.items():
                                    if task['status'] == 'in_progress':
                                        task['status'] = 'assigned'
                                        print(f"🔄 重置任务 {task_id} 状态: in_progress → assigned")
                                
                                # 重置UAV状态
                                for agent_id, agent in visualizer.agents.items():
                                    if agent.agent_type == "UAV" and agent.status in ['moving', 'inspecting']:
                                        agent.status = 'idle'
                                        agent.target_position = None
                                        agent.current_task_index = 0
                                        print(f"🔄 重置 {agent_id} 状态为idle")
                            
                            visualizer.assign_tasks(new_result.task_assignment)
                        
                        # 处理充电决策
                        if new_result.charging_decisions:
                            print("  🔋 充电决策执行:")
                            for uav_id, decision in new_result.charging_decisions.items():
                                # 强制执行USV支援决策
                                if decision.option.value == 'usv_support':
                                    print(f"    🚢 强制执行USV支援: {uav_id} ← {decision.target_id}")
                                    # 确保USV开始支援
                                    if decision.target_id in visualizer.agents and uav_id in visualizer.agents:
                                        usv_agent = visualizer.agents[decision.target_id]
                                        uav_agent = visualizer.agents[uav_id]
                                        usv_agent.target_position = list(uav_agent.position)
                                        usv_agent.status = "supporting"
                                        uav_agent.status = "waiting_support"
                                        if hasattr(usv_agent, 'supported_uavs'):
                                            usv_agent.supported_uavs = [uav_id]
                                        print(f"    ✅ 强制{decision.target_id}支援{uav_id}")
                                
                                elif decision.option.value == 'charging_station':
                                    # 导航到充电站
                                    if uav_id in visualizer.agents:
                                        agent = visualizer.agents[uav_id]
                                        station_id = int(decision.target_id)
                                        if 0 <= station_id < len(visualizer.charging_stations):
                                            station = visualizer.charging_stations[station_id]
                                            agent.target_position = station.position
                                            agent.status = "returning"  # 使用returning状态而不是moving
                                            print(f"    💡 {uav_id} → 充电站{station_id}")
                                
                                elif decision.option.value == 'usv_support':
                                    # USV支援决策 - 等待USV到达
                                    print(f"    🚢 {uav_id} 等待 {decision.target_id} 支援")
                                    # UAV可以继续当前任务或悬停等待
                                    if uav_id in visualizer.agents:
                                        agent = visualizer.agents[uav_id]
                                        # 如果UAV电量极低，让它悬停等待
                                        if agent.energy / agent.max_energy < 0.15:
                                            agent.status = "waiting_support"
                                            print(f"    ⏸️ {uav_id} 悬停等待支援")
                    
                    # 检查USV是否到达支援位置
                    for agent_id, agent in visualizer.agents.items():
                        if agent.agent_type.lower() == 'usv' and agent.status == "supporting":
                            # 检查是否到达目标UAV附近
                            if hasattr(agent, 'supported_uavs') and agent.supported_uavs:
                                for target_uav_id in agent.supported_uavs:
                                    if target_uav_id in visualizer.agents:
                                        target_uav = visualizer.agents[target_uav_id]
                                        # 计算距离
                                        dx = agent.position[0] - target_uav.position[0]
                                        dy = agent.position[1] - target_uav.position[1]
                                        distance = math.sqrt(dx*dx + dy*dy)
                                        
                                        # 如果距离小于50像素，开始能量传输
                                        if distance < 50:
                                            # 模拟能量传输
                                            energy_transfer_rate = 2.0  # 每次循环传输的能量
                                            max_transfer = min(
                                                agent.energy * 0.5,  # USV最多传输50%的能量
                                                target_uav.max_energy - target_uav.energy  # UAV需要的能量
                                            )
                                            
                                            if max_transfer > 0:
                                                transfer_amount = min(energy_transfer_rate, max_transfer)
                                                agent.energy -= transfer_amount
                                                target_uav.energy += transfer_amount
                                                
                                                print(f"    ⚡ {agent_id} 正在为 {target_uav_id} 充电 "
                                                      f"(+{transfer_amount:.1f}Wh)")
                                                
                                                # 如果UAV电量恢复到安全水平，结束支援
                                                if target_uav.energy / target_uav.max_energy > 0.6:
                                                    agent.status = "idle"
                                                    agent.supported_uavs = []
                                                    target_uav.status = "idle" if target_uav.status == "waiting_support" else target_uav.status
                                                    print(f"    ✅ {agent_id} 完成对 {target_uav_id} 的支援")
                        
                        # 处理巡逻中的USV
                        elif agent.agent_type.lower() == 'usv' and agent.status == "patrolling":
                            # 检查是否到达巡逻点
                            if agent.target_position:
                                dx = agent.position[0] - agent.target_position[0]
                                dy = agent.position[1] - agent.target_position[1]
                                distance = math.sqrt(dx*dx + dy*dy)
                                
                                # 如果到达巡逻点，准备前往下一个
                                if distance < 20:
                                    print(f"    🚢 {agent_id} 到达巡逻点 {agent.target_position}")
                
                except Exception as e:
                    print(f"调度更新错误: {e}")
                    logging.error(f"调度更新失败: {e}", exc_info=True)
        
        # 启动调度更新线程
        update_thread = threading.Thread(target=scheduler_update_loop, daemon=True)
        update_thread.start()
        
        # 显示性能报告
        print("\n📈 初始性能报告:")
        report = scheduler.get_performance_report()
        stats = report.get('总体统计', {})
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n🎮 启动集成调度器可视化...")
        print("\n控制说明:")
        print("  SPACE: 暂停/继续")
        print("  T: 切换轨迹显示")
        print("  I: 切换信息显示")
        print("  R: 手动触发重调度（测试）")
        print("  ESC: 退出")
        
        print("\n🔍 系统监控说明:")
        print("  • UAV电量低于40%时会触发USV支援评估")
        print("  • UAV电量低于25%时会得到紧急充电决策")
        print("  • USV会自动前往支援低电量UAV")
        print("  • 系统每10秒更新一次调度方案")
        print("\n💡 提示:")
        print("  系统会每10秒自动更新调度")
        print("  低电量UAV会自动获得充电决策")
        print("  USV会自动支援需要的UAV")
        print("\n🚀 3秒后自动启动...")
        
        # 倒计时启动
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        print("▶️ 开始运行!")
        
        # 启动可视化
        visualizer.run()
        
        # 显示最终性能报告
        print("\n📈 最终性能报告:")
        final_report = scheduler.get_performance_report()
        for category, data in final_report.items():
            print(f"\n{category}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"  {key}: {value}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        print("提示: 请确保已安装所需依赖包")
        return False
        
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = main()
        
        if success:
            print("\n✅ 集成调度器演示完成")
        else:
            print("\n❌ 集成调度器演示失败")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断执行")
    except Exception as e:
        print(f"\n❌ 未捕获的异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭日志文件
        print(f"\n📝 日志已保存到: {log_filename}")
        print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # 恢复标准输出
        sys.stdout = tee_logger.terminal
        sys.stderr = tee_logger.terminal
        tee_logger.close()