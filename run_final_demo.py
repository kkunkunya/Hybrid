#!/usr/bin/env python3
"""
最终版动态可视化演示
修复：
1. 保留完整轨迹
2. USV作为后勤支援（不执行巡检）
"""
import sys
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """主函数 - 最终版演示"""
    print("🎮 多UAV-USV协同巡检系统 - 最终版演示")
    print("=" * 60)
    
    # 显示新特性
    print("\n🚀 最终版特性:")
    print("1. 轨迹保留: 完整保存所有运动轨迹，便于路径分析")
    print("2. USV角色: 作为后勤支援，不执行巡检任务")
    print("3. 智能支援: USV自动支援低电量且远离充电桩的UAV")
    print("4. 距离统计: 实时显示UAV和USV的总行驶距离")
    
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
    
    choice = input("\n请选择 (1-4，默认1): ").strip() or '1'
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
        
        # 加载场景
        data_dir = project_root / "data"
        parser = EnhancedSceneParser(scene_name)
        
        if not parser.load_scene(data_dir):
            print("❌ 场景加载失败")
            return False
        
        print("✅ 场景加载成功")
        
        # 创建最终版可视化器
        visualizer = FinalEnhancedPygameVisualizer(parser, (1400, 1000))
        
        if not visualizer.setup_visualization():
            print("❌ 可视化器设置失败")
            return False
        
        print("✅ 最终版可视化器设置完成")
        print(f"   充电桩数量: {len(visualizer.charging_stations)}")
        
        # 获取智能体位置
        charging_positions = [station.position for station in visualizer.charging_stations]
        safe_positions = parser.get_safe_spawn_positions(4, charging_positions)
        
        print(f"✅ 生成智能体位置: {len(safe_positions)}个")
        
        # 添加智能体
        uav_count = 3
        usv_count = 1
        
        print("\n📋 智能体配置:")
        for i in range(uav_count):
            if i < len(safe_positions):
                visualizer.add_agent(f"uav{i+1}", "UAV", safe_positions[i])
        
        for i in range(usv_count):
            if i + uav_count < len(safe_positions):
                visualizer.add_agent(f"usv{i+1}", "USV", safe_positions[i + uav_count])
        
        # 添加任务（只分配给UAV）
        task_positions = parser.get_task_positions()
        num_tasks = min(12, len(task_positions))  # 最多12个任务
        
        for i in range(num_tasks):
            visualizer.add_task(i, task_positions[i], "wind_turbine_inspection")
        
        print(f"\n✅ 添加巡检任务: {num_tasks}个")
        
        # 任务分配（只分配给UAV）
        assignments = {}
        uav_agents = [aid for aid, agent in visualizer.agents.items() if agent.agent_type == "UAV"]
        
        if uav_agents:
            tasks_per_uav = num_tasks // len(uav_agents)
            task_index = 0
            
            for uav_id in uav_agents:
                uav_tasks = []
                for _ in range(tasks_per_uav):
                    if task_index < num_tasks:
                        uav_tasks.append(task_index)
                        task_index += 1
                if uav_tasks:
                    assignments[uav_id] = uav_tasks
            
            # 分配剩余任务
            while task_index < num_tasks:
                uav_id = uav_agents[task_index % len(uav_agents)]
                if uav_id not in assignments:
                    assignments[uav_id] = []
                assignments[uav_id].append(task_index)
                task_index += 1
        
        # USV不分配任务
        for agent_id, agent in visualizer.agents.items():
            if agent.agent_type == "USV":
                assignments[agent_id] = []  # 空任务列表
        
        visualizer.assign_tasks(assignments)
        
        print("\n✅ 任务分配完成:")
        for agent_id, task_ids in assignments.items():
            agent_type = visualizer.agents[agent_id].agent_type
            if agent_type == "UAV":
                print(f"   {agent_id} (巡检无人机): {len(task_ids)}个任务")
            else:
                print(f"   {agent_id} (后勤支援船): 待命支援低电量UAV")
        
        print("\n🎮 启动最终版动态可视化...")
        print("核心改进:")
        print("  ✅ 完整轨迹保留 - 便于路径分析")
        print("  ✅ USV后勤支援 - 不执行巡检任务")
        print("  ✅ 智能支援系统 - 自动支援低电量UAV")
        print("  ✅ 距离统计 - 实时显示总行驶距离")
        print("\n控制说明:")
        print("  SPACE: 暂停/继续")
        print("  T: 切换轨迹显示")
        print("  I: 切换信息显示")
        print("  ESC: 退出")
        print("\n🚀 3秒后自动启动...")
        
        # 倒计时启动
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        print("▶️ 开始运行!")
        
        # 启动可视化
        visualizer.run()
        
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
    success = main()
    
    if success:
        print("\n✅ 最终版演示完成")
    else:
        print("\n❌ 最终版演示失败")
        sys.exit(1)