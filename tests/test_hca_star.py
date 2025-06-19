"""
HCA-A*路径规划器测试
测试路径规划算法的正确性和性能
"""
import pytest
import numpy as np
from typing import List, Tuple

from src.planner.hca_star import HCAStarPlanner, SimpleEnvironment, GridNode
from src.planner.base_planner import BasePlanner


class TestGridNode:
    """网格节点测试类"""
    
    @pytest.mark.unit
    def test_grid_node_creation(self):
        """测试网格节点创建"""
        node = GridNode(10, 20)
        
        assert node.x == 10
        assert node.y == 20
        assert node.g_cost == float('inf')
        assert node.h_cost == 0.0
        assert node.f_cost == float('inf')
        assert node.parent is None
        assert node.energy_cost == float('inf')
        assert node.time_cost == float('inf')
        assert node.is_obstacle is False
    
    @pytest.mark.unit
    def test_grid_node_comparison(self):
        """测试网格节点比较"""
        node1 = GridNode(0, 0)
        node1.f_cost = 10.0
        
        node2 = GridNode(1, 1)
        node2.f_cost = 20.0
        
        assert node1 < node2
    
    @pytest.mark.unit
    def test_grid_node_hash_equality(self):
        """测试网格节点哈希和相等性"""
        node1 = GridNode(5, 5)
        node2 = GridNode(5, 5)
        node3 = GridNode(5, 6)
        
        assert node1 == node2
        assert node1 != node3
        assert hash(node1) == hash(node2)
        assert hash(node1) != hash(node3)


class TestSimpleEnvironment:
    """简单环境测试类"""
    
    @pytest.mark.unit
    def test_simple_environment_creation(self):
        """测试简单环境创建"""
        env = SimpleEnvironment(800, 600, [(100, 100, 200, 200)])
        
        assert env.width == 800
        assert env.height == 600
        assert len(env.obstacles) == 1
        assert env.obstacles[0] == (100, 100, 200, 200)
    
    @pytest.mark.unit
    def test_obstacle_detection(self):
        """测试障碍物检测"""
        obstacles = [(200, 200, 300, 300), (500, 500, 600, 600)]
        env = SimpleEnvironment(1000, 1000, obstacles)
        
        # UAV可以飞越障碍物
        assert env.is_obstacle(250, 250, 'uav') is False
        assert env.is_obstacle(550, 550, 'uav') is False
        
        # USV不能通过障碍物
        assert env.is_obstacle(250, 250, 'usv') is True
        assert env.is_obstacle(550, 550, 'usv') is True
        
        # 空旷区域对所有智能体都可通行
        assert env.is_obstacle(100, 100, 'uav') is False
        assert env.is_obstacle(100, 100, 'usv') is False
    
    @pytest.mark.unit
    def test_map_bounds(self):
        """测试地图边界"""
        env = SimpleEnvironment(1200, 800)
        
        bounds = env.get_map_bounds()
        assert bounds == (0, 0, 1200, 800)


class TestHCAStarPlanner:
    """HCA-A*规划器测试类"""
    
    @pytest.mark.unit
    def test_planner_initialization(self, sample_config):
        """测试规划器初始化"""
        planner = HCAStarPlanner(sample_config['planner'])
        
        assert planner.grid_resolution == 20.0
        assert planner.time_weight == 1.0
        assert planner.energy_weight == 1.0
        assert planner.energy_calculator is not None
        assert len(planner.neighbor_directions) == 8
    
    @pytest.mark.unit
    def test_coordinate_conversion(self, hca_planner):
        """测试坐标转换"""
        planner = hca_planner
        
        # 世界坐标到网格坐标
        world_pos = (100.0, 200.0)
        grid_pos = planner._world_to_grid(world_pos)
        expected_grid = (5, 10)  # 100/20=5, 200/20=10
        assert grid_pos == expected_grid
        
        # 网格坐标到世界坐标
        back_to_world = planner._grid_to_world(grid_pos)
        expected_world = (110.0, 210.0)  # 5*20+10=110, 10*20+10=210
        assert back_to_world == expected_world
    
    @pytest.mark.unit
    def test_heuristic_cost(self, hca_planner):
        """测试启发式成本函数"""
        planner = hca_planner
        
        # 测试相同位置
        cost = planner._heuristic_cost((0, 0), (0, 0))
        assert cost == 0.0
        
        # 测试不同位置
        cost = planner._heuristic_cost((0, 0), (3, 4))
        assert cost > 0
        
        # 距离应该单调递增
        cost1 = planner._heuristic_cost((0, 0), (1, 1))
        cost2 = planner._heuristic_cost((0, 0), (2, 2))
        cost3 = planner._heuristic_cost((0, 0), (3, 3))
        
        assert cost1 < cost2 < cost3
    
    @pytest.mark.unit
    def test_simple_path_planning(self, hca_planner, simple_environment):
        """测试简单路径规划"""
        planner = hca_planner
        env = simple_environment
        
        start = (50.0, 50.0)
        target = (150.0, 150.0)
        
        path, time_cost, energy_cost = planner.plan_path(
            start, target, 'uav', env
        )
        
        assert len(path) >= 2  # 至少包含起点和终点
        assert path[0] == start or np.allclose(path[0], start, atol=planner.grid_resolution)
        assert path[-1] == target or np.allclose(path[-1], target, atol=planner.grid_resolution)
        assert time_cost > 0
        assert energy_cost > 0
    
    @pytest.mark.unit
    def test_path_planning_with_obstacles(self, hca_planner):
        """测试带障碍物的路径规划"""
        planner = hca_planner
        
        # 创建有障碍物的环境
        obstacles = [(140, 140, 160, 160)]  # 在起点和终点之间放置障碍物
        env = SimpleEnvironment(1000, 1000, obstacles)
        
        start = (120.0, 120.0)
        target = (180.0, 180.0)
        
        # USV需要绕过障碍物
        path, time_cost, energy_cost = planner.plan_path(
            start, target, 'usv', env
        )
        
        if path:  # 如果找到路径
            assert len(path) >= 2
            # 路径不应该穿过障碍物
            for x, y in path:
                assert not env.is_obstacle(x, y, 'usv')
    
    @pytest.mark.unit
    def test_multi_target_planning(self, hca_planner, simple_environment):
        """测试多目标路径规划"""
        planner = hca_planner
        env = simple_environment
        
        start = (50.0, 50.0)
        targets = [(150.0, 150.0), (250.0, 200.0), (300.0, 100.0)]
        
        path, time_cost, energy_cost = planner.plan_multi_target(
            start, targets, 'uav', env
        )
        
        assert len(path) >= len(targets) + 1  # 起点 + 所有目标点
        assert time_cost > 0
        assert energy_cost > 0
        
        # 检查是否访问了所有目标点
        for target in targets:
            # 找到路径中最接近目标的点
            min_dist = min(
                np.sqrt((px - target[0])**2 + (py - target[1])**2) 
                for px, py in path
            )
            assert min_dist < planner.grid_resolution * 2  # 应该足够接近
    
    @pytest.mark.unit
    def test_impossible_path(self, hca_planner):
        """测试不可能的路径"""
        planner = hca_planner
        
        # 创建完全封闭的环境
        large_obstacle = [(100, 100, 900, 900)]
        env = SimpleEnvironment(1000, 1000, large_obstacle)
        
        start = (50.0, 50.0)  # 在障碍物外
        target = (500.0, 500.0)  # 在障碍物内
        
        path, time_cost, energy_cost = planner.plan_path(
            start, target, 'usv', env
        )
        
        # 应该找不到路径
        assert len(path) == 0
        assert time_cost == float('inf')
        assert energy_cost == float('inf')
    
    @pytest.mark.unit
    def test_same_start_target(self, hca_planner, simple_environment):
        """测试起点和终点相同的情况"""
        planner = hca_planner
        env = simple_environment
        
        position = (200.0, 200.0)
        
        path, time_cost, energy_cost = planner.plan_path(
            position, position, 'uav', env
        )
        
        # 应该返回包含单个点的路径或空路径
        assert len(path) <= 2
        if len(path) > 0:
            assert np.allclose(path[0], position, atol=planner.grid_resolution)
    
    @pytest.mark.unit
    def test_path_cost_calculation(self, hca_planner, simple_environment):
        """测试路径成本计算"""
        planner = hca_planner
        env = simple_environment
        
        # 创建一个简单的直线路径
        path = [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)]
        
        total_time, total_energy = planner._calculate_path_cost(path, 'uav', env)
        
        assert total_time > 0
        assert total_energy > 0
        
        # 更长的路径应该消耗更多
        longer_path = [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0), (300.0, 0.0)]
        longer_time, longer_energy = planner._calculate_path_cost(longer_path, 'uav', env)
        
        assert longer_time > total_time
        assert longer_energy > total_energy
    
    @pytest.mark.parametrize("agent_type", ['uav', 'usv'])
    def test_different_agent_types(self, hca_planner, simple_environment, agent_type):
        """测试不同智能体类型的路径规划"""
        planner = hca_planner
        env = simple_environment
        
        start = (100.0, 100.0)
        target = (400.0, 400.0)
        
        path, time_cost, energy_cost = planner.plan_path(
            start, target, agent_type, env
        )
        
        # 应该能找到路径
        assert len(path) >= 2
        assert time_cost > 0
        assert energy_cost > 0


@pytest.mark.integration  
class TestHCAStarIntegration:
    """HCA-A*集成测试"""
    
    def test_realistic_mission_planning(self, sample_config):
        """测试真实任务场景规划"""
        planner = HCAStarPlanner(sample_config['planner'])
        
        # 创建复杂环境
        obstacles = [
            (200, 200, 300, 300),  # 陆地1
            (500, 100, 600, 250),  # 陆地2
            (100, 500, 250, 650)   # 陆地3
        ]
        env = SimpleEnvironment(1000, 1000, obstacles)
        
        # 规划多段任务
        waypoints = [
            (50.0, 50.0),    # 起点
            (350.0, 150.0),  # 目标1
            (150.0, 350.0),  # 目标2
            (700.0, 700.0),  # 目标3
            (50.0, 50.0)     # 返回起点
        ]
        
        total_time = 0.0
        total_energy = 0.0
        complete_path = []
        
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            target = waypoints[i + 1]
            
            segment_path, segment_time, segment_energy = planner.plan_path(
                start, target, 'usv', env  # USV需要避开陆地
            )
            
            if not segment_path:
                pytest.fail(f"无法规划从 {start} 到 {target} 的路径")
            
            # 添加路径段（避免重复点）
            if i == 0:
                complete_path.extend(segment_path)
            else:
                complete_path.extend(segment_path[1:])
            
            total_time += segment_time
            total_energy += segment_energy
        
        # 验证完整任务
        assert len(complete_path) >= len(waypoints)
        assert total_time > 0
        assert total_energy > 0
        
        # 验证路径有效性
        for x, y in complete_path:
            assert not env.is_obstacle(x, y, 'usv')
    
    def test_performance_large_map(self, sample_config):
        """测试大地图性能"""
        import time
        
        # 增大网格分辨率以减少搜索空间
        config = sample_config['planner'].copy()
        config['grid_resolution'] = 50.0
        config['max_iterations'] = 5000
        
        planner = HCAStarPlanner(config)
        
        # 创建大地图
        env = SimpleEnvironment(5000, 5000, [(2000, 2000, 2500, 2500)])
        
        start = (100.0, 100.0)
        target = (4900.0, 4900.0)
        
        start_time = time.time()
        path, time_cost, energy_cost = planner.plan_path(
            start, target, 'uav', env
        )
        planning_time = time.time() - start_time
        
        # 性能要求
        assert planning_time < 10.0  # 应该在10秒内完成
        assert len(path) >= 2
        assert time_cost > 0
        assert energy_cost > 0
    
