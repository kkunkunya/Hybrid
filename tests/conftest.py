"""
测试配置和夹具
提供通用的测试数据和模拟对象
"""
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
import cv2

from src.env.satellite_scene import SatelliteScene
from src.utils.energy import EnergyCalculator, AgentType
from src.planner.hca_star import HCAStarPlanner, SimpleEnvironment
from src.planner.opt_2opt import TwoOptOptimizer
from src.scheduler.rl_agent import RLScheduler
from src.config.config_loader import ConfigLoader


@pytest.fixture
def temp_dir():
    """临时目录夹具"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """示例配置夹具"""
    return {
        'system': {
            'name': 'test_system',
            'random_seed': 42
        },
        'agents': {
            'uav': {
                'count': 2,
                'max_speed': 15.0,
                'battery_capacity': 300.0
            },
            'usv': {
                'count': 1,
                'max_speed': 8.0,
                'battery_capacity': 1000.0
            }
        },
        'planner': {
            'grid_resolution': 20.0,
            'time_weight': 1.0,
            'energy_weight': 1.0
        },
        'scheduler': {
            'state_dim': 32,
            'action_dim': 20,
            'learning_rate': 1e-4,
            'buffer_size': 1000
        }
    }


@pytest.fixture
def energy_calculator():
    """能源计算器夹具"""
    return EnergyCalculator()


@pytest.fixture
def simple_environment():
    """简单环境夹具"""
    return SimpleEnvironment(
        width=1000, 
        height=1000, 
        obstacles=[(200, 200, 300, 300)]
    )


@pytest.fixture
def hca_planner(sample_config):
    """HCA-A*规划器夹具"""
    return HCAStarPlanner(sample_config['planner'])


@pytest.fixture
def two_opt_optimizer():
    """2-opt优化器夹具"""
    return TwoOptOptimizer({'max_iterations': 100})


@pytest.fixture
def rl_scheduler(sample_config):
    """RL调度器夹具"""
    return RLScheduler(sample_config['scheduler'])


@pytest.fixture
def sample_agents_state() -> Dict[str, Dict[str, Any]]:
    """示例智能体状态夹具"""
    return {
        'uav1': {
            'position': (100.0, 100.0),
            'energy': 250.0,
            'max_energy': 300.0,
            'type': 'uav',
            'status': 'idle'
        },
        'uav2': {
            'position': (200.0, 200.0),
            'energy': 280.0,
            'max_energy': 300.0,
            'type': 'uav',
            'status': 'idle'
        },
        'usv1': {
            'position': (50.0, 50.0),
            'energy': 900.0,
            'max_energy': 1000.0,
            'type': 'usv',
            'status': 'idle'
        }
    }


@pytest.fixture
def sample_tasks() -> List[Dict[str, Any]]:
    """示例任务列表夹具"""
    return [
        {
            'task_id': 0,
            'position': (150.0, 150.0),
            'priority': 2.0,
            'estimated_duration': 60.0,
            'energy_requirement': 15.0
        },
        {
            'task_id': 1,
            'position': (250.0, 250.0),
            'priority': 1.5,
            'estimated_duration': 90.0,
            'energy_requirement': 20.0
        },
        {
            'task_id': 2,
            'position': (350.0, 350.0),
            'priority': 1.0,
            'estimated_duration': 45.0,
            'energy_requirement': 12.0
        },
        {
            'task_id': 3,
            'position': (400.0, 200.0),
            'priority': 1.8,
            'estimated_duration': 75.0,
            'energy_requirement': 18.0
        }
    ]


@pytest.fixture
def sample_environment_state() -> Dict[str, Any]:
    """示例环境状态夹具"""
    return {
        'temperature': 25.0,
        'visibility': 10000.0
    }


@pytest.fixture
def mock_satellite_scene(temp_dir):
    """模拟卫星场景夹具"""
    # 创建临时场景和标注目录
    scenes_dir = temp_dir / "scenes"
    labels_dir = temp_dir / "labels"
    scenes_dir.mkdir()
    labels_dir.mkdir()
    
    # 创建模拟图像
    test_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    test_image[100:200, 100:200] = [0, 255, 0]  # 绿色区域
    test_image[300:400, 300:400] = [255, 0, 0]  # 红色区域
    
    image_path = scenes_dir / "test_scene.png"
    cv2.imwrite(str(image_path), test_image)
    
    # 创建模拟XML标注
    xml_content = """<annotation>
    <folder>test</folder>
    <filename>test_scene.png</filename>
    <size>
        <width>1024</width>
        <height>1024</height>
        <depth>3</depth>
    </size>
    <object>
        <name>bybrid</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>100</ymin>
            <xmax>150</xmax>
            <ymax>150</ymax>
        </bndbox>
    </object>
    <object>
        <name>land</name>
        <bndbox>
            <xmin>300</xmin>
            <ymin>300</ymin>
            <xmax>400</xmax>
            <ymax>400</ymax>
        </bndbox>
    </object>
</annotation>"""
    
    xml_path = labels_dir / "test_scene.xml"
    xml_path.write_text(xml_content, encoding='utf-8')
    
    # 创建场景处理器
    scene = SatelliteScene(str(scenes_dir), str(labels_dir))
    scene.load_scene("test_scene")
    
    return scene


@pytest.fixture
def sample_positions() -> List[tuple]:
    """示例位置列表夹具"""
    np.random.seed(42)
    return [
        (100.0, 200.0),
        (300.0, 150.0),
        (450.0, 350.0),
        (200.0, 400.0),
        (500.0, 100.0)
    ]


@pytest.fixture
def config_loader(temp_dir):
    """配置加载器夹具"""
    config_dir = temp_dir / "config"
    config_dir.mkdir()
    
    # 创建测试配置文件
    test_config = {
        'system': {'name': 'test'},
        'agents': {'uav': {'count': 2}, 'usv': {'count': 1}},
        'scheduler': {'learning_rate': 1e-4}
    }
    
    config_file = config_dir / "test.yaml"
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(test_config, f)
    
    return ConfigLoader(str(config_dir))


class MockEnvironment:
    """模拟环境类（用于测试）"""
    
    def __init__(self, width=1000, height=1000, obstacles=None):
        self.width = width
        self.height = height
        self.obstacles = obstacles or []
    
    def is_obstacle(self, x, y, agent_type):
        """检查是否为障碍物"""
        if agent_type == 'uav':
            return False  # UAV可飞越
        
        for xmin, ymin, xmax, ymax in self.obstacles:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return True
        return False
    
    def get_map_bounds(self):
        """返回地图边界"""
        return 0, 0, self.width, self.height


@pytest.fixture
def mock_environment():
    """模拟环境夹具"""
    return MockEnvironment(1000, 1000, [(200, 200, 300, 300)])


# 参数化测试数据
@pytest.fixture(params=['uav', 'usv'])
def agent_type(request):
    """智能体类型参数化夹具"""
    return request.param


@pytest.fixture(params=[
    (AgentType.UAV, 100.0, 10.0),
    (AgentType.USV, 500.0, 6.0),
])
def agent_energy_speed(request):
    """智能体类型、能量、速度参数化夹具"""
    return request.param


# 性能测试相关夹具
@pytest.fixture
def large_task_set():
    """大规模任务集合（用于性能测试）"""
    np.random.seed(42)
    tasks = []
    for i in range(100):
        tasks.append({
            'task_id': i,
            'position': (np.random.uniform(50, 950), np.random.uniform(50, 950)),
            'priority': np.random.uniform(0.5, 3.0),
            'estimated_duration': np.random.uniform(30, 300),
            'energy_requirement': np.random.uniform(5, 50)
        })
    return tasks


@pytest.fixture
def large_agent_fleet():
    """大规模智能体群（用于性能测试）"""
    agents = {}
    
    # 创建多个UAV
    for i in range(10):
        agents[f'uav{i}'] = {
            'position': (np.random.uniform(50, 950), np.random.uniform(50, 950)),
            'energy': np.random.uniform(200, 300),
            'max_energy': 300.0,
            'type': 'uav',
            'status': 'idle'
        }
    
    # 创建多个USV
    for i in range(5):
        agents[f'usv{i}'] = {
            'position': (np.random.uniform(50, 950), np.random.uniform(50, 950)),
            'energy': np.random.uniform(800, 1000),
            'max_energy': 1000.0,
            'type': 'usv',
            'status': 'idle'
        }
    
    return agents