"""
卫星场景模块测试
测试场景加载、标注解析和任务生成功能
"""
import pytest
import numpy as np
import cv2
from pathlib import Path

from src.env.satellite_scene import SatelliteScene, AnnotatedObject


class TestSatelliteScene:
    """卫星场景测试类"""
    
    @pytest.mark.unit
    def test_initialization(self, temp_dir):
        """测试初始化"""
        scenes_dir = temp_dir / "scenes"
        labels_dir = temp_dir / "labels"
        scenes_dir.mkdir()
        labels_dir.mkdir()
        
        scene = SatelliteScene(str(scenes_dir), str(labels_dir))
        
        assert scene.scene_dir == scenes_dir
        assert scene.label_dir == labels_dir
        assert scene.current_scene is None
        assert scene.image is None
        assert len(scene.wind_turbines) == 0
        assert len(scene.land_areas) == 0
    
    @pytest.mark.unit
    def test_load_scene_success(self, mock_satellite_scene):
        """测试成功加载场景"""
        scene = mock_satellite_scene
        
        assert scene.current_scene == "test_scene"
        assert scene.image is not None
        assert scene.image.shape == (1024, 1024, 3)
        assert len(scene.wind_turbines) == 1
        assert len(scene.land_areas) == 1
        
        # 检查地图信息
        map_info = scene.get_scene_info()
        assert map_info['width'] == 1024
        assert map_info['height'] == 1024
        assert map_info['wind_turbine_count'] == 1
        assert map_info['land_area_count'] == 1
        assert map_info['scene_name'] == "test_scene"
    
    @pytest.mark.unit
    def test_load_nonexistent_scene(self, temp_dir):
        """测试加载不存在的场景"""
        scenes_dir = temp_dir / "scenes"
        labels_dir = temp_dir / "labels"
        scenes_dir.mkdir()
        labels_dir.mkdir()
        
        scene = SatelliteScene(str(scenes_dir), str(labels_dir))
        
        # 尝试加载不存在的场景
        success = scene.load_scene("nonexistent_scene")
        
        assert success is False
        assert scene.current_scene is None
        assert scene.image is None
    
    @pytest.mark.unit
    def test_get_wind_turbine_positions(self, mock_satellite_scene):
        """测试获取风机位置"""
        scene = mock_satellite_scene
        
        positions = scene.get_wind_turbine_positions()
        
        assert len(positions) == 1
        assert isinstance(positions[0], tuple)
        assert len(positions[0]) == 2
        
        # 检查位置是否在预期范围内
        x, y = positions[0]
        assert 100 <= x <= 150  # XML中定义的边界框中心
        assert 100 <= y <= 150
    
    @pytest.mark.unit
    def test_get_land_obstacles(self, mock_satellite_scene):
        """测试获取陆地障碍物"""
        scene = mock_satellite_scene
        
        obstacles = scene.get_land_obstacles()
        
        assert len(obstacles) == 1
        assert isinstance(obstacles[0], tuple)
        assert len(obstacles[0]) == 4  # (xmin, ymin, xmax, ymax)
        
        xmin, ymin, xmax, ymax = obstacles[0]
        assert xmin == 300
        assert ymin == 300
        assert xmax == 400
        assert ymax == 400
    
    @pytest.mark.unit
    def test_position_validation(self, mock_satellite_scene):
        """测试位置有效性检查"""
        scene = mock_satellite_scene
        
        # 测试有效位置（UAV）
        valid_uav = scene.is_position_valid(500, 500, 'uav')
        assert valid_uav is True
        
        # 测试陆地上的位置（USV不能通过）
        land_position = scene.is_position_valid(350, 350, 'usv')
        assert land_position is False
        
        # 测试陆地上的位置（UAV可以通过）
        land_position_uav = scene.is_position_valid(350, 350, 'uav')
        assert land_position_uav is True
        
        # 测试边界外位置
        out_of_bounds = scene.is_position_valid(-10, 500, 'uav')
        assert out_of_bounds is False
        
        out_of_bounds2 = scene.is_position_valid(500, 2000, 'uav')
        assert out_of_bounds2 is False
    
    @pytest.mark.unit
    def test_generate_patrol_tasks(self, mock_satellite_scene):
        """测试生成巡检任务"""
        scene = mock_satellite_scene
        
        # 生成所有风机的巡检任务
        tasks = scene.generate_patrol_tasks()
        
        assert len(tasks) == 1  # 一个风机
        task = tasks[0]
        
        assert task['task_id'] == 0
        assert task['type'] == 'inspection'
        assert isinstance(task['position'], tuple)
        assert task['priority'] == 1.0
        assert task['estimated_duration'] == 60.0
        assert task['energy_requirement'] == 10.0
        
        # 生成指定数量的任务
        limited_tasks = scene.generate_patrol_tasks(num_tasks=0)
        assert len(limited_tasks) == 0
    
    @pytest.mark.unit
    def test_visualize_scene(self, mock_satellite_scene):
        """测试场景可视化"""
        scene = mock_satellite_scene
        
        # 测试带标注的可视化
        vis_image = scene.visualize_scene(show_annotations=True)
        
        assert vis_image is not None
        assert vis_image.shape == (1024, 1024, 3)
        assert not np.array_equal(vis_image, scene.image)  # 应该有所不同（添加了标注）
        
        # 测试不带标注的可视化
        vis_image_no_anno = scene.visualize_scene(show_annotations=False)
        
        assert vis_image_no_anno is not None
        assert np.array_equal(vis_image_no_anno, scene.image)  # 应该相同
    
    @pytest.mark.unit
    def test_list_available_scenes(self, temp_dir):
        """测试列出可用场景"""
        scenes_dir = temp_dir / "scenes"
        scenes_dir.mkdir()
        
        # 创建几个测试图像文件
        for i in range(3):
            img_path = scenes_dir / f"scene_{i}.png"
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), test_img)
        
        # 添加一个非PNG文件（应该被忽略）
        (scenes_dir / "not_image.txt").write_text("test")
        
        available_scenes = SatelliteScene.list_available_scenes(str(scenes_dir))
        
        assert len(available_scenes) == 3
        assert "scene_0" in available_scenes
        assert "scene_1" in available_scenes
        assert "scene_2" in available_scenes
        assert "not_image" not in available_scenes
        
        # 检查排序
        assert available_scenes == sorted(available_scenes)
    
    @pytest.mark.unit
    def test_empty_directory(self, temp_dir):
        """测试空目录情况"""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        available_scenes = SatelliteScene.list_available_scenes(str(empty_dir))
        assert len(available_scenes) == 0
    
    @pytest.mark.unit
    def test_nonexistent_directory(self):
        """测试不存在的目录"""
        available_scenes = SatelliteScene.list_available_scenes("/nonexistent/path")
        assert len(available_scenes) == 0


class TestAnnotatedObject:
    """标注对象测试类"""
    
    @pytest.mark.unit
    def test_annotated_object_creation(self):
        """测试标注对象创建"""
        obj = AnnotatedObject(
            name="bybrid",
            bbox=(100, 100, 200, 200),
            center=(150.0, 150.0),
            truncated=False,
            difficult=False
        )
        
        assert obj.name == "bybrid"
        assert obj.bbox == (100, 100, 200, 200)
        assert obj.center == (150.0, 150.0)
        assert obj.truncated is False
        assert obj.difficult is False
    
    @pytest.mark.unit
    def test_annotated_object_defaults(self):
        """测试标注对象默认值"""
        obj = AnnotatedObject(
            name="land",
            bbox=(0, 0, 100, 100),
            center=(50.0, 50.0)
        )
        
        assert obj.truncated is False
        assert obj.difficult is False


@pytest.mark.integration
class TestSatelliteSceneIntegration:
    """卫星场景集成测试"""
    
    def test_complete_workflow(self, temp_dir):
        """测试完整工作流程"""
        # 创建目录结构
        scenes_dir = temp_dir / "scenes"
        labels_dir = temp_dir / "labels"
        scenes_dir.mkdir()
        labels_dir.mkdir()
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image_path = scenes_dir / "integration_test.png"
        cv2.imwrite(str(image_path), test_image)
        
        # 创建测试标注
        xml_content = '''<annotation>
        <folder>test</folder>
        <filename>integration_test.png</filename>
        <size>
            <width>512</width>
            <height>512</height>
            <depth>3</depth>
        </size>
        <object>
            <name>bybrid</name>
            <bndbox>
                <xmin>50</xmin>
                <ymin>50</ymin>
                <xmax>100</xmax>
                <ymax>100</ymax>
            </bndbox>
        </object>
        <object>
            <name>bybrid</name>
            <bndbox>
                <xmin>200</xmin>
                <ymin>200</ymin>
                <xmax>250</xmax>
                <ymax>250</ymax>
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
        </annotation>'''
        
        xml_path = labels_dir / "integration_test.xml"
        xml_path.write_text(xml_content, encoding='utf-8')
        
        # 执行完整流程
        scene = SatelliteScene(str(scenes_dir), str(labels_dir))
        
        # 1. 加载场景
        success = scene.load_scene("integration_test")
        assert success is True
        
        # 2. 检查解析结果
        assert len(scene.wind_turbines) == 2
        assert len(scene.land_areas) == 1
        
        # 3. 生成任务
        tasks = scene.generate_patrol_tasks()
        assert len(tasks) == 2
        
        # 4. 获取位置信息
        turbine_positions = scene.get_wind_turbine_positions()
        assert len(turbine_positions) == 2
        
        obstacles = scene.get_land_obstacles()
        assert len(obstacles) == 1
        
        # 5. 位置验证
        # 在陆地区域
        land_check = scene.is_position_valid(350, 350, 'usv')
        assert land_check is False
        
        # 在空旷区域
        open_check = scene.is_position_valid(150, 150, 'usv')
        assert open_check is True
        
        # 6. 可视化
        vis_image = scene.visualize_scene(show_annotations=True)
        assert vis_image is not None
        assert vis_image.shape == (512, 512, 3)
        
        # 7. 获取场景信息
        info = scene.get_scene_info()
        assert info['wind_turbine_count'] == 2
        assert info['land_area_count'] == 1
        assert info['scene_name'] == "integration_test"
    
    def test_large_scene_performance(self, temp_dir):
        """测试大场景性能"""
        import time
        
        scenes_dir = temp_dir / "scenes"
        labels_dir = temp_dir / "labels"
        scenes_dir.mkdir()
        labels_dir.mkdir()
        
        # 创建大图像
        large_image = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
        image_path = scenes_dir / "large_scene.png"
        cv2.imwrite(str(image_path), large_image)
        
        # 创建包含多个对象的标注
        xml_content = '<annotation><folder>test</folder><filename>large_scene.png</filename><size><width>2048</width><height>2048</height><depth>3</depth></size>'
        
        # 添加100个风机
        for i in range(100):
            x = 50 + (i % 10) * 200
            y = 50 + (i // 10) * 200
            xml_content += f'''<object>
            <name>bybrid</name>
            <bndbox>
                <xmin>{x}</xmin>
                <ymin>{y}</ymin>
                <xmax>{x+50}</xmax>
                <ymax>{y+50}</ymax>
            </bndbox>
            </object>'''
        
        xml_content += '</annotation>'
        
        xml_path = labels_dir / "large_scene.xml"
        xml_path.write_text(xml_content, encoding='utf-8')
        
        # 测试加载性能
        scene = SatelliteScene(str(scenes_dir), str(labels_dir))
        
        start_time = time.time()
        success = scene.load_scene("large_scene")
        load_time = time.time() - start_time
        
        assert success is True
        assert len(scene.wind_turbines) == 100
        assert load_time < 5.0  # 应该在5秒内完成
        
        # 测试任务生成性能
        start_time = time.time()
        tasks = scene.generate_patrol_tasks()
        task_time = time.time() - start_time
        
        assert len(tasks) == 100
        assert task_time < 1.0  # 应该在1秒内完成