"""
增强版场景解析器
支持Marine_Obstacles标注和多场景加载
"""
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np
from pathlib import Path


class SceneObject:
    """场景对象基类"""
    
    def __init__(self, name: str, bbox: Tuple[int, int, int, int], object_type: str):
        """
        初始化场景对象
        
        Args:
            name: 对象名称
            bbox: 边界框 (xmin, ymin, xmax, ymax)
            object_type: 对象类型
        """
        self.name = name
        self.bbox = bbox
        self.object_type = object_type
        self.center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]


class WindTurbine(SceneObject):
    """风力发电机对象"""
    
    def __init__(self, turbine_id: int, bbox: Tuple[int, int, int, int]):
        super().__init__(f"Turbine_{turbine_id:02d}", bbox, "wind_turbine")
        self.turbine_id = turbine_id
        self.status = "idle"  # idle, inspecting, completed
        self.priority = 1.0
        self.inspection_duration = 60.0  # 秒


class LandArea(SceneObject):
    """陆地区域对象"""
    
    def __init__(self, land_id: int, bbox: Tuple[int, int, int, int]):
        super().__init__(f"Land_{land_id:02d}", bbox, "land")
        self.land_id = land_id


class MarineObstacle(SceneObject):
    """海洋障碍物对象 - USV无法通过，UAV可以飞越"""
    
    def __init__(self, obstacle_id: int, bbox: Tuple[int, int, int, int]):
        super().__init__(f"MarineObstacle_{obstacle_id:02d}", bbox, "marine_obstacle")
        self.obstacle_id = obstacle_id
        self.is_truncated = False  # 是否被图像边缘截断


class EnhancedSceneParser:
    """增强版场景解析器"""
    
    # 支持的场景列表
    SUPPORTED_SCENES = [
        "xinghua_bay_wind_farm",
        "fuqing_haitan_wind_farm",
        "pingtan_wind_farm",
        "putian_nanri_wind_farm_phase1_2"
    ]
    
    # 场景中文名映射（向后兼容）
    SCENE_NAME_MAPPING = {
        "兴化湾海上风电场": "xinghua_bay_wind_farm",
        "福清海坛海峡海上风电场": "fuqing_haitan_wind_farm",
        "平潭风电站": "pingtan_wind_farm",
        "莆田南日岛海上风电场一期2": "putian_nanri_wind_farm_phase1_2"
    }
    
    def __init__(self, scene_name: str = "xinghua_bay_wind_farm"):
        """
        初始化场景解析器
        
        Args:
            scene_name: 场景名称（支持英文名或中文名）
        """
        # 处理中文名映射
        if scene_name in self.SCENE_NAME_MAPPING:
            scene_name = self.SCENE_NAME_MAPPING[scene_name]
        
        # 验证场景名称
        if scene_name not in self.SUPPORTED_SCENES:
            print(f"⚠️ 未知场景: {scene_name}，使用默认场景: xinghua_bay_wind_farm")
            scene_name = "xinghua_bay_wind_farm"
        
        self.scene_name = scene_name
        self.image_path = None
        self.xml_path = None
        self.image = None
        self.wind_turbines: List[WindTurbine] = []
        self.land_areas: List[LandArea] = []
        self.marine_obstacles: List[MarineObstacle] = []
        self.image_size = (1024, 1024)
        
    def load_scene(self, data_dir: Path) -> bool:
        """
        加载场景数据
        
        Args:
            data_dir: 数据目录路径
            
        Returns:
            是否加载成功
        """
        try:
            # 构建文件路径
            self.image_path = data_dir / "scenes" / f"{self.scene_name}.png"
            self.xml_path = data_dir / "labels" / f"{self.scene_name}.xml"
            
            # 检查文件是否存在
            if not self.image_path.exists():
                print(f"❌ 图像文件不存在: {self.image_path}")
                return False
                
            if not self.xml_path.exists():
                print(f"❌ 标注文件不存在: {self.xml_path}")
                return False
            
            # 加载图像
            self.image = self._load_image_unicode(self.image_path)
            if self.image is None:
                print(f"❌ 无法加载图像: {self.image_path}")
                return False
                
            self.image_size = (self.image.shape[1], self.image.shape[0])  # (width, height)
            print(f"✅ 已加载图像: {self.image_size}")
            
            # 解析标注文件
            self._parse_annotations()
            
            print(f"✅ 场景 '{self.scene_name}' 加载完成:")
            print(f"   风力发电机: {len(self.wind_turbines)}个")
            print(f"   陆地区域: {len(self.land_areas)}个")
            print(f"   海洋障碍物: {len(self.marine_obstacles)}个")
            
            # 场景特殊说明
            if self.scene_name == "pingtan_wind_farm" and len(self.marine_obstacles) == 0:
                print("   ⚠️ 注意: 平潭风电站场景暂无海洋障碍物标注")
            elif self.scene_name == "putian_nanri_wind_farm_phase1_2" and len(self.land_areas) == 0:
                print("   ⚠️ 注意: 莆田南日岛场景暂无陆地标注")
            
            return True
            
        except Exception as e:
            print(f"❌ 场景加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_image_unicode(self, image_path: Path) -> np.ndarray:
        """
        加载图像，解决中文路径问题
        
        Args:
            image_path: 图像路径
            
        Returns:
            图像数组，失败返回None
        """
        try:
            # 读取文件为字节流
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # 转换为numpy数组
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            
            # 使用cv2解码
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is not None:
                print(f"✅ 成功加载图像: {image.shape}")
                return image
            
            print(f"❌ 无法加载图像: {image_path}")
            return None
            
        except Exception as e:
            print(f"❌ 图像加载异常: {e}")
            return None
    
    def _parse_annotations(self):
        """解析XML标注文件"""
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
            
            turbine_count = 0
            land_count = 0
            obstacle_count = 0
            
            for obj in root.findall('object'):
                name = obj.find('name').text
                
                # 解析边界框
                bbox_elem = obj.find('bndbox')
                xmin = int(bbox_elem.find('xmin').text)
                ymin = int(bbox_elem.find('ymin').text)
                xmax = int(bbox_elem.find('xmax').text)
                ymax = int(bbox_elem.find('ymax').text)
                bbox = (xmin, ymin, xmax, ymax)
                
                # 检查是否被截断
                truncated_elem = obj.find('truncated')
                truncated = truncated_elem is not None and truncated_elem.text == '1'
                
                # 根据类型创建对象
                if name.lower() in ['bybrid', 'wind_turbine', 'turbine']:
                    turbine = WindTurbine(turbine_count, bbox)
                    self.wind_turbines.append(turbine)
                    turbine_count += 1
                    
                elif name.lower() in ['land', 'ground']:
                    land = LandArea(land_count, bbox)
                    self.land_areas.append(land)
                    land_count += 1
                    
                elif name == 'Marine_Obstacles':
                    obstacle = MarineObstacle(obstacle_count, bbox)
                    obstacle.is_truncated = truncated
                    self.marine_obstacles.append(obstacle)
                    obstacle_count += 1
                    
        except Exception as e:
            print(f"❌ 标注解析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def get_task_positions(self) -> List[Tuple[int, int]]:
        """获取所有任务位置（风机中心点）"""
        return [turbine.center for turbine in self.wind_turbines]
    
    def get_obstacle_areas(self) -> List[Tuple[int, int, int, int]]:
        """获取障碍物区域（陆地边界框）"""
        return [land.bbox for land in self.land_areas]
    
    def get_marine_obstacle_areas(self) -> List[Tuple[int, int, int, int]]:
        """获取海洋障碍物区域（USV无法通过）"""
        return [obstacle.bbox for obstacle in self.marine_obstacles]
    
    def get_safe_spawn_positions(self, num_positions: int = 10, 
                               charging_stations: List[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """
        获取安全的智能体生成位置（优先在充电桩附近）
        
        Args:
            num_positions: 需要的位置数量
            charging_stations: 充电桩位置列表
            
        Returns:
            安全位置列表
        """
        safe_positions = []
        
        # 如果有充电桩信息，优先在充电桩附近生成位置
        if charging_stations and len(charging_stations) > 0:
            print(f"🔌 基于 {len(charging_stations)} 个充电桩生成智能体位置")
            
            for i, station_pos in enumerate(charging_stations):
                if len(safe_positions) >= num_positions:
                    break
                
                station_x, station_y = station_pos
                
                # 在每个充电桩周围生成2-3个位置
                positions_per_station = min(3, num_positions - len(safe_positions))
                
                for j in range(positions_per_station):
                    # 在充电桩周围60-120像素范围内生成位置
                    for attempt in range(50):  # 最多尝试50次
                        angle = np.random.uniform(0, 2 * np.pi)
                        distance = np.random.uniform(60, 120)
                        
                        agent_x = station_x + int(distance * np.cos(angle))
                        agent_y = station_y + int(distance * np.sin(angle))
                        
                        # 确保在图像范围内
                        agent_x = max(30, min(self.image_size[0] - 30, agent_x))
                        agent_y = max(30, min(self.image_size[1] - 30, agent_y))
                        
                        pos = (agent_x, agent_y)
                        
                        # 检查是否安全且与已有位置不冲突
                        if (self._is_position_safe(pos, min_distance=40) and 
                            self._is_far_from_existing(pos, safe_positions, min_distance=80)):
                            safe_positions.append(pos)
                            print(f"  充电桩{i+1}附近: {pos}")
                            break
        
        # 如果充电桩附近位置不够，使用场景特定的战略位置
        if len(safe_positions) < num_positions:
            strategic_positions = self._get_scene_strategic_positions()
            
            # 只在第一次调用时输出战略位置信息
            if not hasattr(self, '_strategic_positions_shown'):
                print(f"🎯 使用战略位置补充剩余 {num_positions - len(safe_positions)} 个位置")
                self._strategic_positions_shown = True
            
            for pos in strategic_positions:
                if len(safe_positions) >= num_positions:
                    break
                    
                if (self._is_position_safe(pos, min_distance=30) and 
                    self._is_far_from_existing(pos, safe_positions, min_distance=80)):
                    safe_positions.append(pos)
        
        # 如果还不够，随机生成补充
        max_attempts = 500
        attempts = 0
        
        while len(safe_positions) < num_positions and attempts < max_attempts:
            attempts += 1
            
            # 在海域中随机生成位置
            x = np.random.randint(50, self.image_size[0] - 50)
            y = np.random.randint(50, self.image_size[1] - 50)
            
            pos = (x, y)
            if (self._is_position_safe(pos, min_distance=30) and 
                self._is_far_from_existing(pos, safe_positions, min_distance=60)):
                safe_positions.append(pos)
        
        # 只在第一次调用时输出详细信息，避免重复输出
        if not hasattr(self, '_positions_generated'):
            print(f"✅ 生成 {len(safe_positions)} 个安全位置（充电桩附近优先）")
            for i, pos in enumerate(safe_positions):
                if i < 5:  # 只显示前5个位置
                    print(f"  位置{i+1}: {pos}")
            self._positions_generated = True
        
        return safe_positions[:num_positions]
    
    def _get_scene_strategic_positions(self) -> List[Tuple[int, int]]:
        """获取场景特定的战略位置"""
        # 根据不同场景返回不同的战略位置
        if self.scene_name == "xinghua_bay_wind_farm":
            return [
                (150, 350),   # 左侧海域
                (450, 80),    # 北侧海域  
                (880, 700),   # 右侧海域
                (300, 900),   # 南侧海域
                (700, 150),   # 东北海域
                (150, 800),   # 西南海域
            ]
        elif self.scene_name == "fuqing_haitan_wind_farm":
            return [
                (200, 200),   # 左上海域
                (800, 200),   # 右上海域
                (200, 800),   # 左下海域
                (800, 800),   # 右下海域
                (500, 500),   # 中心海域
            ]
        elif self.scene_name == "pingtan_wind_farm":
            return [
                (300, 300),   # 西北海域
                (700, 300),   # 东北海域
                (300, 700),   # 西南海域
                (700, 700),   # 东南海域
                (500, 150),   # 北部海域
            ]
        elif self.scene_name == "putian_nanri_wind_farm_phase1_2":
            return [
                (150, 500),   # 西侧海域
                (850, 500),   # 东侧海域
                (500, 150),   # 北侧海域
                (500, 850),   # 南侧海域
                (350, 350),   # 西北海域
                (650, 650),   # 东南海域
            ]
        else:
            # 默认位置
            return [
                (150, 350),   
                (450, 80),    
                (880, 700),   
                (300, 900),   
            ]
    
    def _is_far_from_existing(self, pos: Tuple[int, int], 
                            existing_positions: List[Tuple[int, int]], 
                            min_distance: int = 60) -> bool:
        """检查位置是否与已有位置保持足够距离"""
        x, y = pos
        
        for existing_x, existing_y in existing_positions:
            distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
            if distance < min_distance:
                return False
        
        return True
    
    def _is_position_safe(self, pos: Tuple[int, int], min_distance: int = 50) -> bool:
        """
        检查位置是否安全
        
        Args:
            pos: 位置坐标
            min_distance: 最小安全距离
            
        Returns:
            是否安全
        """
        x, y = pos
        
        # 检查与陆地的距离
        for land in self.land_areas:
            if self._point_in_bbox(pos, land.bbox, min_distance):
                return False
        
        # 检查与风机的距离
        for turbine in self.wind_turbines:
            if self._point_in_bbox(pos, turbine.bbox, min_distance):
                return False
        
        # 检查与海洋障碍物的距离（只对USV生成位置时考虑）
        # 注意：UAV可以飞越海洋障碍物，所以这里不严格限制
        
        return True
    
    def _point_in_bbox(self, point: Tuple[int, int], bbox: Tuple[int, int, int, int], 
                      margin: int = 0) -> bool:
        """检查点是否在边界框内（含边距）"""
        x, y = point
        xmin, ymin, xmax, ymax = bbox
        
        return (xmin - margin <= x <= xmax + margin and 
                ymin - margin <= y <= ymax + margin)
    
    def is_path_blocked_for_usv(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """
        检查USV的路径是否被海洋障碍物阻挡
        
        Args:
            start: 起始位置
            end: 结束位置
            
        Returns:
            路径是否被阻挡
        """
        # 简化的线段与矩形相交检测
        for obstacle in self.marine_obstacles:
            if self._line_intersects_bbox(start, end, obstacle.bbox):
                return True
        
        return False
    
    def _line_intersects_bbox(self, p1: Tuple[int, int], p2: Tuple[int, int], 
                            bbox: Tuple[int, int, int, int]) -> bool:
        """检查线段是否与边界框相交"""
        # 简化实现：检查线段端点是否在边界框内，或边界框顶点是否在线段两侧
        x1, y1 = p1
        x2, y2 = p2
        xmin, ymin, xmax, ymax = bbox
        
        # 检查线段端点是否在边界框内
        if self._point_in_bbox(p1, bbox) or self._point_in_bbox(p2, bbox):
            return True
        
        # 更复杂的相交检测可以后续实现
        # 这里使用简化版本
        
        return False
    
    def create_visualization_data(self) -> Dict[str, Any]:
        """
        创建可视化数据字典
        
        Returns:
            包含所有场景信息的字典
        """
        return {
            'scene_name': self.scene_name,
            'image_path': str(self.image_path),
            'image_size': self.image_size,
            'wind_turbines': [
                {
                    'id': t.turbine_id,
                    'name': t.name,
                    'center': t.center,
                    'bbox': t.bbox,
                    'status': t.status,
                    'priority': t.priority
                }
                for t in self.wind_turbines
            ],
            'land_areas': [
                {
                    'id': l.land_id,
                    'name': l.name,
                    'bbox': l.bbox
                }
                for l in self.land_areas
            ],
            'marine_obstacles': [
                {
                    'id': o.obstacle_id,
                    'name': o.name,
                    'bbox': o.bbox,
                    'is_truncated': o.is_truncated
                }
                for o in self.marine_obstacles
            ],
            'safe_positions': self.get_safe_spawn_positions(20)
        }


def test_enhanced_scene_parser():
    """测试增强版场景解析器"""
    import os
    from pathlib import Path
    
    # 获取项目根目录
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    data_dir = project_root / "data"
    
    print(f"数据目录: {data_dir}")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 测试所有支持的场景
    for scene_name in EnhancedSceneParser.SUPPORTED_SCENES:
        print(f"\n{'='*60}")
        print(f"测试场景: {scene_name}")
        print(f"{'='*60}")
        
        # 创建解析器并加载场景
        parser = EnhancedSceneParser(scene_name)
        
        if parser.load_scene(data_dir):
            # 创建可视化数据
            viz_data = parser.create_visualization_data()
            
            print("\n🎯 可视化数据概览:")
            print(f"风机数量: {len(viz_data['wind_turbines'])}")
            print(f"陆地区域: {len(viz_data['land_areas'])}")
            print(f"海洋障碍物: {len(viz_data['marine_obstacles'])}")
            print(f"安全位置: {len(viz_data['safe_positions'])}")
            
            # 显示前几个风机位置
            if viz_data['wind_turbines']:
                print("\n🌀 风机位置 (前3个):")
                for i, turbine in enumerate(viz_data['wind_turbines'][:3]):
                    print(f"  {turbine['name']}: {turbine['center']}")
            
            # 显示海洋障碍物信息
            if viz_data['marine_obstacles']:
                print(f"\n🚫 海洋障碍物 (前3个):")
                for i, obstacle in enumerate(viz_data['marine_obstacles'][:3]):
                    print(f"  {obstacle['name']}: {obstacle['bbox']} {'[截断]' if obstacle['is_truncated'] else ''}")
            
            print(f"\n✅ 场景 {scene_name} 解析测试成功！")
        else:
            print(f"\n❌ 场景 {scene_name} 解析测试失败！")
    
    # 测试中文名映射
    print(f"\n{'='*60}")
    print("测试中文名映射")
    print(f"{'='*60}")
    
    parser_cn = EnhancedSceneParser("兴化湾海上风电场")
    print(f"中文名 '兴化湾海上风电场' 映射到: {parser_cn.scene_name}")
    
    return True


if __name__ == "__main__":
    test_enhanced_scene_parser()