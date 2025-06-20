"""
高精度岛屿检测器
使用多种图像处理技术精确识别岛屿位置
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt


class AdvancedIslandDetector:
    """高精度岛屿检测器"""
    
    def __init__(self):
        """初始化检测器"""
        pass
    
    def detect_islands_precise(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        精确检测岛屿位置
        
        Args:
            image: 输入图像
            
        Returns:
            岛屿信息列表
        """
        try:
            print("🏝️ 开始精确岛屿检测...")
            
            # 多种方法组合检测
            islands_hsv = self._detect_by_hsv(image)
            islands_lab = self._detect_by_lab(image)
            islands_edge = self._detect_by_edges(image)
            
            # 合并结果
            all_islands = islands_hsv + islands_lab + islands_edge
            
            # 去重和优化
            final_islands = self._merge_and_filter_islands(all_islands, image.shape)
            
            print(f"✅ 检测到 {len(final_islands)} 个高精度岛屿")
            
            return final_islands
            
        except Exception as e:
            print(f"❌ 岛屿检测失败: {e}")
            return []
    
    def _detect_by_hsv(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """基于HSV色彩空间检测"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 更精确的陆地颜色范围
        land_masks = []
        
        # 棕色陆地
        lower_brown = np.array([8, 50, 80])
        upper_brown = np.array([25, 200, 180])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # 灰褐色陆地
        lower_gray_brown = np.array([15, 30, 100])
        upper_gray_brown = np.array([35, 150, 200])
        mask_gray_brown = cv2.inRange(hsv, lower_gray_brown, upper_gray_brown)
        
        # 浅棕色
        lower_light_brown = np.array([10, 40, 120])
        upper_light_brown = np.array([30, 180, 220])
        mask_light_brown = cv2.inRange(hsv, lower_light_brown, upper_light_brown)
        
        # 合并掩码
        combined_mask = cv2.bitwise_or(mask_brown, mask_gray_brown)
        combined_mask = cv2.bitwise_or(combined_mask, mask_light_brown)
        
        return self._extract_islands_from_mask(combined_mask, "HSV")
    
    def _detect_by_lab(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """基于LAB色彩空间检测"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # LAB空间中的陆地特征
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # 陆地通常在a通道和b通道有特定范围
        # a通道: 绿色到红色
        # b通道: 蓝色到黄色
        
        # 创建陆地掩码
        mask_a = cv2.inRange(a_channel, 125, 140)  # 偏红
        mask_b = cv2.inRange(b_channel, 130, 150)  # 偏黄
        mask_l = cv2.inRange(l_channel, 80, 180)   # 中等亮度
        
        combined_mask = cv2.bitwise_and(mask_a, mask_b)
        combined_mask = cv2.bitwise_and(combined_mask, mask_l)
        
        return self._extract_islands_from_mask(combined_mask, "LAB")
    
    def _detect_by_edges(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """基于边缘检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 形态学操作连接边缘
        kernel = np.ones((3, 3), np.uint8)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 填充封闭区域
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros_like(gray)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 20000:  # 合理的岛屿大小
                cv2.fillPoly(mask, [contour], 255)
        
        return self._extract_islands_from_mask(mask, "Edge")
    
    def _extract_islands_from_mask(self, mask: np.ndarray, method: str) -> List[Dict[str, Any]]:
        """从掩码中提取岛屿"""
        # 形态学操作清理
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        islands = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # 过滤太小或太大的区域
            if 1000 < area < 50000:
                # 计算边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算中心点
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                else:
                    center_x = x + w // 2
                    center_y = y + h // 2
                
                island_info = {
                    'method': method,
                    'center': (center_x, center_y),
                    'bbox': (x, y, x + w, y + h),
                    'area': area,
                    'contour': contour
                }
                
                islands.append(island_info)
        
        print(f"  {method}方法检测到 {len(islands)} 个候选岛屿")
        return islands
    
    def _merge_and_filter_islands(self, all_islands: List[Dict[str, Any]], 
                                image_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """合并和过滤岛屿"""
        if not all_islands:
            return []
        
        # 按位置聚类，合并相近的检测结果
        clusters = []
        
        for island in all_islands:
            center = island['center']
            
            # 寻找相近的聚类
            found_cluster = False
            for cluster in clusters:
                cluster_center = cluster['center']
                distance = np.sqrt((center[0] - cluster_center[0])**2 + 
                                 (center[1] - cluster_center[1])**2)
                
                if distance < 100:  # 100像素内认为是同一个岛
                    # 选择面积更大的
                    if island['area'] > cluster['area']:
                        clusters[clusters.index(cluster)] = island
                    found_cluster = True
                    break
            
            if not found_cluster:
                clusters.append(island)
        
        # 进一步过滤和验证
        final_islands = []
        
        for island in clusters:
            center_x, center_y = island['center']
            
            # 确保岛屿在图像范围内
            if (50 < center_x < image_shape[1] - 50 and 
                50 < center_y < image_shape[0] - 50):
                
                final_islands.append(island)
        
        # 按面积排序，选择最大的几个
        final_islands.sort(key=lambda x: x['area'], reverse=True)
        
        return final_islands[:3]  # 最多返回3个最大的岛屿
    
    def get_optimal_charging_positions(self, islands: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        """获取最优充电桩位置"""
        charging_positions = []
        
        for i, island in enumerate(islands):
            center_x, center_y = island['center']
            bbox = island['bbox']
            
            # 在岛屿上找最合适的位置
            # 优先选择岛屿内部较平坦的区域
            
            # 候选位置：岛屿中心附近
            candidates = [
                # 岛屿中心
                (center_x, center_y),
                # 岛屿中心偏移位置
                (center_x + 10, center_y),
                (center_x - 10, center_y),
                (center_x, center_y + 10),
                (center_x, center_y - 10),
                # 边界框内的稳定位置
                (bbox[0] + (bbox[2] - bbox[0]) // 3, bbox[1] + (bbox[3] - bbox[1]) // 3),
                (bbox[0] + 2*(bbox[2] - bbox[0]) // 3, bbox[1] + 2*(bbox[3] - bbox[1]) // 3)
            ]
            
            # 选择最靠近岛屿中心的位置
            best_pos = candidates[0]  # 默认使用中心位置
            
            charging_positions.append(best_pos)
            print(f"  岛屿{i+1}充电桩: {best_pos} (岛屿中心: {island['center']})")
        
        return charging_positions
    
    def visualize_detection_process(self, image: np.ndarray, islands: List[Dict[str, Any]], 
                                  charging_positions: List[Tuple[int, int]], 
                                  save_path: str = None):
        """可视化检测过程"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            
            # 原始图像
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axes[0, 0].imshow(image_rgb)
            axes[0, 0].set_title('原始图像', fontsize=14)
            axes[0, 0].axis('off')
            
            # HSV检测结果
            hsv_islands = self._detect_by_hsv(image)
            axes[0, 1].imshow(image_rgb)
            for island in hsv_islands:
                center = island['center']
                bbox = island['bbox']
                rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                   linewidth=2, edgecolor='red', facecolor='none')
                axes[0, 1].add_patch(rect)
                axes[0, 1].plot(center[0], center[1], 'ro', markersize=8)
            axes[0, 1].set_title(f'HSV检测 ({len(hsv_islands)}个)', fontsize=14)
            axes[0, 1].axis('off')
            
            # LAB检测结果
            lab_islands = self._detect_by_lab(image)
            axes[1, 0].imshow(image_rgb)
            for island in lab_islands:
                center = island['center']
                bbox = island['bbox']
                rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                   linewidth=2, edgecolor='blue', facecolor='none')
                axes[1, 0].add_patch(rect)
                axes[1, 0].plot(center[0], center[1], 'bo', markersize=8)
            axes[1, 0].set_title(f'LAB检测 ({len(lab_islands)}个)', fontsize=14)
            axes[1, 0].axis('off')
            
            # 最终结果
            axes[1, 1].imshow(image_rgb)
            
            # 绘制检测到的岛屿
            for i, island in enumerate(islands):
                center = island['center']
                bbox = island['bbox']
                
                # 岛屿边界
                rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                   linewidth=3, edgecolor='green', facecolor='green', alpha=0.3)
                axes[1, 1].add_patch(rect)
                
                # 岛屿中心
                axes[1, 1].plot(center[0], center[1], 'go', markersize=10)
                axes[1, 1].text(center[0], center[1]-30, f'岛屿{i+1}', ha='center', 
                               fontsize=12, fontweight='bold', 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # 绘制充电桩位置
            for i, pos in enumerate(charging_positions):
                # 充电桩图标
                rect = plt.Rectangle((pos[0]-8, pos[1]-8), 16, 16,
                                   linewidth=3, edgecolor='darkgreen', facecolor='lightgreen')
                axes[1, 1].add_patch(rect)
                axes[1, 1].text(pos[0], pos[1]-25, f'⚡C{i}', ha='center', 
                               fontsize=12, fontweight='bold')
            
            axes[1, 1].set_title(f'最终结果: {len(islands)}个岛屿', fontsize=14)
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ 检测过程可视化已保存: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ 可视化失败: {e}")


def test_advanced_island_detection():
    """测试高精度岛屿检测"""
    try:
        from pathlib import Path
        
        # 获取项目根目录
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        
        # 加载图像
        image_path = project_root / "data" / "scenes" / "xinghua_bay_wind_farm.png"
        
        if not image_path.exists():
            print(f"❌ 图像文件不存在: {image_path}")
            return False
        
        # 加载图像
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            print("❌ 图像加载失败")
            return False
        
        print(f"✅ 图像加载成功: {image.shape}")
        
        # 创建检测器
        detector = AdvancedIslandDetector()
        
        # 检测岛屿
        islands = detector.detect_islands_precise(image)
        
        if not islands:
            print("❌ 未检测到岛屿")
            return False
        
        # 获取充电桩位置
        charging_positions = detector.get_optimal_charging_positions(islands)
        
        print(f"\\n✅ 检测结果:")
        for i, (island, pos) in enumerate(zip(islands, charging_positions)):
            print(f"  岛屿{i+1}: 中心{island['center']}, 面积{island['area']:.0f}, 充电桩{pos}")
        
        # 可视化
        save_path = project_root / "advanced_island_detection.png"
        detector.visualize_detection_process(image, islands, charging_positions, str(save_path))
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_advanced_island_detection()