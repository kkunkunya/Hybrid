"""
简化的Pygame演示 - 解决无响应问题
"""
import pygame
import sys
import time
import numpy as np
from pathlib import Path


class SimplePygameDemo:
    """简化的Pygame演示"""
    
    def __init__(self, window_size=(800, 600)):
        """初始化"""
        self.window_size = window_size
        self.screen = None
        self.clock = None
        self.font = None
        self.is_running = False
        
    def initialize(self):
        """初始化Pygame"""
        try:
            print("🎮 初始化简化Pygame演示...")
            
            # 初始化pygame
            pygame.init()
            pygame.font.init()
            
            # 创建窗口
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("简化Pygame测试")
            
            # 创建时钟和字体
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            
            print("✅ Pygame初始化成功")
            return True
            
        except Exception as e:
            print(f"❌ Pygame初始化失败: {e}")
            return False
    
    def run_simple_test(self):
        """运行简单测试"""
        if not self.initialize():
            return False
        
        print("🎮 开始简单测试...")
        print("   按ESC退出，按空格改变颜色")
        
        self.is_running = True
        frame_count = 0
        color_index = 0
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        try:
            while self.is_running:
                # 处理事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.is_running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.is_running = False
                        elif event.key == pygame.K_SPACE:
                            color_index = (color_index + 1) % len(colors)
                            print(f"颜色切换到: {colors[color_index]}")
                
                # 清屏
                self.screen.fill((100, 149, 237))  # 蓝色背景
                
                # 绘制彩色圆圈
                center = (self.window_size[0] // 2, self.window_size[1] // 2)
                radius = 50 + int(20 * np.sin(frame_count * 0.1))  # 动态半径
                pygame.draw.circle(self.screen, colors[color_index], center, radius)
                
                # 绘制文本
                text = self.font.render(f"Frame: {frame_count}", True, (255, 255, 255))
                self.screen.blit(text, (10, 10))
                
                fps_text = self.font.render(f"FPS: {self.clock.get_fps():.1f}", True, (255, 255, 255))
                self.screen.blit(fps_text, (10, 50))
                
                # 更新显示
                pygame.display.flip()
                
                # 控制帧率
                self.clock.tick(60)
                frame_count += 1
                
                # 每60帧打印一次信息
                if frame_count % 60 == 0:
                    print(f"运行正常: 第{frame_count}帧")
                
        except Exception as e:
            print(f"❌ 运行时错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            pygame.quit()
            print("✅ Pygame已退出")
        
        return True


def test_pygame_functionality():
    """测试Pygame基本功能"""
    print("🧪 测试Pygame基本功能...")
    
    try:
        # 测试pygame导入
        import pygame
        print(f"✅ Pygame版本: {pygame.version.ver}")
        
        # 测试基本初始化
        pygame.init()
        
        # 测试显示模式
        info = pygame.display.Info()
        print(f"✅ 显示信息: {info.bitsize}位, {info.fmt}")
        
        # 测试字体
        pygame.font.init()
        font = pygame.font.Font(None, 24)
        print("✅ 字体系统正常")
        
        pygame.quit()
        print("✅ 基本功能测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False


def run_minimal_demo():
    """运行最小化演示"""
    print("🎮 Pygame最小化演示")
    print("=" * 40)
    
    # 测试基本功能
    if not test_pygame_functionality():
        print("❌ 基本功能测试失败，无法继续")
        return False
    
    # 运行简单演示
    demo = SimplePygameDemo()
    success = demo.run_simple_test()
    
    if success:
        print("✅ 简化演示运行成功")
    else:
        print("❌ 简化演示运行失败")
    
    return success


if __name__ == "__main__":
    run_minimal_demo()