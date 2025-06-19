"""
配置加载器
统一的配置文件管理和加载机制
"""
import yaml
import os
from typing import Dict, Any, Optional, Union
from pathlib import Path
import copy


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_dir: str = "src/config"):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def load_config(self, 
                   config_name: str = "default",
                   override_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_name: 配置文件名（不含扩展名）
            override_config: 覆盖配置字典
            
        Returns:
            配置字典
        """
        # 从缓存中获取
        if config_name in self._cache:
            base_config = copy.deepcopy(self._cache[config_name])
        else:
            # 加载配置文件
            config_path = self.config_dir / f"{config_name}.yaml"
            
            if not config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                base_config = yaml.safe_load(f)
            
            # 缓存配置
            self._cache[config_name] = copy.deepcopy(base_config)
        
        # 应用覆盖配置
        if override_config:
            base_config = self._merge_configs(base_config, override_config)
        
        # 环境变量覆盖
        base_config = self._apply_env_overrides(base_config)
        
        return base_config
    
    def _merge_configs(self, 
                      base_config: Dict[str, Any], 
                      override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并配置字典
        
        Args:
            base_config: 基础配置
            override_config: 覆盖配置
            
        Returns:
            合并后的配置
        """
        result = copy.deepcopy(base_config)
        
        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用环境变量覆盖
        环境变量格式: HYBRID_SECTION_KEY=value
        
        Args:
            config: 配置字典
            
        Returns:
            应用环境变量后的配置
        """
        result = copy.deepcopy(config)
        
        for env_key, env_value in os.environ.items():
            if env_key.startswith('HYBRID_'):
                # 解析环境变量路径
                path_parts = env_key[7:].lower().split('_')  # 去掉HYBRID_前缀
                
                if len(path_parts) >= 2:
                    section = path_parts[0]
                    key = '_'.join(path_parts[1:])
                    
                    if section in result:
                        # 尝试类型转换
                        try:
                            if env_value.lower() in ('true', 'false'):
                                typed_value = env_value.lower() == 'true'
                            elif '.' in env_value:
                                typed_value = float(env_value)
                            else:
                                typed_value = int(env_value)
                        except ValueError:
                            typed_value = env_value
                        
                        result[section][key] = typed_value
        
        return result
    
    def save_config(self, config: Dict[str, Any], filename: str):
        """
        保存配置到文件
        
        Args:
            config: 配置字典
            filename: 文件名
        """
        config_path = self.config_dir / filename
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        验证配置的有效性
        
        Args:
            config: 配置字典
            
        Returns:
            是否有效
        """
        required_sections = ['agents', 'environment', 'planner', 'scheduler']
        
        # 检查必需的配置段
        for section in required_sections:
            if section not in config:
                print(f"缺少必需的配置段: {section}")
                return False
        
        # 检查智能体配置
        agents_config = config['agents']
        if 'uav' not in agents_config or 'usv' not in agents_config:
            print("智能体配置必须包含UAV和USV")
            return False
        
        # 检查数值范围
        uav_config = agents_config['uav']
        if uav_config.get('max_speed', 0) <= 0:
            print("UAV最大速度必须大于0")
            return False
        
        if uav_config.get('battery_capacity', 0) <= 0:
            print("UAV电池容量必须大于0")
            return False
        
        # 检查调度器配置
        scheduler_config = config['scheduler']
        if scheduler_config.get('learning_rate', 0) <= 0:
            print("学习率必须大于0")
            return False
        
        return True
    
    def get_section(self, config: Dict[str, Any], section: str) -> Dict[str, Any]:
        """
        获取配置的特定段
        
        Args:
            config: 配置字典
            section: 段名
            
        Returns:
            配置段
        """
        return config.get(section, {})
    
    def update_section(self, 
                      config: Dict[str, Any], 
                      section: str, 
                      updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新配置的特定段
        
        Args:
            config: 原配置字典
            section: 段名
            updates: 更新内容
            
        Returns:
            更新后的配置
        """
        result = copy.deepcopy(config)
        
        if section not in result:
            result[section] = {}
        
        result[section].update(updates)
        return result


# 全局配置实例
config_loader = ConfigLoader()


def load_default_config() -> Dict[str, Any]:
    """加载默认配置"""
    from .config_validator import ensure_numeric_types
    config = config_loader.load_config("default")
    # 确保数值类型正确
    return ensure_numeric_types(config)


def load_experiment_config(experiment_name: str) -> Dict[str, Any]:
    """
    加载实验配置
    
    Args:
        experiment_name: 实验名称
        
    Returns:
        实验配置
    """
    try:
        return config_loader.load_config(experiment_name)
    except FileNotFoundError:
        print(f"实验配置文件不存在: {experiment_name}.yaml，使用默认配置")
        return load_default_config()


def create_experiment_config(base_config_name: str = "default",
                           experiment_name: str = "experiment",
                           modifications: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    创建实验配置
    
    Args:
        base_config_name: 基础配置名
        experiment_name: 实验名称
        modifications: 修改内容
        
    Returns:
        实验配置
    """
    base_config = config_loader.load_config(base_config_name)
    
    if modifications:
        experiment_config = config_loader._merge_configs(base_config, modifications)
    else:
        experiment_config = base_config
    
    # 保存实验配置
    config_loader.save_config(experiment_config, f"{experiment_name}.yaml")
    
    return experiment_config


# 演示用法
if __name__ == "__main__":
    print("=== 配置管理系统测试 ===")
    
    # 加载默认配置
    config = load_default_config()
    print(f"系统名称: {config['system']['name']}")
    print(f"UAV数量: {config['agents']['uav']['count']}")
    print(f"USV数量: {config['agents']['usv']['count']}")
    
    # 验证配置
    is_valid = config_loader.validate_config(config)
    print(f"配置有效性: {is_valid}")
    
    # 创建实验配置
    modifications = {
        'agents': {
            'uav': {'count': 5},
            'usv': {'count': 2}
        },
        'scheduler': {
            'learning_rate': 5e-4
        }
    }
    
    exp_config = create_experiment_config(
        base_config_name="default",
        experiment_name="test_experiment",
        modifications=modifications
    )
    
    print(f"\\n实验配置创建完成:")
    print(f"UAV数量: {exp_config['agents']['uav']['count']}")
    print(f"USV数量: {exp_config['agents']['usv']['count']}")
    print(f"学习率: {exp_config['scheduler']['learning_rate']}")
    
    # 测试环境变量覆盖
    os.environ['HYBRID_AGENTS_UAV_COUNT'] = '10'
    config_with_env = config_loader.load_config("default")
    print(f"\\n环境变量覆盖后UAV数量: {config_with_env['agents']['uav']['count']}")
    
    print("\\n配置管理系统测试完成！")