"""
配置验证和类型转换工具
"""
from typing import Dict, Any, Union


def ensure_numeric_types(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    确保配置中的数值类型正确
    
    Args:
        config: 原始配置字典
        
    Returns:
        类型修正后的配置
    """
    # 定义需要确保为数值类型的配置项
    numeric_fields = {
        'scheduler': {
            'learning_rate': float,
            'gamma': float,
            'epsilon': float,
            'epsilon_min': float,
            'epsilon_decay': float,
            'target_update_freq': int,
            'state_dim': int,
            'action_dim': int,
            'buffer_size': int,
            'batch_size': int,
            'max_episodes': int,
            'max_steps_per_episode': int,
            'eval_freq': int
        },
        'planner': {
            'grid_resolution': float,
            'time_weight': float,
            'energy_weight': float,
            'safety_margin': float,
            'max_iterations': int
        },
        'optimizer': {
            'max_iterations': int,
            'improvement_threshold': float,
            'time_weight': float,
            'energy_weight': float,
            'random_seed': int
        },
        'agents': {
            'uav': {
                'count': int,
                'max_speed': float,
                'battery_capacity': float,
                'base_power': float,
                'task_power': float,
                'cruise_speed_ratio': float
            },
            'usv': {
                'count': int,
                'max_speed': float,
                'battery_capacity': float,
                'base_power': float,
                'task_power': float,
                'cruise_speed_ratio': float
            }
        },
        'environment': {
            'map_width': int,
            'map_height': int,
            'real_scale': float
        }
    }
    
    def convert_value(value: Any, target_type: type) -> Any:
        """转换单个值的类型"""
        if value is None:
            return value
        
        try:
            if target_type == float:
                return float(value)
            elif target_type == int:
                return int(value)
            else:
                return value
        except (ValueError, TypeError):
            print(f"警告: 无法将 {value} 转换为 {target_type}")
            return value
    
    def process_dict(config_dict: Dict[str, Any], type_spec: Dict[str, Any]) -> Dict[str, Any]:
        """递归处理配置字典"""
        result = config_dict.copy()
        
        for key, value in config_dict.items():
            if key in type_spec:
                if isinstance(type_spec[key], dict) and isinstance(value, dict):
                    # 递归处理子字典
                    result[key] = process_dict(value, type_spec[key])
                elif isinstance(type_spec[key], type):
                    # 转换类型
                    result[key] = convert_value(value, type_spec[key])
        
        return result
    
    # 处理配置
    validated_config = process_dict(config, numeric_fields)
    
    # 特殊处理hidden_dims（应该是整数列表）
    if 'scheduler' in validated_config and 'hidden_dims' in validated_config['scheduler']:
        dims = validated_config['scheduler']['hidden_dims']
        if isinstance(dims, list):
            validated_config['scheduler']['hidden_dims'] = [int(d) for d in dims]
    
    return validated_config


def validate_scheduler_config(config: Dict[str, Any]) -> bool:
    """
    验证调度器配置的有效性
    
    Args:
        config: 调度器配置
        
    Returns:
        是否有效
    """
    required_fields = [
        'learning_rate', 'gamma', 'epsilon', 'epsilon_min', 
        'epsilon_decay', 'buffer_size', 'batch_size'
    ]
    
    for field in required_fields:
        if field not in config:
            print(f"缺少必需字段: {field}")
            return False
        
        value = config[field]
        
        # 检查类型
        if field in ['learning_rate', 'gamma', 'epsilon', 'epsilon_min', 'epsilon_decay']:
            if not isinstance(value, (int, float)):
                print(f"{field} 应该是数值类型，但得到 {type(value)}")
                return False
            
            # 检查范围
            if field in ['gamma', 'epsilon', 'epsilon_min', 'epsilon_decay']:
                if not 0 <= float(value) <= 1:
                    print(f"{field} 应该在 [0, 1] 范围内，但得到 {value}")
                    return False
        
        elif field in ['buffer_size', 'batch_size']:
            if not isinstance(value, int) or value <= 0:
                print(f"{field} 应该是正整数，但得到 {value}")
                return False
    
    return True