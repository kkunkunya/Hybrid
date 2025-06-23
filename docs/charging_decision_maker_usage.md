# 充电决策优化器使用指南

## 概述

充电决策优化器（ChargingDecisionMaker）是多UAV-USV协同巡检系统的Layer 3组件，负责为需要充电的UAV智能选择最优充电方式。

## 核心功能

### 1. 多因子决策评估
- **距离因子 (30%)**：评估UAV到充电设施的距离
- **排队因子 (25%)**：预测充电桩排队时间
- **可用性因子 (25%)**：评估充电资源的可用性
- **紧急度因子 (20%)**：基于UAV电量和任务重要性

### 2. 充电选项
- **充电桩充电**：固定充电站，功率高但可能需要排队
- **USV支援充电**：移动充电，无需排队但功率较低
- **暂不充电**：电量充足时的选择

### 3. 动态阈值调整
- 根据系统负载动态调整决策阈值
- 高负载时优先使用充电桩
- 排队严重时提高USV支援权重

## 使用示例

```python
from src.scheduler import ChargingDecisionMaker, ChargingStation

# 创建决策器
config = {
    'charging_decision': {
        'decision_factors': {
            'distance': 0.3,
            'queue_time': 0.25,
            'availability': 0.25,
            'urgency': 0.2
        },
        'queue_prediction_window': 900,
        'emergency_threshold': 0.1
    }
}

decision_maker = ChargingDecisionMaker(config)

# UAV状态
uav_state = {
    'id': 'uav_001',
    'position': (500, 500),
    'current_energy': 45,  # 当前电量
    'max_energy': 300,     # 最大电量
    'energy_consumption_rate': 15.0,  # 能耗率 Wh/min
    'current_task_priority': 2.5      # 任务优先级
}

# 充电桩信息
charging_stations = [
    ChargingStation(
        station_id='station_01',
        position=(300, 400),
        capacity=2,
        queue=['uav_002']  # 当前排队UAV
    )
]

# USV状态
usv_states = {
    'usv_001': {
        'position': (600, 450),
        'available_energy': 200,
        'max_energy': 1000,
        'status': 'available'
    }
}

# 系统状态
system_state = {
    'timestamp': time.time(),
    'total_uavs': 10,
    'charging_uavs': 3,
    'avg_station_queue': 1.5,
    'available_usvs': 2
}

# 做出充电决策
decision = decision_maker.make_decision(
    uav_state, 
    charging_stations, 
    usv_states, 
    system_state
)

print(f"充电选项: {decision.option.value}")
print(f"目标: {decision.target_id}")
print(f"预计时间: {decision.estimated_time/60:.1f}分钟")
```

## 决策流程

1. **紧急度评估**
   - 基于当前电量百分比
   - 考虑能耗速率和剩余飞行时间
   - 任务优先级加权

2. **选项评估**
   - 计算到每个充电设施的成本
   - 预测排队等待时间
   - 评估资源可用性

3. **综合决策**
   - 加权计算各选项得分
   - 紧急情况优先选择时间最短
   - 应用动态阈值调整

## 配置参数

### 决策因子权重
```yaml
decision_factors:
  distance: 0.3      # 距离权重
  queue_time: 0.25   # 排队时间权重
  availability: 0.25 # 可用性权重
  urgency: 0.2       # 紧急度权重
```

### 系统参数
```yaml
queue_prediction_window: 900  # 排队预测窗口（秒）
emergency_threshold: 0.1      # 紧急电量阈值（10%）
charging_station_capacity: 2  # 充电桩默认容量
usv_charging_rate: 50        # USV充电功率（W）
station_charging_power: 100  # 充电桩功率（W）
```

## 高级特性

### 1. 排队时间预测
- 基于历史充电时间数据
- 考虑当前队列长度
- 系统负载趋势调整

### 2. 动态权重调整
- 系统繁忙时降低USV充电阈值
- 充电桩排队严重时提高USV权重
- USV不足时优先使用充电桩

### 3. 历史数据学习
```python
# 更新历史数据
decision_maker.update_history(
    uav_id='uav_001',
    charging_option=ChargingOption.CHARGING_STATION,
    actual_time=1800.0  # 实际充电时间
)

# 获取系统统计
stats = decision_maker.get_system_statistics()
```

## 最佳实践

1. **定期更新系统状态**
   - 保持充电桩队列信息实时
   - 更新USV位置和可用能量
   - 监控系统负载变化

2. **历史数据维护**
   - 定期清理过期数据
   - 验证预测准确性
   - 调整预测参数

3. **紧急情况处理**
   - 设置合理的紧急阈值
   - 确保紧急UAV优先级
   - 预留应急充电资源

## 性能优化

1. **缓存优化**
   - 缓存距离计算结果
   - 复用排队预测结果
   - 批量处理决策请求

2. **计算优化**
   - 使用向量化计算
   - 并行评估多个选项
   - 提前剪枝不可行选项

## 故障处理

1. **充电设施故障**
   - 自动排除故障设施
   - 重新评估可用选项
   - 记录故障历史

2. **通信中断**
   - 使用最后已知状态
   - 降级到保守决策
   - 触发应急预案

## 扩展接口

决策器提供扩展接口，支持自定义：
- 决策因子
- 评分算法
- 预测模型
- 阈值策略

## 集成示例

与其他系统组件集成：

```python
# 与能源感知调度器集成
from src.scheduler import EnergyAwareScheduler

scheduler = EnergyAwareScheduler(config)
charging_decision_maker = ChargingDecisionMaker(config)

# 在调度循环中
for uav in uavs_needing_charge:
    decision = charging_decision_maker.make_decision(
        uav.get_state(),
        charging_stations,
        usv_states,
        system_state
    )
    
    if decision.option != ChargingOption.NONE:
        scheduler.assign_charging_task(uav, decision)
```

## 监控指标

关键性能指标：
- 平均决策时间
- 充电等待时间
- 决策准确率
- 系统利用率
- 紧急响应成功率

## 故障排查

常见问题：
1. **决策总是选择充电桩**
   - 检查USV可用性
   - 验证距离计算
   - 调整权重配置

2. **排队预测不准确**
   - 增加历史数据量
   - 调整预测窗口
   - 校准充电时间

3. **紧急UAV未优先处理**
   - 降低紧急阈值
   - 增加紧急度权重
   - 检查优先级计算