# 多UAV-USV协同任务分配系统技术设计文档

## 📋 文档信息
- **创建日期**: 2025-06-22
- **版本**: v2.0
- **作者**: 项目开发团队
- **目的**: 详细记录任务分配系统的技术架构和实现方案
- **最新更新**: 2025-06-22 - 改进Layer 1算法以支持多任务分配

## 🎯 系统概述

### 问题定义
设计一个能源约束下的多UAV-USV协同任务分配系统，需要解决：
1. **UAV任务分配**：在电量限制下合理分配巡检任务
2. **USV后勤调度**：智能决策支援哪些需要充电的UAV
3. **充电策略优化**：UAV选择充电桩还是USV充电的决策
4. **动态重调度**：运行时根据实际情况调整分配方案

### 技术挑战
- 多约束优化：能源、时间、距离、优先级
- 实时性要求：动态环境下的快速决策
- 多智能体协调：UAV与USV之间的协同
- 不确定性处理：电量消耗、任务执行时间的预测

## 🏗️ 总体架构：分层优化算法

### 设计理念
采用**分而治之**的策略，将复杂的多约束优化问题分解为4个相互协调的子问题：

```
┌─────────────────────────────────────────────────────┐
│                   Layer 4                           │
│            动态重调度管理器                          │
│        (事件驱动 + 滑动窗口重规划)                   │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────┐
│                   Layer 3                           │
│             充电决策优化器                           │
│      (多因子启发式决策树 + 实时评分)                 │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────┐
│                   Layer 2                           │
│            USV后勤智能调度器                         │
│        (动态规划 + 启发式贪心策略)                   │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────┐
│                   Layer 1                           │
│            能源感知任务分配器                        │
│     (改进贪心多任务分配 + 能源约束)                  │
└─────────────────────────────────────────────────────┘
```

## 📐 Layer 1: 能源感知任务分配器

### 算法选择：改进的贪心多任务分配算法（v2.0更新）
**原因**：
- 支持一对多分配（一个UAV可执行多个任务）
- 时间复杂度O(n²m)，n为UAV数，m为任务数
- 考虑UAV电池容量和任务链能耗
- 实时性强，适合动态环境

### 核心创新

#### 1. 多目标成本函数设计
```python
Cost(UAV_i, Task_j) = α×距离成本 + β×能耗成本 + γ×时间成本 + δ×风险成本

其中：
- 距离成本 = euclidean_distance(uav.position, task.position)
- 能耗成本 = energy_calculator.estimate_consumption(uav, task)  
- 时间成本 = 飞行时间 + 任务执行时间
- 风险成本 = 环境风险因子(天气、障碍物等)
```

#### 2. 自适应权重机制
```python
权重调整策略：
- 系统负载高 → 提高时间权重γ，优先快速完成
- 电量普遍偏低 → 提高能耗权重β，节能优先  
- 任务优先级差异大 → 提高风险权重δ，优先重要任务
- 均衡状态 → 距离权重α最高，减少总里程
```

#### 3. 硬约束检查
```python
约束条件：
1. UAV剩余电量 >= 分配任务总需求 × (1 + 安全系数)
2. 任务执行时间窗口满足
3. UAV负载不超过最大任务数量
4. 关键任务必须分配给指定类型UAV
5. 任务链能耗累积检查（v2.0新增）
```

#### 4. 多任务分配策略（v2.0新增）
```python
改进的贪心分配流程：
1. 初始化：每个UAV分配列表为空
2. 迭代分配：
   while 存在未分配任务 and 未达到最大迭代次数:
       for each UAV:
           if UAV已达到最大任务数: continue
           计算UAV剩余能量（考虑已分配任务）
           if 剩余能量不足最小预留: continue
           找到成本最低的未分配任务
           if 可以安全执行该任务:
               分配任务给UAV
               更新UAV状态和任务状态
3. 验证和调整最终分配方案
```

### 技术实现框架
```python
class EnergyAwareTaskAllocator(BaseScheduler):
    def __init__(self, config):
        self.cost_weights = {
            "distance": 0.3, 
            "energy": 0.4, 
            "time": 0.2, 
            "risk": 0.1
        }
        self.safety_margin = 0.25  # 25%安全余量
        self.energy_calculator = EnergyCalculator()
    
    def allocate(self, uavs, tasks, environment):
        # 核心流程
        return self._execute_allocation_pipeline(uavs, tasks, environment)
```

## 📐 Layer 2: USV后勤智能调度器

### 算法选择：动态规划 + 贪心策略组合
**原因**：
- 动态规划：处理USV路径规划的最优子结构
- 贪心策略：快速响应紧急支援请求

### 核心创新

#### 1. 支援效益评估模型
```python
Support_Benefit(USV_i, UAV_j) = 拯救任务价值 / USV支援成本

其中：
- 拯救任务价值 = Σ(task.priority × task.completion_probability)
- USV支援成本 = 移动能耗 + 充电能耗 + 返回基地能耗 + 时间成本
```

#### 2. USV自身电量管理策略
```python
USV电量管理原则：
1. 预留返回基地电量：reserve_energy = distance_to_base × energy_rate × 1.3
2. 支援能力评估：available_energy = current_energy - reserve_energy  
3. 优先级排序：emergency_level × support_benefit / support_cost
4. 充电时机：当available_energy < min_support_threshold时返回充电
```

#### 3. 多USV协调机制
```python
协调策略：
- 避免多个USV支援同一UAV（除非紧急情况）
- 区域分工：根据海域划分USV责任区
- 负载均衡：支援任务数量尽量平均分配
- 备用机制：至少保持一个USV处于待命状态
```

## 📐 Layer 3: 充电决策优化器

### 算法选择：多因子启发式决策树
**原因**：
- 可解释性强，便于调试和优化
- 计算速度快，适合实时决策
- 易于添加新的决策因子

### 核心创新

#### 1. 综合评分模型
```python
决策因子权重分配：
- 距离因子 (30%)：到充电桩 vs 到最近USV的距离比较
- 排队因子 (25%)：充电桩当前排队情况和预计等待时间  
- 可用性因子 (25%)：USV当前状态和可支援能力
- 紧急度因子 (20%)：UAV当前电量百分比和剩余任务重要性
```

#### 2. 动态阈值调整
```python
阈值调整策略：
- 系统繁忙时：降低USV充电阈值，优先使用充电桩
- 充电桩排队严重：提高USV充电权重
- USV数量不足：提高充电桩使用优先级
- 紧急情况：无视排队，强制使用最近充电方式
```

#### 3. 排队时间预测算法
```python
预测模型：
1. 历史数据分析：过去N小时的充电桩使用模式
2. 当前队列状态：排队UAV数量和预计充电时间
3. 趋势预测：基于当前系统负载预测未来15分钟排队情况
4. 不确定性处理：添加置信区间，保守估计排队时间
```

## 📐 Layer 4: 动态重调度管理器

### 算法选择：事件驱动 + 滑动窗口重规划
**原因**：
- 事件驱动：及时响应系统状态变化
- 滑动窗口：平衡计算成本和规划质量

### 核心创新

#### 1. 智能触发机制
```python
触发条件层次：
级别1 - 立即触发：
  - UAV电量低于临界阈值(<15%)
  - 任务执行失败需要重新分配
  - 设备故障(UAV/USV失效)

级别2 - 延迟触发(5分钟内)：
  - 新的高优先级任务到达
  - 电量消耗超出预期20%以上
  - USV支援请求积压

级别3 - 周期性触发(15分钟)：
  - 系统性能指标下降
  - 负载不均衡严重
  - 环境条件显著变化
```

#### 2. 局部vs全局重调度决策
```python
决策规则：
局部调整(影响<30%智能体)：
  - 单个UAV电量不足 → 任务转移给邻近UAV
  - USV支援单个UAV → 调整USV路径规划
  - 充电桩排队 → 引导部分UAV到其他充电点

全局重调度(影响>=30%智能体)：
  - 多个UAV同时电量不足
  - 系统任务完成率<80%
  - 环境条件剧变(恶劣天气)
```

#### 3. 平滑过渡算法
```python
过渡策略：
1. 任务状态保护：正在执行的任务不被打断
2. 渐进式调整：分批次调整，避免系统震荡  
3. 回滚机制：新方案效果不佳时能快速回到原方案
4. 影响最小化：优先调整空闲或即将完成任务的智能体
```

## 🔧 关键算法细节

### 1. 多任务分配核心算法（v2.0更新）
```python
def solve_multi_task_assignment(uavs, tasks, environment):
    """改进的贪心多任务分配算法"""
    assignment = {uav_id: [] for uav_id in uavs}
    assigned_tasks = set()
    
    # 构建成本矩阵
    cost_matrix = build_enhanced_cost_matrix(uavs, tasks, environment)
    
    for i, uav in enumerate(uavs):
        for j, task in enumerate(tasks):
            # 基础成本计算
            distance_cost = euclidean_distance(uav.position, task.position)
            energy_cost = energy_calculator.estimate_total_consumption(uav, task)
            time_cost = distance_cost / uav.cruise_speed + task.duration
            
            # 环境风险评估
            risk_cost = environment.assess_risk(task.position, uav.type)
            
            # 自适应权重
            weights = adapt_weights(system_state)
            
            # 综合成本
            matrix[i][j] = (weights.distance * distance_cost + 
                           weights.energy * energy_cost +
                           weights.time * time_cost +
                           weights.risk * risk_cost)
    
    return matrix
```

### 2. 任务链能耗计算（v2.0新增）
```python
def calculate_task_chain_energy(uav, task_sequence):
    """计算执行任务序列的总能耗"""
    total_energy = 0
    current_pos = uav.position
    
    for task in task_sequence:
        # 移动能耗
        travel_energy = calculate_movement_energy(
            current_pos, task.position, uav.cruise_speed
        )
        # 任务执行能耗
        task_energy = calculate_task_energy(
            task.duration, task.intensity
        )
        total_energy += travel_energy + task_energy
        current_pos = task.position
    
    # 返回基地能耗
    return_energy = calculate_return_energy(current_pos, uav.home_base)
    total_energy += return_energy
    
    return total_energy
```

### 3. 约束检查算法
```python
def validate_energy_feasibility(assignment, uavs, tasks):
    """验证分配方案的能源可行性"""
    for uav_id, task_ids in assignment.items():
        uav = uavs[uav_id]
        assigned_tasks = [tasks[tid] for tid in task_ids]
        
        # 计算总能耗需求
        total_energy_needed = 0
        current_pos = uav.position
        
        for task in assigned_tasks:
            # 移动到任务点的能耗
            travel_energy = energy_calculator.calculate_movement_energy(
                uav.type, distance(current_pos, task.position), uav.cruise_speed
            )
            
            # 任务执行能耗
            task_energy = energy_calculator.calculate_task_energy(
                uav.type, task.duration, task.intensity
            )
            
            total_energy_needed += travel_energy + task_energy
            current_pos = task.position
        
        # 返回基地或最近充电点的能耗
        return_energy = min(
            energy_calculator.calculate_return_energy(current_pos, charging_stations),
            energy_calculator.calculate_return_energy(current_pos, uav.home_base)
        )
        
        total_energy_needed += return_energy
        
        # 安全余量检查
        required_energy = total_energy_needed * (1 + safety_margin)
        
        if uav.current_energy < required_energy:
            # 分配不可行，需要调整
            assignment = adjust_assignment(assignment, uav_id, task_ids)
    
    return assignment
```

### 3. USV支援效益计算
```python
def calculate_support_benefit(usv, target_uav, environment):
    """计算USV支援特定UAV的效益"""
    
    # 可拯救的任务价值
    at_risk_tasks = [task for task in target_uav.assigned_tasks 
                     if predict_task_failure_risk(target_uav, task) > 0.5]
    
    rescuable_value = sum(task.priority * task.completion_probability 
                         for task in at_risk_tasks)
    
    # USV支援成本计算
    travel_distance = distance(usv.position, target_uav.position)
    travel_energy = energy_calculator.calculate_movement_energy(
        AgentType.USV, travel_distance, usv.cruise_speed
    )
    
    # 充电能量转移成本
    charging_energy = min(target_uav.energy_deficit, usv.available_charging_capacity)
    
    # 返回基地成本
    return_distance = distance(target_uav.position, usv.home_base)
    return_energy = energy_calculator.calculate_movement_energy(
        AgentType.USV, return_distance, usv.cruise_speed
    )
    
    total_cost = travel_energy + charging_energy + return_energy
    
    # 时间成本
    time_cost = (travel_distance / usv.cruise_speed + 
                charging_energy / usv.charging_rate +
                return_distance / usv.cruise_speed)
    
    # 综合效益计算
    if total_cost > 0:
        return rescuable_value / (total_cost + time_cost * time_weight)
    else:
        return float('inf')  # 无成本支援，效益无限大
```

## 📊 性能评估体系

### 系统级指标
```python
system_metrics = {
    "task_completion_rate": completed_tasks / total_tasks,
    "total_energy_consumption": sum(agent.energy_consumed for agent in all_agents),
    "average_completion_time": mean(task.completion_time for task in completed_tasks),
    "uav_utilization_rate": sum(uav.active_time) / (len(uavs) * total_time),
    "usv_support_success_rate": successful_supports / total_support_requests,
    "system_efficiency": task_completion_rate / total_energy_consumption
}
```

### 智能体级指标
```python
agent_metrics = {
    "uav_metrics": {
        "average_flight_time": mean(uav.flight_time for uav in uavs),
        "energy_efficiency": mean(uav.tasks_completed / uav.energy_consumed),
        "charging_frequency": mean(uav.charging_count),
        "task_success_rate": mean(uav.successful_tasks / uav.assigned_tasks)
    },
    "usv_metrics": {
        "support_response_time": mean(usv.response_times),
        "support_efficiency": mean(usv.successful_supports / usv.energy_consumed),
        "availability_rate": mean(usv.available_time / total_time),
        "coverage_area": mean(usv.coverage_radius)
    }
}
```

## 🛠️ 实现计划

### 阶段1: 基础架构 (Week 1-2)
- [ ] 实现`EnergyAwareTaskAllocator`基础类
- [ ] 成本矩阵构建算法
- [ ] 改进的匈牙利算法实现
- [ ] 基础能源约束检查

### 阶段2: USV调度 (Week 3-4)  
- [ ] 实现`USVLogisticsScheduler`类
- [ ] 支援效益计算算法
- [ ] USV自身电量管理逻辑
- [ ] 多USV协调机制

### 阶段3: 充电决策 (Week 5-6)
- [ ] 实现`ChargingDecisionMaker`类
- [ ] 多因子评分模型
- [ ] 排队时间预测算法
- [ ] 动态阈值调整机制

### 阶段4: 动态重调度 (Week 7-8)
- [ ] 实现`DynamicReallocator`类
- [ ] 事件触发机制
- [ ] 局部vs全局调度决策
- [ ] 平滑过渡算法

### 阶段5: 集成测试 (Week 9-10)
- [ ] 系统集成和接口联调
- [ ] 性能测试和参数调优
- [ ] 可视化系统集成
- [ ] 文档完善和代码审查

## 📝 配置参数设计

### 核心参数配置
```yaml
# 能源感知任务分配器配置
energy_aware_allocator:
  cost_weights:
    distance: 0.3
    energy: 0.4  
    time: 0.2
    risk: 0.1
  safety_margin: 0.25  # 25%安全余量
  max_tasks_per_uav: 5
  energy_threshold: 0.15  # 15%电量阈值

# USV后勤调度器配置  
usv_logistics:
  support_benefit_weights:
    task_value: 0.4
    energy_cost: 0.3
    time_cost: 0.2
    distance_cost: 0.1
  min_reserve_energy_ratio: 0.3  # 30%预留电量
  max_support_distance: 5000  # 5km最大支援距离

# 充电决策器配置
charging_decision:
  decision_factors:
    distance: 0.3
    queue_time: 0.25
    availability: 0.25  
    urgency: 0.2
  queue_prediction_window: 900  # 15分钟预测窗口
  emergency_threshold: 0.1  # 10%紧急电量阈值

# 动态重调度器配置
dynamic_reallocator:
  trigger_thresholds:
    critical_energy: 0.15
    performance_degradation: 0.2
    load_imbalance: 0.3
  reallocation_interval: 900  # 15分钟周期性检查
  smooth_transition_steps: 3  # 3步平滑过渡
```

## 🔍 测试策略

### 单元测试
- 成本函数计算准确性
- 约束检查逻辑正确性  
- 各算法模块独立功能

### 集成测试
- 多层算法协调工作
- 数据流传递正确性
- 异常情况处理

### 性能测试
- 大规模场景下的实时性
- 内存使用和计算复杂度
- 长时间运行稳定性

### 场景测试
- 极端电量情况
- 设备故障恢复
- 高负载压力测试

## 🚀 预期效果

### 性能提升目标
- 任务完成率: 95%+ (相比现有系统提升15%)
- 能源利用效率: 提升25%
- 平均任务完成时间: 减少20%
- 系统响应时间: <5秒(99%的情况下)
- UAV任务负载: 平均3-4个任务/UAV（v2.0提升）

### 技术创新点
1. **多目标自适应优化**：动态调整权重适应不同场景
2. **智能后勤调度**：USV支援效益最大化算法
3. **预测性充电决策**：基于排队预测的智能选择
4. **平滑动态重调度**：最小化系统扰动的重规划
5. **多任务链能耗优化**：支持UAV执行任务序列（v2.0新增）

## 📚 参考文献与相关技术

### 算法基础
- Hungarian Algorithm for Assignment Problems
- Multi-Objective Optimization in UAV Path Planning
- Dynamic Programming for Vehicle Routing
- Heuristic Decision Trees for Real-time Systems

### 相关开源项目
- SUMO (Simulation of Urban MObility)
- ArduPilot Mission Planner
- ROS Navigation Stack
- OMPL (Open Motion Planning Library)

---

**文档维护说明**：
- 本文档应随着实现进度实时更新
- 重要设计变更需要版本记录
- 算法性能数据需要持续补充
- 问题和解决方案需要及时记录

**最后更新**: 2025-06-22

## 📝 版本更新记录

### v2.0 (2025-06-22)
- **重大改进**: Layer 1算法从匈牙利算法改为贪心多任务分配算法
- **新增功能**: 支持一个UAV执行多个任务的任务链
- **优化**: 任务链能耗计算和累积约束检查
- **修复**: 解决了UAV只能分配一个任务的限制

### v1.0 (2025-06-22)
- 初始版本发布
- 4层优化架构设计
- 基于匈牙利算法的任务分配