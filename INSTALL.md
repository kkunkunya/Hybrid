# 安装指导

## 环境要求

- Python 3.11+ (当前检测到 Python 3.12.3 ✅)
- Linux/Ubuntu 系统 (WSL2 支持)

## 快速安装步骤

### 1. 安装系统依赖

```bash
# 安装Python venv支持
sudo apt update
sudo apt install python3.12-venv python3-pip

# 安装OpenCV依赖（可选）
sudo apt install libopencv-dev python3-opencv
```

### 2. 创建虚拟环境

```bash
# 进入项目目录
cd /mnt/c/Users/sxk27/OneDrive\ -\ MSFT/Project/Hybrid

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级pip
pip install --upgrade pip setuptools wheel
```

### 3. 安装项目依赖

```bash
# 安装核心依赖（最小化安装）
pip install numpy>=1.24.0 scipy>=1.10.0 pyyaml>=6.0.0

# 安装深度学习依赖
pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cpu

# 安装计算机视觉依赖
pip install opencv-python>=4.8.0 Pillow>=10.0.0

# 安装其他依赖
pip install matplotlib>=3.7.0 pandas>=2.0.0 tqdm>=4.66.0

# 安装开发工具（可选）
pip install pytest>=7.4.0 black>=23.0.0 mypy>=1.5.0
```

### 4. 验证安装

```bash
# 测试基础模块
python3 -c "
import sys
sys.path.insert(0, '.')
from src.utils.energy import EnergyCalculator, AgentType
calc = EnergyCalculator()
energy, time = calc.calculate_movement_energy(AgentType.UAV, 1000.0, 10.0)
print(f'能源计算测试: {energy:.2f}Wh, {time:.1f}s')
print('✅ 基础功能正常')
"

# 运行配置测试
python3 -c "
import sys
sys.path.insert(0, '.')
from src.config.config_loader import load_default_config
config = load_default_config()
print(f'配置加载测试: UAV数量={config[\"agents\"][\"uav\"][\"count\"]}')
print('✅ 配置系统正常')
"

# 运行演示（需要完整依赖）
python3 src/demo.py
```

## 故障排除

### 问题1: `python3-venv not available`

```bash
sudo apt update
sudo apt install python3.12-venv python3-dev
```

### 问题2: `No module named 'numpy'`

```bash
# 确保虚拟环境已激活
source venv/bin/activate
pip install numpy
```

### 问题3: OpenCV安装失败

```bash
# 使用pip安装headless版本
pip install opencv-python-headless
```

### 问题4: PyTorch CUDA支持

```bash
# CPU版本（推荐）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA版本（如果有GPU）
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 最小测试环境

如果完整安装遇到问题，可以创建最小测试环境：

```bash
# 只安装核心依赖
pip install numpy pyyaml

# 测试核心功能
python3 -c "
import sys
sys.path.insert(0, '.')
from src.utils.energy import EnergyCalculator, AgentType
from src.config.config_loader import load_default_config
print('✅ 最小环境测试通过')
"
```

## 开发环境设置

如果要进行开发，建议完整安装：

```bash
# 克隆项目后
cd Hybrid

# 使用Makefile自动设置（推荐）
make setup

# 或手动设置
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 运行测试
make test
```

## 目录权限问题

在WSL2中可能遇到权限问题：

```bash
# 修复权限
chmod +x src/demo.py
chmod +x Makefile

# 或者直接用python运行
python3 src/demo.py
```

## 下一步

安装完成后，建议：

1. 运行 `python3 src/demo.py` 查看系统演示
2. 阅读 `README.md` 了解项目详情
3. 查看 `src/config/default.yaml` 了解配置选项
4. 运行测试 `python3 -m pytest tests/` 验证功能