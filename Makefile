# 多UAV-USV协同巡检系统 Makefile
# 提供统一的开发和部署命令

.PHONY: help setup install test lint format clean demo train eval docker

# 默认目标
help:
	@echo "多UAV-USV协同巡检系统 - 可用命令:"
	@echo ""
	@echo "  setup      - 设置开发环境（创建虚拟环境并安装依赖）"
	@echo "  install    - 安装项目依赖"
	@echo "  test       - 运行所有测试"
	@echo "  test-unit  - 运行单元测试"
	@echo "  test-integration - 运行集成测试"
	@echo "  lint       - 运行代码检查（ruff + mypy）"
	@echo "  format     - 格式化代码（black + ruff）"
	@echo "  clean      - 清理临时文件"
	@echo "  demo       - 运行系统演示"
	@echo "  train      - 训练强化学习模型"
	@echo "  eval       - 评估模型性能"
	@echo "  docker     - 构建Docker镜像"
	@echo ""

# 环境设置
VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

# 检查虚拟环境是否存在
check-venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "虚拟环境不存在，请先运行 'make setup'"; \
		exit 1; \
	fi

# 设置开发环境
setup:
	@echo "设置开发环境..."
	python3.11 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	@echo "开发环境设置完成！"
	@echo "激活虚拟环境: source $(VENV_DIR)/bin/activate"

# 安装依赖
install: check-venv
	@echo "安装项目依赖..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

# 运行所有测试
test: check-venv
	@echo "运行所有测试..."
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

# 运行单元测试
test-unit: check-venv
	@echo "运行单元测试..."
	$(PYTHON) -m pytest tests/ -m unit -v

# 运行集成测试
test-integration: check-venv
	@echo "运行集成测试..."
	$(PYTHON) -m pytest tests/ -m integration --runslow -v

# 代码检查
lint: check-venv
	@echo "运行代码检查..."
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m mypy src/ --strict

# 代码格式化
format: check-venv
	@echo "格式化代码..."
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

# 清理临时文件
clean:
	@echo "清理临时文件..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/ htmlcov/
	@echo "清理完成！"

# 运行系统演示
demo: check-venv
	@echo "运行系统演示..."
	$(PYTHON) -m src.demo

# 训练强化学习模型
train: check-venv
	@echo "开始训练强化学习模型..."
	$(PYTHON) -m src.train --config default

# 评估模型性能
eval: check-venv
	@echo "评估模型性能..."
	$(PYTHON) -m src.evaluate --config default

# 构建Docker镜像
docker:
	@echo "构建Docker镜像..."
	docker build -t hybrid-uav-usv:latest -f docker/Dockerfile .

# 运行Docker容器
docker-run:
	@echo "运行Docker容器..."
	docker run -it --rm -v $(PWD)/data:/app/data hybrid-uav-usv:latest

# 快速检查（格式化+检查+测试）
check: format lint test-unit
	@echo "快速检查完成！"

# 完整验证（包括集成测试）
verify: format lint test
	@echo "完整验证完成！"

# 开发模式（监听文件变化并自动测试）
dev-watch: check-venv
	@echo "启动开发监控模式..."
	$(PYTHON) -m pytest-watch -- tests/ -m unit -v

# 性能分析
profile: check-venv
	@echo "运行性能分析..."
	$(PYTHON) -m src.profile_performance

# 生成文档
docs: check-venv
	@echo "生成文档..."
	$(PYTHON) -m pdoc src/ -o docs/

# 检查依赖安全性
security: check-venv
	@echo "检查依赖安全性..."
	$(PYTHON) -m safety check

# 更新依赖
update-deps: check-venv
	@echo "更新依赖包..."
	$(PIP) list --outdated
	@echo "手动更新 requirements.txt 中的版本号"

# 备份结果
backup-results:
	@echo "备份实验结果..."
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz results/ models/ logs/

# 快速重置环境
reset: clean
	@echo "重置开发环境..."
	rm -rf $(VENV_DIR)
	$(MAKE) setup

# WSL2 特定命令
wsl-setup: setup
	@echo "WSL2环境额外设置..."
	export WSL2_NETWORK=1

# GPU检查
check-gpu: check-venv
	@echo "检查GPU可用性..."
	$(PYTHON) -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"