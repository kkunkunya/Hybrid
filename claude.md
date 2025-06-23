# CLAUDE.md – 项目开发速览  
面向**能源约束 + 动态任务**的 **多 UAV‑USV 协同巡检系统**  
> Reinforcement‑Learning Scheduler + Heuristic Calculators + 2‑Opt Corrector


---

## 1 · Core files and utility functions
| 文件 / 模块 | 一句话说明 |
|-------------|-----------|
| `src/env/satellite_scene.py` | 读取带标注的遥感图（⽣成可交互任务地图） |
| `src/scheduler/rl_agent.py` | **RL 调度器** – 观察全局状态并输出任务包分配 |
| `src/planner/hca_star.py` | **HCA‑A\*** – 代价感知路径规划（耗时 & 耗能双目标） |
| `src/planner/opt_2opt.py` | **2‑Opt** 路径序列微调，消除交叉，缩短总程 |
| `src/utils/energy.py` | 统一的 UAV/USV 电量‑动力学计算工具箱 |
> ⚠️ *如果文件名与实际仓库不符，请按实际情况及时调整。*

---

## 2 · Code style guidelines
- **Black + Ruff**：自动格式化；行宽 88。  
- **类型注解全覆盖**：`mypy --strict` 必须通过。  
- **面向接口**：调度器、规划器均以 **抽象基类** 暴露 `plan()` / `evaluate()` 接口，便于替换算法。

---

## 3 · Testing instructions
| 测试类型 | 命令 | 说明 |
|----------|------|------|
| 单元测试 | `pytest -m unit -q` | 独立函数 / 模块级 |
| 集成测试 | `pytest -m integration --runslow` | 完整仿真一轮，验证调度 + 规划闭环 |
| 覆盖率 | `pytest --cov=src --cov-report=term-missing` | 目标 ≥ 85 % |
> 测试夹具集中在 `tests/conftest.py`，可模拟电量衰减、突发任务到达等场景。

---

## 4 · Repository etiquette
- **分支命名**：`feature/<issue-id>-<slug>`、`bugfix/<slug>`、`experiment/<algo-name>`  
- **Commit Message**：遵循 Conventional Commits  
  - `feat: add prioritized replay buffer`  
  - `fix(hca-star): handle zero‑wind edge case`  
- **PR checklist**（简要）：  
  1. 自测通过 `make lint && make test`  
  2. 说明 *动机 / 主要变更 / 验证方法*  
  3. 若修改公共接口，请同时更新文档和示例脚本  

---

## 5 · Developer environment setup
| 依赖 | 版本 / 说明 |
|------|-------------|
| Python | 3.11 (见 `pyproject.toml`) |
| 包管理 | `pip` + `requirements.txt` （建议使用 **系统级依赖** 打包在 `docker/Dockerfile`） |
| 强化学习框架 | `torch>=2.2`, `gymnasium`, `ray[rllib]` |
| 地图处理 | `numpy`, `opencv-python`, `geopandas` |
| 可视化 | `matplotlib`, `plotly` (*可选*) |

> **虚拟环境**：仓库根目录已提供 `venv/` 占位符，`make setup` 会自动创建/激活。  
> **Docker**：`docker compose up sim` 可一键在容器内复现实验。

---

## 6 · Project directory outline
```

.
├─ data/                  # 标注好的卫星图及任务 JSON
│  ├─ scenes/
│  └─ labels/
├─ src/
│  ├─ env/                # 场景 & 任务生成
│  ├─ scheduler/          # RL 智能体
│  ├─ planner/            # HCA-A\*、2‑Opt 等算法
│  ├─ utils/              # 通用工具
│  └─ config/             # \*.yaml 超参数与常量
├─ tests/
├─ scripts/               # 一次性实验脚本（结束后应删除）
├─ docker/
└─ docs/

```
> ✅ **原则 1**：目录结构需先定版，模块新增必须对应到已有路径，避免杂散脚本  
> ✅ **原则 2**：实验 / debug 脚本放 `scripts/`，验证后务必清理，保持仓库整洁

---

## 7 · Any unexpected behaviors or warnings
- **首次载入大图像**：`opencv.imread()` 对超大分辨率图片可能返回 `None`；请确认本机已安装带大文件支持的 OpenCV 4.  
- **GPU 不足**：若显存 < 8 GB，`ray[rllib]` 可能自动回退到 CPU，训练速度显著下降。请在 `configs/train.yaml` 中设置 `num_gpus: 0` 明确声明。  
- **Windows + WSL2**：Docker 网络中 UDP 广播包转发受限，集成测试需加环境变量 `WSL2_NETWORK=1`。  

---

### 维护者须知
- **修改公共接口** → 必须同时更新 `docs/` 与测试用例  
- **添加外部依赖** → 在 `requirements.txt`、`docker/Dockerfile`、`docs/dependencies.md` 三处同步  
- **版本升级** → 提前在 `dev` 分支跑一轮全量实验以保障 reproducibility  

---


# 原则



1.在项目开始把项目目录大纲指定好,后面程序都要按照这个目录进行编写



2.不要乱新建程序,如果是测试程序测试没问题以后要删除保证项目可读性


3.用中文写好注释,保证后续个人维护项目可行性


4.要进行代码大范围修改时候一定要谨慎,遇到这种情况请立刻详细阅读一遍项目代码文件,不要吝啬上下文


4.编写的程序都要带日志功能,能将运行的命令行结果保存到logs文件夹中,同时要求代码有足够的调试输出,保证我和你都能得到对代码运行足够详细的的了解


# 测试要求



- 能我手动运行命令进行测试程序的尽量指导我手动运行,我会把结果反馈给你



# 使用要求

多用planning mode进行编程,你的编程能力比我强很多,我的回复有时候会问题,请你及时指正


