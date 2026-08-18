# MNIST 脉冲 SNN 论文演示 —— 双击即用

**Local Online Backpropagation with probabilistic spiking synapses** 论文配套演示程序。

拿到本文件夹后：**双击 `mnist_train_demo.bat`**，训练自动开始，实时显示进度，结束后自动生成训练曲线图、权重 checkpoint 和结果摘要。**全程离线，不需要下载任何东西**（MNIST 数据已随包内置在 `mnist_data/` 文件夹里）。

---

## 环境要求（仅此一项）

- **Windows** + **Python 3.9 及以上**（安装时勾选 *Add python.exe to PATH*；下载：https://www.python.org/downloads/）
- numpy / matplotlib 未安装时，启动脚本会自动帮你装（需网络仅此一次；装不上会给出手动命令）

> 若目标机器没有 Python 也没关系：把根目录的 `dist/mnist_train_demo`（53MB 单文件）复制过去直接运行，无需安装任何环境（Linux/macOS 可执行；Windows 版需在 Windows 上用同一条 PyInstaller 命令现打，见项目 README）。

## 怎么跑

| 方式 | 命令 |
|---|---|
| 默认旗舰配置（推荐） | 双击 `mnist_train_demo.bat`（等价 `python mnist_demo_train.py`） |
| 快速演示 | `python mnist_demo_train.py --samples 1000` |
| 论文四协议对照 | `python mnist_demo_train.py --config uncal`（未标定）/ `--config reset`（硬清零）/ `--config if`（IF 无泄漏） |
| 其他 | `--seed N` 随机种子、`--out 目录` 输出位置、`--eval-every N` 评估间隔、`--no-gui` 无窗口模式 |
| 帮助 | `python mnist_demo_train.py --help` |

## 训练什么（与论文正文完全一致的协议）

- **网络**：MNIST 784 → 共享卷积（5×5×4，stride 2，104 个共享参数）→ FC32 → 10（总参数 18,898）；**随机初始化，无任何预训练或权重注入**。
- **默认配置（论文主结果）**：LIF 膜泄漏（τ_m = 0.5，论文 §4.5）+ 样本间静默间隔（ISI = 50 步）+ 输出层重标定（TARGET=5000, κ=1.0 全局抑制池）——即论文多 seed 冻结验收 **0.877 ± 0.007** 的旗舰配置。
- **学习规则**：纯**在线、局部**的反向传播（论文 SDE 框架）：每个神经元只用本地误差与资格迹更新，无全局梯度、无误差传播网络。

## 跑多久 / 能看到什么

- 默认 `--samples 3000` 约 **40–60 分钟**；`--samples 1000` 约 15–20 分钟。
- 训练中每 200 样本打印一行实时进度（loss / 训练准确率 / 冻结测试准确率 + 进度条）；屏幕可达时开实时曲线窗口，否则自动存图。
- 结束后在 `--out` 目录（默认脚本所在目录）生成：

| 文件 | 内容 |
|---|---|
| `demo_training_curves.png` | 三面板训练曲线（loss / train acc / frozen test acc），论文可直接引用 |
| `demo_checkpoint.npz` | 训练好的权重（与项目 `exp4/*.npz` 同格式） |
| `demo_summary.txt` | 配置、最终多 seed 冻结准确率、产物路径 |

## 结果怎么看

- 训练曲线中 **frozen test accuracy** 面板是论文验收口径（冻结权重、τ_m=0.5 / ISI=100 读出协议的多 seed 平均）的精简实时版；终值同时在终端和 `demo_summary.txt` 里。
- 参考标尺（论文 14k–20k 长跑的最终验收）：旗舰重标定 **0.877±0.007**，未标定 0.501，硬清零 0.824——演示的短训练（1k–3k 样本）数值会低一些（1k ≈ 0.6，3k ≈ 0.7+），曲线形状与趋势与长跑一致。

## 常见问题

- **双击后一闪而过？** 说明 Python 未找到或依赖未装上——用命令行手动跑 `python mnist_demo_train.py` 看报错；或按 bat 里提示安装 Python 并勾选 Add to PATH。
- **提示缺少 mnist_data？** 必须把**整个文件夹**一起拷贝（脚本与 `mnist_data/` 同级）；若确实丢了数据且网络可用，加 `--download` 可自动补回。
- **想更快看效果？** `--samples 500 --eval-every 250` 约 8 分钟即可看到完整曲线。

## 关于本文件夹

- `mnist_demo_train.py` —— 演示主脚本（含数据自检与实时可视化，~12KB）
- `mnist_shared.py` —— 训练内核（共享卷积 + LIF 在线局部学习，与论文实验同源）
- `mnist_loader.py` —— MNIST 加载器（数据目录指向本文件夹的 `mnist_data/`）
- `mnist_train_demo.bat` —— Windows 双击启动器（自动检查/安装依赖）
- `mnist_data/` —— MNIST 数据集（12MB，已内置，全程离线）