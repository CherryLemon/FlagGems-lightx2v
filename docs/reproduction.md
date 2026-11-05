# 环境、镜像与复现指南

目标是让接手者先确认工具链和算子，再运行 LightX2V。设备支持、算子精度通过与完整模型通过是三个不同阶段。新国产后端必须逐阶段验收，不能直接套用 H100 的结论。

## 1. 镜像必须包含什么

| 层次 | 必需内容 | 检查重点 |
|---|---|---|
| 主机 | 厂商驱动、容器 runtime、所分配设备 | 驱动/runtime 匹配，设备空闲，不能覆盖他人服务 |
| Python/张量库 | 与芯片匹配的 Python、PyTorch、厂商扩展 | 不用 PyPI 通用 torch 覆盖厂商 torch |
| 编译器 | 普通 Triton，或所选档位需要的 FlagTree/TLE/Gluon | 在实际执行脚本的同一解释器中检查版本与 API |
| 算子库 | 固定提交的 FlagGems、FlagGems-lightx2v | 公共 API 可导入，实际选择正确后端 |
| 框架 | 固定版本 LightX2V、lightx2v_platform、配套插件 | 使用实际模型入口，不能只测 stub |
| H3 Python 依赖 | safetensors、transformers、diffusers、accelerate、einops、omegaconf、ftfy、peft、AV/图像库 | 模型动态导入的组件也必须齐备 |
| 媒体 | ffmpeg、ffprobe、PyAV，以及可用的音视频 codec | 实际完整解码，核对帧数、尺寸、帧率、双声道 |
| 多卡（可选） | 厂商 collective 或 FlagCX 与相匹配的 PyTorch binding | 先做限时通信自检，再运行 Ulysses |

参考工具链：H100 使用 Torch2.11.0+cu130、Triton3.6.0 与提供所需 API 的 FlagTree 构建；S5000 历史算子验证使用 Python3.10、Torch/torch_musa2.9.1、MUSA5.2 和 FlagTree0.7.1a1+mthreads3.6；当前环境可能存在系统 Python 与业务 overlay 的差异，必须按下面的检查方法核实实际解释器。这是已验证环境描述，不表示任意同版本镜像或任意国产芯片已经通过完整模型验收。

本库不把厂商 torch/Triton 作为可以随意升级的普通依赖。先安装芯片工具链，再安装业务代码；最终运行 `python -m pip check`。公开代码依赖应使用交付时固定的 commit/release，镜像和脚本保持一致，不使用移动开发分支作为生产依赖。

### S5000 安装 FlagTree

在目标机器/容器内直接安装，避免把模型或大镜像经跳板机中转：

```bash
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3 -m pip install 'flagtree===0.7.1a1+mthreads3.6' $RES
python3 -m pip show torch torch_musa flagtree
```

如果业务在 venv 中运行，以上 `python3` 必须替换为该 venv 的解释器。系统 Python 与业务 venv 可以同时存在不同 FlagTree 版本；只查看系统 `pip show` 不能证明作业使用了正确工具链。TLE 档还应检查所需 `gpu.alloc/copy/wgmma/wgmma_wait` API，不能仅以包可导入判定支持。

NVIDIA 的 SM90 档需要可获取的 FlagTree/Gluon/TLE 构建。普通 upstream Triton 不能作为这些 API 的替代；没有这些能力时，应显式选择普通基线，不能静默换算术格式。

## 2. 容器配置不能只复制镜像名

以同机正在正常工作的 GPU 容器为配置参考，核对 runtime、设备可见性、IPC、shared memory、memlock、PID/network 模式、内存限制、CPU 绑定和必要挂载。复制经过批准的运行配置，再缩小到分配给自己的空闲卡；不要复制他人的进程、凭据或模型输出，也不要关闭安全策略。

S5000 验证容器使用 mthreads runtime，物理卡通过 `MTHREADS_VISIBLE_DEVICES` 选择，容器内再使用逻辑 MUSA0。配置包括无限 memlock、2GiB shared memory，以及足够加载业务的主机内存。仅供小算子测试的16GiB memory limit 不适合直接拿来加载完整 H3；完整模型容器应依据实际权重/加载峰值重新预算。IPC、网络等以同机正常模板为准，不能把 `--ipc=host` 当成所有设备的固定要求。

示例中的路径与卡号由使用者填写，不含任何特定实验节点：

```bash
docker run --name lightx2v-h3 --runtime=mthreads \
  -e MTHREADS_VISIBLE_DEVICES="$PHYSICAL_GPU" -e MUSA_VISIBLE_DEVICES=0 \
  --shm-size=2g --ulimit memlock=-1:-1 \
  --memory=192g --cpuset-cpus="$ALLOCATED_CPUS" \
  --ipc=private --network=bridge \
  -v "$MODEL_DIR:/models/MiniMax-H3:ro" \
  -v "$WORK_DIR:/workspace" \
  "$MUSA_IMAGE"
```

这是该 S5000 环境实际核验过的资源配置轮廓（192GiB 主机内存、2GiB shm、private IPC、bridge network、无限 memlock、分配8个 CPU）。`ALLOCATED_CPUS` 应填本站分配的 CPU 集合；运行前逐项对照本机正常模板，包括必要的厂商库/设备挂载，不要遗漏。NVIDIA 对应使用其 runtime 和 `CUDA_VISIBLE_DEVICES`；其他厂商必须按本机 runtime 文档处理，不能照抄 MUSA 设备变量。

## 3. 模型与依赖获取

在最终目标机器上从可访问的模型源下载，固定模型 revision，并核对所需组件完整。H3 需要的不仅是 DiT 权重，还包括配置、视频/音频 VAE、tokenizer/文本编码器等实际使用的组件。使用固定 embedding 可以跳过文本编码，但不是完整提示词流程的精度验收。

公开模型来源为 `MiniMax/MiniMax-H3`，摩尔环境下载记录固定 revision `57559a67`。例如在目标机安装其 ModelScope 工具后执行：

```bash
modelscope download --model MiniMax/MiniMax-H3 --revision 57559a67 --local_dir "$MODEL_DIR"
```

该短 revision 应由下载工具解析并保存完整版本/文件清单；核对模型许可后使用，不将权重提交到代码仓库。

ModelScope/Hugging Face 下载工具应在目标节点执行；是否使用代理取决于该节点网络。不要把模型权重经资源紧张的跳板机转发。下载中断时使用工具的完整性/续传功能，不把部分目录误当成可加载模型。

H100 全驻卡验证使用 AdaLN cache、固定 embedding 和未加载文本编码器的工况。若需要同时加载文本编码器，必须重新检查显存；不能直接复用那组显存数字。只有请求设置完全相同，性能结果才可比较。

## 4. 先跑算子 smoke 与精度测试

```bash
python -m pip install --no-deps .
python examples/operator_smoke.py --device musa:0 --tier baseline
python examples/operator_smoke.py --device musa:0 --tier tle
# H100, compatible FlagTree/Gluon:
python examples/operator_smoke.py --device cuda:0 --tier sm90
python -m pytest tests/test_contracts.py -q
python -m pytest tests/test_accelerator.py --accelerator-device musa:0 --require-accelerator -q
# H100 acceptance, including the >INT32 packed SwiGLU case (~tens of GiB):
python -m pytest tests/test_accelerator.py --accelerator-device cuda:0 --require-accelerator -q
python benchmark/benchmark_ops.py --device cuda:0 --tier sm90 --rows 1108
```

smoke 只验证设备可用、公开入口及可解析的常量输入，不替代完整精度测试。正式验收需使用仓库的独立参考、随机/边界/非连续输入、输入不变检查和 Graph 刷新测试。Sage 的 BF16-PV 与 FP8-PV 必须使用各自的量化参考与测试；dense SDPA 只能作为另一种精度基线，不能当成相同量化路径。

## 5. 再运行框架示例

使用配套 `lightx2v-plugin-FL` 的 H3 示例和配置。关键项包括：模型路径、`model_cls=minimax_h3`、`model_variant=fl2av`、实际 H×W、帧数、步数、seed、offload 策略，以及显式的 Sage 数值格式/实现档位。不要把普通 dense attention 的 registry key 当作量化 Sage 的别名。

固定 embedding 对照时，每侧加载相同 tensor 和 token tags，核对 shape/dtype/身份；输入准备和初始噪声也必须相同。embedding 传输、模型初始化、编译预热和精度审计应与稳态计时分开。记录每轮，而不是只保留最好一轮。

H3 专用上下文需要严格的补丁与全局注册所有权。按示例在一个独占进程内管理其生命周期，不与其他 runner 的全局 ATen 注册或未知 wrapper 混用。退出/失败时恢复自己安装的对象，不覆盖别人的后续安装。

单卡先使用低分辨率验证最终 latent、视频/音频 VAE、媒体元数据与完整解码。确认没有 dtype 被偷偷改变、fallback 或 offload 策略偏离后，再扩大分辨率。

## 6. 多卡与新国产后端

H3 使用的是 sequence parallel/Ulysses；不要将其记录成已经验证 tensor parallel。固定 rank 到实际分配的设备，核对拓扑、可见卡数、实际 collective 后端、各 rank 输入及同步。GPU 卡数改变时，VAE 是否并行也必须记录。通信自检失败时先诊断，不启动完整模型等待超时。

新增国产后端按以下顺序推进：

1. 安装匹配的驱动/runtime、PyTorch 厂商扩展和 Triton/FlagTree；确认设备检测不是错误的 CUDA 假阳性。
2. 运行普通基线的最小算子与独立精度测试。当前公开 API 明确限定 CUDA/MUSA；其他设备类型必须补充实际后端接入和测试，不能仅改一个设备字符串。
3. 验证 INT8/BF16、所需 dtype/stride 和 Graph 功能，再尝试 TLE。没有 TLE 编译能力时保留可运行基线，不伪装成已优化。
4. 对 LightX2V 及依赖中的 CUDA 专用调用，使用该厂商已有的受支持适配；将必要接线放在框架 vendor 层，不修改已安装依赖或编写全局无条件 alias。
5. 完成单卡固定输入模型验证、完整媒体解码，再验证多卡通信和 Ulysses。
6. 记录代码/镜像版本、实际命中实现、数值与模型质量、性能及未通过项目。编译通过、算子通过、完整模型通过分别报告。

目前没有宣称完整 H3 已在所有国产芯片跑通。S5000 的算子结果是接入参考，不是其他芯片的硬件验证或完整模型质量结论。

## 7. 构建与完整命令入口

仓库提供 `docker/Dockerfile` 和 `docker/requirements-h3.txt` 作为 H100/Python3.12 的依赖层参考。基础镜像先提供厂商 torch、对应 torchvision、编译器、ffmpeg/ffprobe 和 libav；不要用普通 CUDA 镜像替代国产工具链。此 Dockerfile 是可审查的构建配方，本次没有重新构建/验收整张镜像。Python3.10/MUSA 必须使用其已验证的依赖锁：例如参考文件的 NumPy2.3.5 需要较新 Python，不能直接用于 Python3.10；同时检查 NumPy 与厂商 torch 的 ABI。

```bash
docker build --build-arg BASE_IMAGE="$VENDOR_BASE_IMAGE" -f docker/Dockerfile -t flaggems-lightx2v:local .
```

在业务镜像中另外安装固定版本 LightX2V（包含 `lightx2v_platform`）、FlagGems 和插件，安装专用算子包时使用 `--no-deps` 保留既有 torch/编译器。公开入口位于插件 `examples/minimax_h3.py`，配套 `examples/minimax_h3_sm90.json`；完整参数和验收范围见[插件 H3 指南](https://github.com/ModelTC/lightx2v-plugin-FL/blob/main/docs/minimax-h3.md)（新 H3 PR 合入前请从对应 PR 分支读取）。

先把 JSON 中 `adaln_cache_dir` 指向匹配 cache。上游 `tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh` 要设置 `lightx2v_path`、`model_path` 和同一 `--config_json`；T2AV 选择 `fl2av`。无匹配 cache 时先生成，不混用其他步数/模型的 cache。

固定 embedding 文件不随仓库分发。文件须由自己获授权的输入生成，包含 BF16 `[tokens,5120]` `prompt_embeds` 与 INT64 `[tokens]` `text_token_tags`；可另有 `token_ids` 审计字段。脚本核对文件 SHA256 后，在计时前放入设备。

```bash
CUDA_VISIBLE_DEVICES=0 LIGHTX2V_FL_DISABLE_FLAGCX=1 \
python examples/minimax_h3.py \
  --model-path /models/MiniMax-H3 --config examples/minimax_h3_sm90.json \
  --conditioning /inputs/conditioning.pt --conditioning-sha256 "$CONDITIONING_SHA256" \
  --prompt "$PROMPT" --output /outputs/h3.mp4
ffprobe -v error -show_streams -of json /outputs/h3.mp4
ffmpeg -v error -i /outputs/h3.mp4 -f null -
```

这组命令禁用 FlagCX，使用已验证的单卡路径。多卡须另外验收所选 collective。S5000 初始接入把 Sage 改为 BF16 PV+baseline/TLE，把 RMS tier 设为对应实现，并使用 `--no-vae-fusion`；完整模型和视频质量仍需现场验收。
