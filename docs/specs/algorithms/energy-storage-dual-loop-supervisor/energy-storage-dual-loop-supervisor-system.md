# 储能双向 DC-DC 双环与开关协调控制算法——系统规格

## 状态：历史设计草案

> 本文件保留早期系统设计思路，不代表当前实现完成度；当前模型角色、结果和限制以根 README 与 `docs/PROJECT_STATUS.md` 为准。

**最后更新：** 2026-07-23
**当前参考模型：** `simulink/models/main_model_fd_v05_energyprotect.slx`
**算法名称：** `EnergyStorageDualLoopSupervisor`
**范围：** 电压外环、电流内环、PWM/死区使能、充放电模式管理、直流母线主开关与预充开关协调

---

## 1. 执行摘要

本算法用于 400 V 直流母线与约 211 V 电池之间的双向 DC-DC 变换器。控制器采用离散双环结构：1 ms 电压外环产生电感电流参考，50 μs 电流内环产生占空比修正量，20 kHz PWM 驱动同步半桥；Stateflow 以 1 ms 周期负责待机、预充、充电、放电、方向切换和故障锁止。

本规格将功率调节和高层开关时序分离：

- Simulink 连续信号链负责测量滤波、PI、限幅、斜率限制、前馈、PWM 和死区。
- Stateflow 负责模式仲裁、参考值方向约束、PI 复位、PWM 总使能、主开关和预充开关命令。
- 直流主开关不得承担 PWM 功能，只允许在启动、模式切换、停机或故障过程中动作。

---

## 2. 当前模型基线

### 2.1 已实现链路

当前模型已经包含：

- 电压外环 PI：`Kp=0.8`、`Ki=2.0`、`Ts=1 ms`、输出限幅 `[-5, 5]`。
- 电流内环 PI：`Kp=0.236`、`Ki=74`、`Ts=50 μs`、输出限幅 `[-0.2, 0.2]`。
- 前馈占空比：`D_ff = sat(V_bat / max(V_bus, 1), 0, 1)`。
- PWM：`20 kHz`，计算步长 `1 μs`。
- Buck/Boost 互补门极与 `2 μs` 数字死区。
- Stateflow 模式：`Standby`、`Precharge`、`Charge`、`Discharge`、`Transition`、`Fault_lockout`。
- Stateflow 新增但尚未赋值的输出：
  - `main_bus_switch_cmd`
  - `precharge_bus_switch_cmd`

### 2.2 已确认问题

1. 主开关仍由独立 Step 链路控制，没有与 Stateflow 实际状态和 `trip` 统一仲裁。
2. 两个新开关输出已定义但未在 Stateflow 状态动作中使用，编译产生未使用对象警告。
3. 当前 `Transition` 没有区分充电转放电和放电转充电，无法规范主开关动作先后顺序。
4. 当前 `Precharge` 仅等待固定 5 ms，没有母线电压差、开关反馈或超时判断。
5. 充电和放电共同使用 `Iref_bus_raw`。当 400 V 理想电源把母线钳位在参考值附近时，外环误差接近零，无法形成明确的充电电流命令。
6. 当前 Specialized Power Systems `Breaker` 依赖电流过零完成开断，不适合直流隔离；目标实现应使用直流适用的 Ideal Switch/接触器等效模型。

---

## 3. 目标与成功指标

| ID | 目标 | 成功指标 |
|---|---|---|
| G1 | 双环稳定调节 | 充电电流稳态误差不大于命令的 ±5%；放电母线稳态误差暂定不大于 400 V 的 ±2% |
| G2 | 无冲击模式切换 | 主开关动作前 `|I_L_avg| < 0.5 A` 连续保持至少 5 ms；切换期间电流参考按不大于 200 A/s 归零 |
| G3 | 开关安全互锁 | `main_bus_switch_cmd` 与 `precharge_bus_switch_cmd` 不得同时为 1；主开关命令变化时 `gate_enable=0` |
| G4 | 快速故障响应 | `trip` 有效后一个 Stateflow 周期内，即不超过 1 ms，使 `gate_enable=0`、`Iref_target=0` 并请求断开主开关 |
| G5 | 门极安全 | Buck 与 Boost 有效导通命令不得重叠；保持至少 2 μs 死区 |
| G6 | 输出有界 | `Iref_target ∈ [-5,5] A`、电流 PI 修正量 `∈[-0.2,0.2]`、最终占空比 `D∈[0,1]` |
| G7 | 状态可诊断 | 所有状态输出唯一的 `mode_id`；模式变化、故障和开关命令均可记录 |

> 母线电压动态指标属于暂定指标，最终应根据母线电容、200 Ω 负载、电池能力和功率等级重新标定。

---

## 4. 非目标

| 非目标 | 说明 |
|---|---|
| 功率器件损耗与热模型 | 本版本只规范控制与开关时序 |
| SOC 估计算法重构 | 使用 Battery 模块现有 SOC 输出 |
| 真实接触器电弧建模 | 控制接口保留反馈和超时，但不定义电弧物理 |
| 自适应 PI 或在线辨识 | 第一版采用固定可标定参数 |
| 直接由 Stateflow 产生 PWM | PWM 和死区必须保留在 Simulink 快速链路 |

---

## 5. 外部接口契约

### 5.1 输入

| 名称 | 含义 | 单位 | 类型 | 周期 | 符号/范围 | 当前状态 |
|---|---|---:|---|---:|---|---|
| `mode_cmd` | 上层模式请求 | — | 建议枚举/`uint8` | 1 ms | 0=Standby，1=Charge，2=Discharge | 当前为 `double` |
| `I_L_avg` | 滤波电感电流 | A | `double` | 1 ms | 正值：母线→电池；负值：电池→母线 | 已有 |
| `Iref_bus_raw` | 母线电压外环原始电流参考 | A | `double` | 1 ms | 正值吸收母线能量，负值支撑母线 | 已有 |
| `Iref_charge_raw` | 充电 CC/CV 电流参考 | A | `double` | 1 ms | `[0, I_charge_max]` | 建议新增 |
| `charge_allowed` | 充电许可 | — | `boolean` | 1 ms | 1=允许 | 已有 |
| `discharge_allowed` | 放电许可 | — | `boolean` | 1 ms | 1=允许 | 已有 |
| `trip` | 故障跳闸请求 | — | `boolean` | 1 ms | 1=立即进入故障锁止 | 已有 |
| `fault_reset` | 故障复位请求 | — | `boolean` | 1 ms | 1=请求复位 | 已有 |
| `V_bus` | 母线电压 | V | `double` | 1 ms | 非负 | 建议接入 Stateflow |
| `V_source` | 外部直流源电压 | V | `double` | 1 ms | 非负 | 建议新增 |
| `main_switch_fb` | 主开关实际闭合反馈 | — | `boolean` | 1 ms | 1=闭合 | 模型可选、硬件必需 |
| `precharge_switch_fb` | 预充开关实际反馈 | — | `boolean` | 1 ms | 1=闭合 | 模型可选、硬件必需 |

### 5.2 输出

| 名称 | 含义 | 类型 | 周期 | 约束 | 去向 |
|---|---|---|---:|---|---|
| `Iref_target` | 模式仲裁后的电流参考 | `double` | 1 ms | `[-5,5] A` | 限幅、斜率限制、电流环 |
| `gate_enable` | PWM 总使能 | `boolean` | 1 ms | 切换主开关时必须为 0 | Buck/Boost 门极互锁 |
| `main_bus_switch_cmd` | 400 V 侧主开关命令 | `boolean` | 1 ms | 1=闭合；与预充开关互斥 | DC Ideal Switch/接触器 |
| `precharge_bus_switch_cmd` | 预充支路命令 | `boolean` | 1 ms | 1=闭合；与主开关互斥 | 预充开关 |
| `reset_pi` | PI 复位脉冲 | `boolean` | 1 ms | 状态进入时产生一个周期脉冲 | 电压 PI、电流 PI |
| `mode_id` | 当前模式标识 | 建议枚举/`uint8` | 1 ms | 见模式表 | 记录与诊断 |
| `transition_flag` | 正处于过渡过程 | `boolean` | 1 ms | 过渡/预充时为 1 | 记录与保护 |
| `fault_latched` | 故障锁存状态 | `boolean` | 1 ms | Fault 状态为 1 | 上层诊断，建议新增 |

### 5.3 关键可标定参数

| 参数 | 默认值 | 单位 | 建议范围 | 说明 |
|---|---:|---:|---:|---|
| `V_bus_ref` | 400 | V | 按系统定义 | 母线电压目标 |
| `I_charge_max` | 5 | A | `[0, 10]` | 当前 Stateflow 常量 |
| `I_discharge_max` | 5 | A | `[0, 10]` | 当前 `I_discharge_cmd` |
| `I_zero` | 0.5 | A | `[0.1, 1]` | 开关动作前零电流判据 |
| `T_zero_hold` | 5 | ms | `[2, 20]` | 零电流持续时间 |
| `V_precharge_tol` | TBD | V | 建议为源电压的 2%～5% | 预充完成电压差 |
| `T_precharge_stable` | 5 | ms | `[5, 100]` | 电压差满足后的稳定时间 |
| `T_precharge_timeout` | TBD | s | 由 RC 常数确定 | 超时进入故障 |
| `T_switch_settle` | 2 | ms | `[1, 20]` | 主开关反馈稳定等待 |
| `Iref_slew_up/down` | ±200 | A/s | 按电感和功率限制 | 当前 Rate Limiter |
| `DeadTime` | 2 | μs | 由器件确定 | 当前一拍延时，采样 2 μs |

---

## 6. 工作模式

| 模式 | `mode_id` | 主开关 | 预充开关 | PWM | 电流参考 |
|---|---:|---:|---:|---:|---|
| `Standby` | 0 | 0 | 0 | 禁止 | 0 |
| `Precharge` | 1 | 0 | 1 | 禁止 | 0 |
| `Charge` | 2 | 1 | 0 | 允许 | `clamp(Iref_charge_raw,0,I_charge_max)` |
| `Discharge` | 3 | 0 | 0 | 允许 | `clamp(Iref_bus_raw,-I_discharge_max,0)` |
| `Transition_C2D` | 4 | 先保持 1，零流后断开 | 0 | 先允许调零，开关动作前禁止 | 斜坡归零 |
| `Transition_D2C` | 4 | 0 | 0，随后进入预充 | 先允许调零，随后禁止 | 斜坡归零 |
| `Fault_lockout` | 5 | 0 | 0 | 禁止 | 0 |

如果系统目标是把放电能量回馈到具备吸收能力的双向直流源，而不是给 200 Ω 负载供电，则 `Discharge` 状态下主开关应保持闭合。该拓扑选择必须在接口冻结前明确，不能由控制器运行时猜测。

---

## 7. 典型运行场景

| ID | 场景 | 初始条件 | 期望行为 | 验收条件 |
|---|---|---|---|---|
| S1 | 上电待机 | `mode_cmd=0` | 所有开关断开、PWM 禁止、参考为零 | 一个监督周期内进入 Standby |
| S2 | 待机转充电 | `mode_cmd=1`、充电许可有效 | 预充→主开关闭合→充电 | 主开关闭合前 PWM=0；预充无超时 |
| S3 | 稳态充电 | Charge | 电流内环跟踪正参考 | 稳态误差 ≤±5%，不超过电流上限 |
| S4 | 充电转放电 | `1→2` | 电流归零、PWM 关闭、主开关断开、再进入放电 | `|I_L_avg|<0.5 A` 持续 5 ms 后才断开 |
| S5 | 稳态放电 | Discharge | 电池维持母线电压 | 母线误差暂定 ≤±2% |
| S6 | 放电转充电 | `2→1` | 电流归零、PWM 关闭、预充、闭合主开关、进入充电 | 不跳过预充；无门极与主开关同时切换 |
| S7 | 任意状态故障 | `trip=1` | 参考清零、门极封锁、开关断开请求、故障锁存 | ≤1 ms 输出安全命令 |
| S8 | 故障复位 | `fault_reset=1 && trip=0` | 只返回 Standby，不自动恢复充/放电 | 必须再次收到有效模式请求 |
| S9 | 预充失败 | 电压差长期超限 | 进入 Fault_lockout | 在 `T_precharge_timeout` 内检测 |

---

## 8. 执行与数值约束

| 层级 | 周期 | 说明 |
|---|---:|---|
| 电气网络与 PWM | 1 μs | powergui 离散步长、PWM 计算步长 |
| 电流测量移动平均 | 1 μs | 基频设为 20 kHz |
| 电流内环 | 50 μs | 与 20 kHz PWM 周期一致 |
| 电压外环 | 1 ms | 比电流环慢 20 倍 |
| Stateflow 监督层 | 1 ms | 与外环同周期 |

要求：

1. 快慢速率之间必须使用 Rate Transition。
2. Stateflow 只接收平均电流，不接收原始 PWM 纹波电流。
3. PI 采用离散 Forward Euler，并保留反算式防积分饱和。
4. 任意反馈闭环至少包含离散状态/延时，不得形成纯直接馈通代数环。

---

## 9. 开放问题

| # | 问题 | 当前结论 |
|---|---|---|
| 1 | 放电能量进入 200 Ω 负载还是回馈 400 V 源？ | 当前按“断开源、供负载”规范；实现前确认 |
| 2 | 是否已有实际预充电阻和独立预充开关？ | 当前模型未形成完整受控预充支路，需确认 |
| 3 | 充电采用母线电压环还是电池 CC/CV？ | 建议充电使用 CC/CV，放电使用母线电压环 |
| 4 | 接触器是否有辅助触点反馈？ | 仿真可选，硬件实现建议必需 |
| 5 | `V_precharge_tol` 和超时值是多少？ | 需根据母线电容和预充电阻标定 |

---

## 附录 A：关联文档

- [控制架构规格](energy-storage-dual-loop-supervisor-architecture.md)
- [Stateflow 详细规格](energy-storage-dual-loop-supervisor-stateflow-detailed-spec.md)
- [实施计划](energy-storage-dual-loop-supervisor-implementation-plan.md)
- [测试计划](energy-storage-dual-loop-supervisor-test-plan.md)

## 附录 B：研究与模型核验记录

- `main_model` 当前可编译并可完成 1 s 仿真。
- Stateflow 编译周期为 1 ms；两个新开关输出当前未使用。
- MathWorks 对 Specialized Power Systems `Breaker` 的说明指出：外部信号 0 请求打开、正值闭合，实际打开需等待电流过零，并明确建议直流电路使用 Ideal Switch：
  [MathWorks Breaker documentation](https://www.mathworks.com/help/sps/powersys/ref/breaker.html)
