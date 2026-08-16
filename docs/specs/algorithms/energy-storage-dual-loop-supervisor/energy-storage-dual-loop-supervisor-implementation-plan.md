# 储能双向 DC-DC 双环与开关协调控制算法——实施计划

## 状态：历史实施规划

> 本文件保留早期规划，不作为当前进度清单；当前模型角色、结果和限制以根 README 与 `docs/PROJECT_STATUS.md` 为准。

**最后更新：** 2026-07-23
**架构规格：** [控制架构规格](energy-storage-dual-loop-supervisor-architecture.md)
**测试计划：** [测试计划](energy-storage-dual-loop-supervisor-test-plan.md)

---

## 1. 实施原则

本计划只描述后续模型修改顺序。当前规格输出阶段不修改 `main_model.slx`。

任何实施必须先完成接口冻结，尤其要确认：

1. 放电时 400 V 源是否需要断开；
2. 是否实现真实预充支路；
3. 充电采用 CC/CV 还是外部电流命令；
4. 是否提供主开关和预充开关反馈。

---

## 2. 目标模型层次

```text
main_model
├─ Measurement_Conditioning
│  ├─ Current_Moving_Average
│  └─ Rate_Transitions
├─ Charge_Reference_Controller
│  └─ CC_CV_Reference
├─ Bus_Voltage_Controller
├─ Mode_Manager                    # Stateflow
│  ├─ Standby
│  ├─ Charge_Connection
│  ├─ Charge
│  ├─ Transition_C2D
│  ├─ Discharge
│  ├─ Transition_D2C
│  └─ Fault_Lockout
├─ Current_Controller
├─ Feedforward_Duty
├─ PWM_And_Deadtime
└─ DC_Bus_Switching
   ├─ Main_DC_Switch
   └─ Precharge_Path
```

---

## 3. 依赖

| 工具箱 | 用途 | 必需 |
|---|---|---|
| Simulink | 双环控制与信号链 | 是 |
| Stateflow | 模式与开关协调 | 是 |
| Simscape Electrical / Specialized Power Systems | 当前功率电路 | 是 |
| Simulink Test 或 SATK `model_test` | 持久化验证 | 建议 |

---

## 4. 实施阶段

### Phase 0：接口冻结

**目标：** 在任何连线修改前确定最终接口。

任务：

1. 确认放电能量路径。
2. 冻结 `mode_cmd`、`mode_id` 枚举。
3. 冻结 Stateflow 新增输入输出。
4. 确定 `V_precharge_tol`、预充时间常数和超时值。
5. 确定开关反馈在仿真中的等效方式。
6. 确认充电参考源为 CC/CV 或上层电流命令。

检查点：

- 接口表无 TBD 的结构性决策；
- 信号名称、单位、类型和周期已经批准；
- 不再需要外部独立 Step 直接控制主开关。

### Phase 1：Stateflow 安全输出与状态重构

**目标：** 先实现不会产生危险命令的监督逻辑。

任务：

1. 给现有所有状态补齐：
   - `main_bus_switch_cmd`
   - `precharge_bus_switch_cmd`
2. 将 `Transition` 重构为方向相关子状态。
3. 增加预充电压差、稳定时间和超时逻辑。
4. 增加开关反馈不一致诊断。
5. 统一 `trip` 最高优先级。
6. 故障复位只返回 Standby。
7. 输出改为 Moore 风格并保证每个状态完整赋值。

验证：

- `model_read` 检查状态和转移；
- Stateflow lint 无错误；
- 编译时不再报告两个开关输出未使用；
- 逻辑真值检查确保主开关和预充开关互斥。

### Phase 2：直流主开关和预充支路

**目标：** 用适合直流的开关模型替代当前 Breaker 控制链。

任务：

1. 明确现有未接线 Ideal Switch 的用途和参数。
2. 将 Main DC Switch 串接到正确母线位置。
3. 建立独立预充电阻和预充开关支路。
4. 连接 Stateflow 两个开关命令。
5. 增加开关状态记录和必要的电压/电流测量。
6. 删除或隔离未使用的 Breaker、Step、Constant4、From9 等遗留测试块。

验证：

- `model_read` 确认物理拓扑；
- `model_check` 无未连接端口错误；
- 主开关断开后 400 V 源确实与母线隔离；
- 预充期间母线电压按 RC 曲线上升，无直接浪涌闭合。

### Phase 3：控制参考链整改

**目标：** 形成模式正确的电流参考。

任务：

1. Charge 接入 `Iref_charge_raw`：
   - 第一阶段可用固定可标定 CC 参考；
   - 后续实现 CC/CV。
2. Discharge 保留母线电压 PI 的负参考。
3. 保留 Stateflow 方向限幅：
   - Charge `[0,+Imax]`
   - Discharge `[-Imax,0]`
4. 合并重复的 Iref Rate Limiter。
5. 确认 `Iref_target` 从 1 ms 到 50 μs 的 Rate Transition。
6. 设计 gate 禁用期间的 PI 冻结/跟踪机制。

验证：

- 400 V 源接入时仍能得到非零、受限的充电参考；
- 放电外环极性正确；
- 模式切换时参考连续且斜率不超过 ±200 A/s。

### Phase 4：PWM 与互锁核验

**目标：** 保持快速链路独立、安全。

任务：

1. 保留 20 kHz PWM 和 1 μs 计算步长。
2. 核验 Buck/Boost 互补关系。
3. 核验 2 μs 死区。
4. 保证 `gate_enable=0` 在一个快速周期内使两个门极归零。
5. 添加门极重叠诊断或断言。

验证：

- 全仿真范围内不存在门极重叠；
- 占空比始终位于 `[0,1]`；
- gate 禁用响应不超过 1 μs～一个 PWM 计算步。

### Phase 5：系统集成与调参

**目标：** 完整充放电闭环满足系统指标。

任务：

1. 运行待机→充电→放电→待机全序列。
2. 调整 CC/CV、母线外环和电流环增益。
3. 校验 200 Ω 负载和母线电容下的动态。
4. 扫描电池电压、SOC、负载和母线电容变化。
5. 验证预充和所有故障路径。

检查点：

- 所有测试计划用例通过；
- 无电流方向错误；
- 无不受控开关动作；
- 无积分饱和导致的恢复迟滞；
- 模型结构检查无 error 级问题。

---

## 5. 参数表

| 参数 | 当前值 | 单位 | 可标定 | 目标位置 |
|---|---:|---:|---|---|
| `V_bus_ref` | 400 | V | 是 | Bus Voltage Controller |
| `Kpv` | 0.8 | — | 是 | Voltage PI |
| `Kiv` | 2.0 | 1/s | 是 | Voltage PI |
| `Ts_voltage` | 1e-3 | s | 否 | Voltage PI |
| `Kpi` | 0.236 | — | 是 | Current PI |
| `Kii` | 74 | 1/s | 是 | Current PI |
| `Ts_current` | 50e-6 | s | 否 | Current PI |
| `DeltaD_min/max` | ±0.2 | — | 是 | Current PI |
| `I_charge_max` | 5 | A | 是 | Mode Manager |
| `I_discharge_max` | 5 | A | 是 | Mode Manager |
| `I_zero` | 0.5 | A | 是 | Mode Manager |
| `T_zero_hold` | 5e-3 | s | 是 | Mode Manager |
| `Iref_slew` | ±200 | A/s | 是 | Iref Limiter |
| `Fsw` | 20e3 | Hz | 否/配置 | PWM |
| `Ts_pwm` | 1e-6 | s | 否 | PWM |
| `DeadTime` | 2e-6 | s | 是 | Gate Interlock |
| `V_precharge_tol` | TBD | V | 是 | Mode Manager |
| `T_precharge_timeout` | TBD | s | 是 | Mode Manager |
| `T_switch_settle` | 2e-3 | s | 是 | Mode Manager |

---

## 6. 同步检查点

每个阶段完成后：

1. `model_read`：确认拓扑和状态逻辑。
2. `model_query_params`：核对参数表。
3. `model_check`：检查未连接端口、悬空线和 Stateflow lint。
4. 编译更新：确认无编译错误和未使用输出警告。
5. 运行对应阶段测试。
6. 规格与模型不一致时，先更新规格并评审，再继续实施。

---

## 7. 完成定义

- [ ] 接口冻结并消除结构性 TBD。
- [ ] Stateflow 所有输出在所有状态中明确赋值。
- [ ] 两个开关命令由 Stateflow 驱动，不再由独立 Step 驱动。
- [ ] 直流开关模型可在 DC 电流条件下可靠隔离。
- [ ] 预充支路和超时保护有效。
- [ ] 充电参考不依赖被理想电源钳位的母线误差。
- [ ] 主开关动作前满足零流、关门极和稳定等待。
- [ ] 故障响应不超过 1 ms。
- [ ] PWM 无上下管重叠。
- [ ] 组件、集成、系统和鲁棒性测试全部通过。

---

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| Stateflow 状态过多 | 使用层次化复合状态，外部 `mode_id` 仍保持六类 |
| 接触器动作与理想模型不一致 | 增加反馈、动作延时和超时接口 |
| 预充参数不合理 | 根据母线总电容和预充电阻计算 RC，并做参数扫描 |
| 电压外环与理想电源冲突 | 充电采用 CC/CV，放电采用母线电压环 |
| PI 在禁用期间积分 | 使用 reset 脉冲＋Enabled/Tracking 冻结 |
| 重复 Rate Limiter 改变响应 | 合并为一个明确限速点并回归测试 |
| 开关更换引入物理连接错误 | 修改前后分别 `model_read`，再做隔离和导通测试 |
