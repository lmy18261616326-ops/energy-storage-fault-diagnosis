# 储能双向 DC-DC 双环与开关协调控制算法——测试计划

## 状态：历史测试草案

> 本文件保留早期验证设计，不代表当前测试完成度；当前验证状态以根 README、`docs/PROJECT_STATUS.md` 与自动化测试结果为准。

**最后更新：** 2026-07-23
**架构规格：** [控制架构规格](energy-storage-dual-loop-supervisor-architecture.md)

---

## 1. 验证阶段

1. **组件 MIL**：Stateflow、外环、内环、参考限幅、PWM 互锁分别测试。
2. **集成 MIL**：控制组件互连，但使用受控信号源代替完整功率电路。
3. **系统 MIL**：连接当前双向 DC-DC、电池、400 V 源、母线电容和 200 Ω 负载。
4. **鲁棒性测试**：参数变化、测量噪声、命令抖动和故障注入。
5. **SIL/PIL**：仅在后续需要生成控制代码时执行。

---

## 2. 仿真基线

| 设置 | 值 |
|---|---:|
| powergui | Discrete |
| 电气步长 | 1 μs |
| 模型求解器 | Fixed-step `ode3` |
| PWM | 20 kHz |
| 电流 PI | 50 μs |
| 电压 PI | 1 ms |
| Stateflow | 1 ms |
| 母线参考 | 400 V |
| 充电/放电电流上限 | ±5 A |
| 零流阈值 | 0.5 A |
| 零流保持 | 5 ms |

必记录信号：

```text
mode_cmd
mode_id
Iref_bus_raw
Iref_charge_raw
Iref_target
I_L
I_L_avg
V_bus
V_bat
gate_enable
gate_buck_cmd
gate_boost_cmd
main_bus_switch_cmd
precharge_bus_switch_cmd
main_switch_fb
transition_flag
reset_pi
trip
fault_latched
duty_raw
duty_cmd
```

---

## 3. Stateflow 组件测试

| ID | 测试 | 激励 | 期望结果 | 验收条件 |
|---|---|---|---|---|
| SF-01 | 默认状态 | 模型启动 | 进入 Standby | 首个 1 ms 周期内所有执行器命令为 0 |
| SF-02 | 待机转预充 | `mode_cmd=1, charge_allowed=1` | 进入 Precharge | `precharge=1, main=0, gate=0` |
| SF-03 | 预充完成 | 电压差进入阈值并稳定 | CloseMain→Charge | 未满足稳定时间前主开关不得闭合 |
| SF-04 | 预充超时 | 电压差始终超限 | Fault | 在超时值±1 ms 内进入 Fault |
| SF-05 | 待机转放电 | `mode_cmd=2, discharge_allowed=1` | DischargeArm→Discharge | 主开关保持断开 |
| SF-06 | 充电转放电 | Charge 中请求 Discharge | C2D 路径 | 零流保持完成前主开关不变 |
| SF-07 | 放电转充电 | Discharge 中请求 Charge | D2C→Precharge→Charge | 不跳过 Precharge |
| SF-08 | 充电停机 | Charge 中请求 Standby | 电流归零后 Standby | 主开关最终断开 |
| SF-09 | 放电停机 | Discharge 中请求 Standby | 电流归零后 Standby | 门极最终关闭 |
| SF-10 | 任意状态故障 | `trip=1` | Fault_lockout | ≤1 ms 输出安全命令 |
| SF-11 | 故障复位被拒绝 | `trip=1, fault_reset=1` | 保持 Fault | 不得退出 Fault |
| SF-12 | 合法故障复位 | `trip=0, fault_reset=1` | 返回 Standby | 不自动恢复 Charge/Discharge |
| SF-13 | 许可撤销 | 活动模式中撤销对应 allowed | 安全停机路径 | 不发生方向反转 |
| SF-14 | 非法模式值 | `mode_cmd=3/NaN` | Standby/Fault 诊断 | 不产生开关或门极命令 |
| SF-15 | 输出互锁 | 穷举全部状态 | 主开关与预充开关互斥 | 全时域无同时为 1 |

---

## 4. 控制链组件测试

### 4.1 电压外环

| ID | 测试 | 激励 | 验收条件 |
|---|---|---|---|
| VC-01 | 正极性 | `V_bus > V_ref` | `Iref_bus_raw > 0` |
| VC-02 | 负极性 | `V_bus < V_ref` | `Iref_bus_raw < 0` |
| VC-03 | 输出饱和 | 大电压误差 | 输出严格限制在 ±5 A |
| VC-04 | 防积分饱和 | 保持饱和后解除误差 | 输出恢复时无长期滞留；暂定 100 ms 内退出饱和 |
| VC-05 | PI 复位 | `reset_pi` 上升沿 | 积分状态回到配置初值 |

### 4.2 充电参考

| ID | 测试 | 激励 | 验收条件 |
|---|---|---|---|
| CC-01 | 恒流区 | 电池电压低于 CV 阈值 | 输出等于 CC 命令且不超过 5 A |
| CC-02 | 恒压区 | 电池电压接近上限 | 电流参考单调下降，不为负 |
| CC-03 | 充电禁止 | `charge_allowed=0` | 输出被 Stateflow 置零 |

### 4.3 电流内环

| ID | 测试 | 激励 | 验收条件 |
|---|---|---|---|
| IC-01 | 正参考跟踪 | `Iref=+1…+5 A` | 稳态误差 ≤±5% |
| IC-02 | 负参考跟踪 | `Iref=-1…-5 A` | 稳态误差 ≤±5% |
| IC-03 | 修正量限幅 | 大电流误差 | `ΔD∈[-0.2,0.2]` |
| IC-04 | 参考斜率 | 正负阶跃 | 实际参考斜率不超过 ±200 A/s |
| IC-05 | 禁用积分器 | `gate_enable=0` 持续 1 s | 积分器不继续累积 |
| IC-06 | 无扰重新使能 | 禁用后恢复 | Duty 不产生超过允许范围的突跳 |

### 4.4 前馈与占空比

| ID | 测试 | 激励 | 验收条件 |
|---|---|---|---|
| FF-01 | 标称前馈 | `Vbat≈211 V,Vbus=400 V` | `Dff≈0.528` |
| FF-02 | 母线接近零 | `Vbus<1 V` | 分母被保护，无 NaN/Inf |
| FF-03 | 最终限幅 | 任意 `Dff+ΔD` | `Dcmd∈[0,1]` |

### 4.5 PWM 和死区

| ID | 测试 | 激励 | 验收条件 |
|---|---|---|---|
| PWM-01 | 频率 | 固定 Duty | 开关频率 20 kHz |
| PWM-02 | 互补 | Duty 扫描 0～1 | Buck 与 Boost 不同时为 1 |
| PWM-03 | 死区 | 上下沿检查 | 两方向死区均不少于 2 μs |
| PWM-04 | 总禁用 | `gate_enable:1→0` | 一个快速计算步内两个门极归零 |

---

## 5. 集成模式切换测试

| ID | 转移 | 起始条件 | 关键检查 |
|---|---|---|---|
| TR-01 | Standby→Charge | 母线未预充 | 先预充、再主开关、后 PWM |
| TR-02 | Charge→Discharge | `I_L=+5 A` | 参考归零；零流保持；关 PWM；断开源；负参考软启动 |
| TR-03 | Discharge→Charge | `I_L=-5 A` | 参考归零；关 PWM；预充；闭合源；正参考软启动 |
| TR-04 | Charge→Standby | 正电流 | 不带电流直接断开 |
| TR-05 | Discharge→Standby | 负电流 | 归零后关 PWM |
| TR-06 | 过渡中反向命令 | 命令在 C2D 中变回 Charge | 先到安全节点，不产生开关抖动 |
| TR-07 | 过渡中 trip | 任意过渡阶段 | ≤1 ms 进入 Fault |

统一验收：

```text
主开关命令变化前：
    gate_enable == 0
    abs(I_L_avg) < 0.5 A 已保持至少 5 ms

任何时刻：
    NOT(main_switch_cmd AND precharge_switch_cmd)
    NOT(gate_buck_cmd AND gate_boost_cmd)
```

---

## 6. 系统级场景

### SYS-01：完整充放电序列

步骤：

1. 0～0.1 s：Standby。
2. 请求 Charge，完成预充并充电。
3. 达到稳定充电电流。
4. 请求 Discharge，完成安全方向切换。
5. 电池支撑 200 Ω 母线负载。
6. 请求 Standby。

验收：

- 状态顺序符合详细规格；
- 无跳过预充；
- 无主开关带 PWM 动作；
- 电流方向与模式一致；
- 母线和电池电压无非物理尖峰；
- 所有输出保持有界。

### SYS-02：母线负载阶跃

在 Discharge 状态改变母线负载。暂定验收：

- 母线最大偏差不超过参考的 10%；
- 200 ms 内恢复至参考的 ±2%；
- 电流不超过 ±5 A 限制；
- PI 饱和后能恢复。

该指标需根据可提供功率重新确认。400 V、200 Ω 对应约 2 A、800 W 母线负载，应确认电池侧电流限制是否足以覆盖损耗和变换比。

### SYS-03：预充

使用不同母线初始电压：

```text
0 V、200 V、390 V、400 V
```

验收：

- 主开关只在电压差满足阈值并稳定后闭合；
- 预充电流不超过设计限制；
- 电压无法建立时进入 Fault。

### SYS-04：故障注入

在 Charge、Discharge、Precharge 和方向切换中分别注入 `trip`。

验收：

- 一个监督周期内门极禁止；
- 两个开关命令均为 0；
- 故障锁存；
- 未复位前不能重新使能。

---

## 7. 鲁棒性与敏感性

| 参数/条件 | 扫描范围 | 验收 |
|---|---:|---|
| 电池电压 | 额定范围 | 控制稳定、占空比不越界 |
| 母线电容 | 标称 ±20% | 预充和电压环仍满足降级指标 |
| 负载电阻 | 100～400 Ω | 不超电流限制，模式正确 |
| PI 增益 | 标称 ±20% | 不出现发散或持续振荡 |
| 电流测量偏置 | ±0.1 A | 零流判据不误动作 |
| 电流噪声 | 根据传感器定义 | 不导致开关抖动 |
| 模式命令抖动 | 1～5 ms 脉冲 | 经去抖后不重复动作开关 |
| 开关反馈延迟 | 0～额定最大值 | 未超时正常，超时进入 Fault |

---

## 8. 回归检查

每次模型修改后至少执行：

1. 模型编译更新。
2. `model_check`：
   - unconnected ports
   - unconnected lines
   - Stateflow lint
3. Stateflow 输出使用检查。
4. SF-01、SF-06、SF-07、SF-10。
5. IC-01、IC-02、PWM-02、PWM-04。
6. SYS-01 完整序列。

---

## 9. 测试自动化说明

后续如需生成持久化 `.feature` 测试文件，应先使用 `testing-simulink-models` 技能，按 SATK 的专用 Gherkin 语法创建，并通过 `model_test` 执行。本规格只定义测试需求和判据，不生成测试脚本。

---

## 10. 通过准则

- [ ] 所有 Stateflow 组件测试通过。
- [ ] 所有控制链边界、限幅和极性测试通过。
- [ ] 全时域无门极重叠。
- [ ] 全时域无主开关与预充开关同时闭合。
- [ ] 所有主开关动作满足零流和关门极前置条件。
- [ ] 所有模式和方向切换路径均被覆盖。
- [ ] 故障响应不超过 1 ms。
- [ ] 模型无 error 级结构问题。
- [ ] 充电、放电和母线调节性能达到最终批准指标。
