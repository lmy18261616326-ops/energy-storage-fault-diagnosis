function cfg = signal_config(overrides)
%SIGNAL_CONFIG 集中配置故障数据集采集程序。
%   CFG = SIGNAL_CONFIG() 返回模型名称、信号映射、工况范围、窗口参数、
%   输出路径和模型适配参数。所有脚本只从本文件读取这些配置，避免把
%   模型名、信号名和路径分散写死。
%
%   CFG = SIGNAL_CONFIG(OVERRIDES) 用结构体字段递归覆盖默认配置。
%
%   输入:
%       overrides - 可选结构体；字段层级应与 cfg 一致。
%   输出:
%       cfg       - 完整配置结构体。

arguments
    overrides (1,1) struct = struct()
end

scriptFolder = string(fileparts(mfilename("fullpath")));
simulinkFolder = string(fileparts(fileparts(fileparts(scriptFolder))));

cfg = struct();
cfg.version = "1.12.0";
cfg.modelName = "main_model_fd_v05_energyprotect";
cfg.modelFile = fullfile(simulinkFolder, "models", cfg.modelName + ".slx");
cfg.scriptFolder = scriptFolder;

% 仿真和统一采样设置
cfg.sampleTime = 50e-6;
cfg.loggingSampleTime = cfg.sampleTime;
cfg.windowLength = 0.010;
cfg.windowStep = 0.005;
cfg.transitionDuration = 0.010;
cfg.nearZeroThreshold = 0.05;
% 模型中的 Ibat 与 IL 正方向相反。统一后的电流一致性残差为
% IL_meas-currentDirectionFactor*Ibat_meas，即当前模型使用 IL_meas+Ibat_meas。
cfg.currentDirectionFactor = -1;
cfg.powerDirectionConvention = ...
    ["Pbat=Vbat_meas*Ibat_meas，符号沿用电池测量块；" + ...
     "Psource=Vbus_meas*Ibus_source，当前传感器正方向下源吸收功率为负；" + ...
     "Pload=Vbus_meas*Iload_meas，负载吸收功率为正；" + ...
     "PowerBalanceResidual=EP_PSOURCE_SIGN*Psource+" + ...
     "EP_PBAT_SIGN*Pbat+EP_PLOAD_SIGN*Pload-Pstored。"];
cfg.allowedModes = [0 1 2]; % 0=待机，1=充电，2=放电
cfg.dutyLimits = [0 1];
cfg.power.sourceDirectionFactor = 1;
cfg.power.batteryDirectionFactor = 1;
cfg.power.loadDirectionFactor = -1;
cfg.power.balanceFilterAlpha = exp(-cfg.sampleTime/0.001);

% 开关器件测量必须快于 PWM 周期采样，否则固定相位会把某个开关的
% 导通电流误记为零。仅四个诊断量使用 1 us，其余日志仍按 50 us。
cfg.switchMeasurement.loggingSampleTime = 1e-6;
cfg.switchMeasurement.currentThreshold = 0.5;
cfg.switchMeasurement.variableNames = [ ...
    "log_S1_device_current", "log_S1_device_voltage", ...
    "log_S2_device_current", "log_S2_device_voltage"];

% 传感器链统一为：偏移注入 -> 白噪声 -> 量化 -> 零阶保持。
% RandomSeed 会传入模型工作区，使各 Run 可复现。
cfg.sensor.sampleTime = cfg.sampleTime;
cfg.sensor.vbusNoiseStd = 0.05;
cfg.sensor.vbatNoiseStd = 0.02;
cfg.sensor.ibatNoiseStd = 0.01;
cfg.sensor.ilNoiseStd = 0.02;
cfg.sensor.vbusQuantStep = 0.01;
cfg.sensor.vbatQuantStep = 0.01;
cfg.sensor.currentQuantStep = 0.001;

cfg.control.chargeCurrentLimit = 25;
cfg.control.dischargeCurrentLimit = 25;

% 默认采用可控的“小规模先导数据集”。扩大数组即可生成正式数据集。
cfg.cases.modeCommands = [0 1 2];
cfg.cases.socInit = [30 70];
cfg.cases.irefLevels = 10;
cfg.cases.vbusRef = 400;
cfg.cases.vbatInit = NaN;  % 当前电池模型由 SOC 决定初始电压。
cfg.cases.rload = 200;
cfg.cases.pload = [0 400]; % 作为故障前已存在的可控电流负载功率。
cfg.cases.faultStartTimes = 0.60;
cfg.cases.randomizeFaultStart = true;
cfg.cases.faultStartRange = [0.45 0.75];
% 默认故障持续 0.20 s；设为 Inf 可生成保持到仿真结束的永久故障。
cfg.cases.faultDurations = 0.20;
cfg.cases.stopTime = 1.00;
cfg.cases.repetitions = 2;
cfg.cases.randomSeedBase = 240727;
cfg.cases.maxRunCount = 5000;
% 可选：保留原始全工况编号后，只运行指定工况；空数组表示全部工况。
cfg.cases.operatingPointIDs = strings(0,1);
cfg.cases.operatingPointPrefix = "op";
% 不同采集阶段使用不同前缀，后续合并时不会发生 RunID 重名。
cfg.cases.runIDPrefix = "run";
% 用已有 FD_RBAT/FD_CBUS/FD_CBUS_ESR 变量做确定性域随机化，使重复运行
% 不再只是更换一个没有作用的 RandomSeed 元数据。
cfg.cases.domainRandomization.enabled = true;
cfg.cases.domainRandomization.RbatNominal = 0.5;
cfg.cases.domainRandomization.CbusNominal = 2.2e-3;
cfg.cases.domainRandomization.CbusESRNominal = 1e-3;
cfg.cases.domainRandomization.RbatRelativeRange = 0.10;
cfg.cases.domainRandomization.CbusRelativeRange = 0.10;
cfg.cases.domainRandomization.CbusESRRelativeRange = 0.20;

% 标签策略：传感器故障按物理激活标注；开关开路只有在对应门极本应导通
% 时才可用于故障分类。故障场景的故障前窗口、不可观测开路窗口和过渡窗口
% 保留在数据中，但默认不进入训练。
cfg.labeling.sensorFaultIDs = [1 2];
cfg.labeling.switchFaultIDs = [3 4];
cfg.labeling.s1OpenFaultID = 3;
cfg.labeling.s2OpenFaultID = 4;
cfg.labeling.activeRatioThreshold = 0.5;
cfg.labeling.observableRatioThreshold = 0.5;
cfg.labeling.minimumGateDuty = 0.01;
cfg.labeling.excludeTransitionFromTraining = true;
cfg.labeling.excludePrefaultFromFaultRuns = true;

% 数据集标签 1~4 与模型内部故障代号 2、3、6、7 分离。
% Magnitudes 是每类故障的可配置严重度；开路故障的 1 表示完全开路。
cfg.faultList = table( ...
    [0; 1; 2; 3; 4], ...
    ["healthy"; "vbus_sensor_bias"; "inductor_current_sensor_bias"; ...
     "switch_S1_open"; "switch_S2_open"], ...
    ["none"; "DC_bus_voltage_sensor"; "inductor_current_sensor"; ...
     "switch_S1"; "switch_S2"], ...
    [0; 2; 3; 6; 7], ...
    [0; 2; 3; 0; 0], ...
    [""; "FD_VBUS_BIAS"; "FD_IL_BIAS"; "FD_S1_OPEN"; "FD_S2_OPEN"], ...
    {0; [-10 -5 5 10]; [-1 -0.5 0.5 1]; 1; 1}, ...
    'VariableNames', {'FaultID','FaultName','FaultLocation', ...
    'ModelFaultID','ConfiguredFaultID','FaultVariable','Magnitudes'});

% 当前模型适配情况。脚本不会静默假装未接入的参数已经生效。
cfg.compatibility.supportsIrefLevel = true;
cfg.compatibility.supportsFaultEndTime = true;
cfg.compatibility.supportsIndependentVbatInit = false;
cfg.compatibility.notes = [ ...
    "Mode_Manager 支持 FD_IREF_LEVEL 覆盖，最终仍受根层电流限幅保护。", ...
    "母线电压/电感电流偏移及 S1/S2 开路均支持 FaultEndTime 撤销。", ...
    "SOCInit 按 Battery 的有效容量系数换算，表示测量口实际初始 SOC。", ...
    "Battery 的初始端电压由 SOC 和电池模型决定，VbatInit 暂作元数据。", ...
    "PIiIntegral 从状态数据集 xout 中读取。"];

% 模型变量与可调模块路径
cfg.adapter.modelWorkspace = cfg.modelName;
cfg.adapter.batteryBlock = cfg.modelName + "/Battery";
cfg.adapter.batterySOCParameter = "SOC";
cfg.adapter.correctBatterySOCDefinition = true;
cfg.adapter.batterySideCapacitorBlock = ...
    cfg.modelName + "/Series RLC Branch1";
cfg.adapter.batterySideCapacitorVoltageParameter = "InitialVoltage";
cfg.adapter.synchronizeBatterySideCapacitor = true;
cfg.adapter.switchS1SID = "3";
cfg.adapter.switchS2SID = "4";
cfg.adapter.powerguiSID = "12";
cfg.adapter.switchRonParameter = "Ron";
cfg.adapter.switchRonNominal = 1e-3;
% gate_blocking: 连续/部分/间歇开路；high_resistance: 整段高导通电阻。
cfg.adapter.switchFaultMechanism = "gate_blocking";
cfg.adapter.switchFaultPeriod = 487e-6;
cfg.adapter.loadResistanceBlock = cfg.modelName + "/R_H_Bus";
cfg.adapter.loadResistanceParameter = "Resistance";
cfg.adapter.vbusReferenceBlock = cfg.modelName + "/V_ref";
cfg.adapter.vbusReferenceParameter = "Value";
cfg.adapter.protectionMode = 2; % 数据采集默认仅监测，不主动切断。
cfg.adapter.loadStepTime = 0;
cfg.adapter.modelVariables = [ ...
    "FD_FAULT_TIME", "FD_FAULT_END_TIME", ...
    "FD_LOAD_STEP_TIME", "FD_MODE_OVERRIDE_ENABLE", ...
    "FD_MODE_COMMAND", "FD_PROTECTION_MODE", "FD_FAULT_ID", ...
    "FD_VBUS_BIAS", "FD_VBAT_BIAS", "FD_IBAT_BIAS", "FD_IL_BIAS", ...
    "FD_DUTY_BIAS", "FD_DUTY_STUCK_ENABLE", "FD_DUTY_STUCK_VALUE", ...
    "FD_LOAD_STEP_A", "FD_S1_OPEN", "FD_S2_OPEN", "FD_RBAT", ...
    "FD_CBUS", "FD_CBUS_ESR", "FD_L_MAIN", "FD_TS", ...
    "FD_SWITCH_FAULT_PERIOD", ...
    "FD_SENSOR_TS", "FD_VBUS_NOISE_STD", "FD_VBAT_NOISE_STD", ...
    "FD_IBAT_NOISE_STD", "FD_IL_NOISE_STD", ...
    "FD_VBUS_QUANT_STEP", "FD_VBAT_QUANT_STEP", ...
    "FD_CURRENT_QUANT_STEP", "FD_IREF_OVERRIDE_ENABLE", ...
    "FD_IREF_LEVEL", "I_charge_max", "I_discharge_cmd", "SOCInit", ...
    "EP_PSOURCE_SIGN", "EP_PBAT_SIGN", "EP_PLOAD_SIGN", ...
    "EP_POWER_BALANCE_ALPHA", "EP_ENERGY_INITIAL", "RandomSeed"];

% 统一原始字段到实际 Simulink 输出的映射。读取顺序为：
% logsout 元素 -> SimulationOutput 中的 To Workspace 变量 -> 派生/NaN。
sn = struct();
sn.ModeCommand = ["ModeCommand", "log_mode_command"];
sn.ModeID = ["mode_id"];
sn.Iref = ["Iref", "log_Iref"];
sn.ILMeas = ["IL_meas", "log_I_L"];
sn.IbatMeas = ["Ibat_meas", "log_Ibat"];
sn.VbusMeas = ["Vbus_meas", "log_Vbus"];
sn.VbatMeas = ["Vbat_meas", "log_Vbat"];
% I_Rh 是 Multimeter 中的另一条支路量，不是总负载电流，禁止作为回退。
sn.IloadMeas = ["Iload_meas", "load_current"];
sn.SOCEst = ["SOC_est"];
sn.CurrentError = ["CurrentError", "log_Ierr", "log_I_tracking_error"];
sn.VoltageError = ["VoltageError", "log_V_err"];
sn.DutyRaw = ["DutyRaw", "log_Duty_cmd"];
sn.DutyApplied = ["DutyApplied", "log_Duty_applied"];
sn.PIiOut = ["PIiOut", "Duty_raw"];
sn.PIiIntegral = ["PIiIntegral"];
sn.PIvOut = ["PIvOut", "PI_voltage"];
sn.SatFlagI = ["sat_flag_I"];
sn.SatFlagV = ["sat_flag_V"];
sn.S1GateCmd = ["S1GateCmd", "gate_buck_cmd"];
sn.S2GateCmd = ["S2GateCmd", "gate_boost_cmd"];
sn.S1GateActual = ["S1GateActualHF", "log_gate_S1_actual"];
sn.S2GateActual = ["S2GateActualHF", "log_gate_S2_actual"];
sn.S1GateMismatch = ["S1GateMismatchHF", "log_gate_mismatch_S1"];
sn.S2GateMismatch = ["S2GateMismatchHF", "log_gate_mismatch_S2"];
sn.S1DeviceCurrent = ["S1DeviceCurrent", "log_S1_device_current"];
sn.S1DeviceVoltage = ["S1DeviceVoltage", "log_S1_device_voltage"];
sn.S2DeviceCurrent = ["S2DeviceCurrent", "log_S2_device_current"];
sn.S2DeviceVoltage = ["S2DeviceVoltage", "log_S2_device_voltage"];
sn.ILTrue = ["IL_true", "log_IL_true"];
sn.IbatTrue = ["Ibat_true", "log_Ibat_true"];
sn.VbusTrue = ["Vbus_true", "log_Vbus_true"];
sn.VbatTrue = ["Vbat_true", "log_Vbat_true"];
sn.SOCTrue = ["SOC_true", "log_SOC"];
sn.IbusSource = ["Ibus_source", "source_current"];
sn.PsourceMeas = ["Psource_meas"];
sn.PloadMeas = ["Pload_meas"];
sn.PstoredMeas = ["Pstored_meas"];
sn.PowerBalanceResidual = ["PowerBalanceResidual"];
sn.FaultActive = ["FaultActive", "log_fault_active"];
sn.TransitionWindow = ["TransitionWindow"];
cfg.signalNames = sn;

% 必要信号缺失时整次运行失败；可选信号缺失时固定保留 NaN 列。
cfg.requiredRawFields = [ ...
    "ModeCommand", "Iref", "IL_meas", "Ibat_meas", "Vbus_meas", ...
    "Vbat_meas", "Iload_meas", "SOC_est", "CurrentError", "DutyApplied", ...
    "S1GateCmd", "S2GateCmd", "S1GateDuty", "S2GateDuty", ...
    "IL_true", "Ibat_true", "Vbus_true", "Vbat_true", ...
    "Ibus_source", "Psource_meas", "Pload_meas", ...
    "Pstored_meas", "PowerBalanceResidual"];
cfg.optionalRawFields = [ ...
    "ConverterEnable", "VbusRef", "VoltageError", ...
    "DutyRaw", "PIiOut", "PIiIntegral", "PIvOut", ...
    "SatFlagI", "SatFlagV", "SatFlag", ...
    "SOC_true", "FaultActive", ...
    "Validation_S1GateActual", "Validation_S2GateActual", ...
    "Validation_S1GateMismatch", "Validation_S2GateMismatch", ...
    "S1_device_current", "S1_device_voltage", ...
    "S2_device_current", "S2_device_voltage", ...
    "TransitionWindow"];
cfg.discreteRawFields = [ ...
    "ModeCommand", "ConverterEnable", "FaultActive", ...
    "TransitionWindow", "SatFlag", "S1GateCmd", "S2GateCmd"];

cfg.featureSignals = [ ...
    "IL_meas", "Ibat_meas", "Vbus_meas", "Vbat_meas", ...
    "Iload_meas", "SOC_est", "Iref", "VbusRef", "CurrentError", ...
    "VoltageError", "DutyRaw", "DutyApplied", "PIiOut", ...
    "PIiIntegral", "PIvOut", "Psource_meas", "Pload_meas", ...
    "Pstored_meas", "PowerBalanceResidual", ...
    "S1_conduction_ratio", "S1_ron_estimate", ...
    "S2_conduction_ratio", "S2_ron_estimate"];
cfg.currentFeatureSignals = ["IL_meas", "Ibat_meas", "Iload_meas"];
cfg.trueFeatureSignals = ["IL_true", "Ibat_true", "Vbus_true", ...
    "Vbat_true", "SOC_true"];

% SOC_est 现在来自独立库仑计；完整功率平衡残差可用于模型输入。
cfg.featurePolicy.useIdealSOCEst = true;
cfg.featurePolicy.useUnbalancedPowerResidual = false;
cfg.featurePolicy.useBalancedPowerResidual = true;
cfg.featurePolicy.excludeAllNaN = true;
cfg.featurePolicy.excludeConstant = true;
cfg.featurePolicy.excludeNearZeroVariance = true;
cfg.featurePolicy.nearZeroRelativeTolerance = 1e-12;

% 执行和输出设置
cfg.execution.useParallel = true;
cfg.execution.numWorkers = 2;
cfg.execution.parallelBatchSize = 16;
cfg.execution.showSimulationManager = false;
cfg.execution.stopOnCompatibilityError = true;
cfg.execution.useFastRestart = false;
% NaN 表示沿用模型设置；正式扩充前会用基准仿真验证 5 us 是否足够精确。
cfg.execution.fixedStep = NaN;
% 全模型状态在 1 us 固定步长下会生成数十 GB 的临时 DMR。PIiIntegral
% 是可选特征；默认不为它保存所有模型状态。
cfg.execution.saveModelStates = false;
cfg.output.root = fullfile(scriptFolder, "dataset_output");
cfg.output.rawRuns = fullfile(cfg.output.root, "raw_runs");
cfg.output.combined = fullfile(cfg.output.root, "combined");
cfg.output.figures = fullfile(cfg.output.root, "figures");
cfg.output.cache = fullfile(cfg.output.root, "simulink_cache");
cfg.output.codegen = fullfile(cfg.output.root, "simulink_codegen");
cfg.output.temp = fullfile(cfg.output.root, "temp");
cfg.output.parallelJobs = fullfile(cfg.output.root, "parallel_jobs");
cfg.output.saveRunCSV = false;
cfg.output.saveCombinedCSV = true;
cfg.output.overwritePolicy = "resume"; % "resume"、"overwrite" 或 "error"

cfg = mergeStruct(cfg, overrides);
if isfield(overrides, "modelName")
    % modelFile and model-scoped block paths are derived from modelName.
    % Rebase them unless the caller supplied an explicit value, so an
    % experiment can safely target a copied model with one override.
    if ~isfield(overrides, "modelFile")
        cfg.modelFile = fullfile( ...
            simulinkFolder, "models", cfg.modelName + ".slx");
    end
    adapterOverrides = struct();
    if isfield(overrides, "adapter")
        adapterOverrides = overrides.adapter;
    end
    if ~isfield(adapterOverrides, "modelWorkspace")
        cfg.adapter.modelWorkspace = cfg.modelName;
    end
    if ~isfield(adapterOverrides, "batteryBlock")
        cfg.adapter.batteryBlock = cfg.modelName + "/Battery";
    end
    if ~isfield(adapterOverrides, "batterySideCapacitorBlock")
        cfg.adapter.batterySideCapacitorBlock = ...
            cfg.modelName + "/Series RLC Branch1";
    end
    if ~isfield(adapterOverrides, "loadResistanceBlock")
        cfg.adapter.loadResistanceBlock = cfg.modelName + "/R_H_Bus";
    end
    if ~isfield(adapterOverrides, "vbusReferenceBlock")
        cfg.adapter.vbusReferenceBlock = cfg.modelName + "/V_ref";
    end
end
if isfield(overrides, "output") && isfield(overrides.output, "root")
    if ~isfield(overrides.output, "rawRuns")
        cfg.output.rawRuns = fullfile(cfg.output.root, "raw_runs");
    end
    if ~isfield(overrides.output, "combined")
        cfg.output.combined = fullfile(cfg.output.root, "combined");
    end
    if ~isfield(overrides.output, "figures")
        cfg.output.figures = fullfile(cfg.output.root, "figures");
    end
    if ~isfield(overrides.output, "cache")
        cfg.output.cache = fullfile(cfg.output.root, "simulink_cache");
    end
    if ~isfield(overrides.output, "codegen")
        cfg.output.codegen = fullfile(cfg.output.root, "simulink_codegen");
    end
    if ~isfield(overrides.output, "temp")
        cfg.output.temp = fullfile(cfg.output.root, "temp");
    end
    if ~isfield(overrides.output, "parallelJobs")
        cfg.output.parallelJobs = fullfile(cfg.output.root, "parallel_jobs");
    end
end
if ~cfg.execution.saveModelStates
    cfg.featureSignals(cfg.featureSignals == "PIiIntegral") = [];
end
end

function target = mergeStruct(target, source)
% 递归覆盖结构体；便于在调用端只改少量参数。
names = fieldnames(source);
for k = 1:numel(names)
    name = names{k};
    if isfield(target, name) && isstruct(target.(name)) && ...
            isstruct(source.(name))
        target.(name) = mergeStruct(target.(name), source.(name));
    else
        target.(name) = source.(name);
    end
end
end
