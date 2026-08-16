function [trainTable, validationTable, testTable, splitSummary] = ...
    split_dataset_by_run(featureDataset, ratios, randomSeed, ...
    outputFolder, groupField, eligibleOnly)
%SPLIT_DATASET_BY_RUN 按独立工况组划分训练、验证和测试集。
%   默认按 OperatingPointID 分组，保证同一工况的所有故障、严重度和重复
%   运行只出现在一个集合中；这比只隔离 RunID 更能避免相邻工况泄漏。
%
%   [TRAIN, VALIDATION, TEST, SUMMARY] = SPLIT_DATASET_BY_RUN(FEATURES)
%   使用 70%/15%/15%，仅划分 IsTrainingEligible=1 的窗口。
%
%   可选输入:
%       ratios       - [训练 验证 测试]，和必须为 1。
%       randomSeed   - 分组随机种子。
%       outputFolder - 非空时保存 MAT/CSV 和 split_summary.csv。
%       groupField   - 默认 "OperatingPointID"；可设为 "RunID" 做较宽松
%                      的 IID 评估。
%       eligibleOnly - 默认 true，排除过渡、故障前和不可观测开路窗口。

arguments
    featureDataset table
    ratios (1,3) double = [0.70 0.15 0.15]
    randomSeed (1,1) double {mustBeInteger,mustBeNonnegative} = 240727
    outputFolder (1,1) string = ""
    groupField (1,1) string = "OperatingPointID"
    eligibleOnly (1,1) logical = true
end

if abs(sum(ratios)-1) > 1e-12 || any(ratios < 0)
    error("faultdataset:InvalidSplitRatio", ...
        "ratios 必须非负且总和为 1。");
end
if ~ismember(groupField, ["OperatingPointID","RunID"])
    error("faultdataset:InvalidGroupField", ...
        "groupField 必须是 OperatingPointID 或 RunID。");
end
required = unique(["RunID","OperatingPointID","ScenarioFaultID", ...
    "WindowFaultID",groupField]);
if eligibleOnly
    required(end+1) = "IsTrainingEligible";
end
missing = setdiff(required, ...
    string(featureDataset.Properties.VariableNames));
if ~isempty(missing)
    error("faultdataset:SplitMissingColumns", ...
        "数据集划分缺少字段: %s", strjoin(missing, ", "));
end

working = featureDataset;
if eligibleOnly
    working = working(working.IsTrainingEligible ~= 0,:);
end
if isempty(working)
    error("faultdataset:NoEligibleWindows", ...
        "筛选后没有可用于划分的训练窗口。");
end

rng(randomSeed, "twister");
if groupField == "RunID"
    groupInfo = unique(working(:,["RunID","ScenarioFaultID"]));
    if numel(unique(string(groupInfo.RunID))) ~= height(groupInfo)
        error("faultdataset:RunHasMultipleLabels", ...
            "同一个 RunID 对应多个 ScenarioFaultID。");
    end
    groupInfo.GroupID = string(groupInfo.RunID);
    groupInfo.Split = strings(height(groupInfo),1);
    classes = unique(groupInfo.ScenarioFaultID, "sorted");
    for classID = classes'
        indices = find(groupInfo.ScenarioFaultID == classID);
        indices = indices(randperm(numel(indices)));
        groupInfo.Split(indices) = allocateNames(numel(indices), ratios);
    end
else
    groupIDs = unique(string(working.(groupField)), "stable");
    groupIDs = groupIDs(randperm(numel(groupIDs)));
    groupInfo = table(groupIDs, ...
        allocateNames(numel(groupIDs), ratios), ...
        'VariableNames', {'GroupID','Split'});
end

trainGroups = groupInfo.GroupID(groupInfo.Split == "train");
validationGroups = groupInfo.GroupID(groupInfo.Split == "validation");
testGroups = groupInfo.GroupID(groupInfo.Split == "test");
values = string(working.(groupField));
trainTable = working(ismember(values,trainGroups),:);
validationTable = working(ismember(values,validationGroups),:);
testTable = working(ismember(values,testGroups),:);

checkLeakage(trainTable, validationTable, testTable, "RunID");
checkLeakage(trainTable, validationTable, testTable, ...
    "OperatingPointID");

splitSummary = summarizeSplits( ...
    trainTable, validationTable, testTable);
fprintf(["按 %s 划分完成：train=%d groups，validation=%d groups，" + ...
    "test=%d groups；仅训练合格窗口=%s。\n"], ...
    groupField, numel(trainGroups), numel(validationGroups), ...
    numel(testGroups), string(eligibleOnly));
disp(splitSummary);

if outputFolder ~= ""
    if ~isfolder(outputFolder)
        mkdir(outputFolder);
    end
    save(fullfile(outputFolder,"dataset_split.mat"), ...
        "trainTable","validationTable","testTable","groupInfo", ...
        "splitSummary","groupField","eligibleOnly","-v7.3");
    writetable(trainTable,fullfile(outputFolder,"train_dataset.csv"));
    writetable(validationTable, ...
        fullfile(outputFolder,"validation_dataset.csv"));
    writetable(testTable,fullfile(outputFolder,"test_dataset.csv"));
    writetable(splitSummary,fullfile(outputFolder,"split_summary.csv"));
end
end

function names = allocateNames(count, ratios)
names = strings(count,1);
if count == 0
    return;
end
[trainCount, validationCount] = allocateCounts(count, ratios);
names(1:trainCount) = "train";
names(trainCount+1:trainCount+validationCount) = "validation";
names(trainCount+validationCount+1:end) = "test";
end

function [trainCount, validationCount] = allocateCounts(count, ratios)
if count == 1
    trainCount = 1;
    validationCount = 0;
    return;
elseif count == 2
    trainCount = 1;
    validationCount = 0;
    return;
end
counts = floor(count .* ratios);
remainder = count - sum(counts);
[~, order] = sort(count.*ratios-counts, "descend");
for k = 1:remainder
    counts(order(k)) = counts(order(k)) + 1;
end
% 组数不少于 3 时确保三个集合均非空。
for target = 1:3
    if counts(target) == 0
        [~, donor] = max(counts);
        counts(donor) = counts(donor)-1;
        counts(target) = 1;
    end
end
trainCount = counts(1);
validationCount = counts(2);
end

function checkLeakage(train, validation, test, fieldName)
trainIDs = unique(string(train.(fieldName)));
validationIDs = unique(string(validation.(fieldName)));
testIDs = unique(string(test.(fieldName)));
leak = [intersect(trainIDs,validationIDs); ...
    intersect(trainIDs,testIDs); ...
    intersect(validationIDs,testIDs)];
if ~isempty(leak)
    error("faultdataset:GroupLeakage", ...
        "检测到 %s 跨集合泄漏: %s", ...
        fieldName, strjoin(unique(leak), ", "));
end
end

function summary = summarizeSplits(train, validation, test)
summary = table();
names = ["train","validation","test"];
tables = {train,validation,test};
for k = 1:3
    data = tables{k};
    if isempty(data)
        continue;
    end
    one = groupcounts(data,"WindowFaultID");
    oneNames = one.Properties.VariableNames;
    oneNames{strcmp(oneNames,"GroupCount")} = 'WindowCount';
    one.Properties.VariableNames = oneNames;
    pair = unique(data(:,["RunID","WindowFaultID"]));
    runCounts = groupcounts(pair,"WindowFaultID");
    runNames = runCounts.Properties.VariableNames;
    runNames{strcmp(runNames,"GroupCount")} = 'RunIDCount';
    runCounts.Properties.VariableNames = runNames;
    one = outerjoin(one,runCounts, ...
        "Keys","WindowFaultID","MergeKeys",true);
    one.Set = repmat(names(k),height(one),1);
    one = movevars(one,"Set","Before",1);
    summary = [summary;one]; %#ok<AGROW>
end
end
