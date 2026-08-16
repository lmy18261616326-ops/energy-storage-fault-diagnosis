%% Train a grouped sensor-bias classifier for v04
% Whole simulation runs are assigned to train or test sets to avoid
% leakage between overlapping windows from the same simulation.

scriptFolder = fileparts(mfilename("fullpath"));
experimentFolder = fileparts(scriptFolder);
resultsFolder = fullfile(experimentFolder, "results", "full");
featurePath = fullfile( ...
    resultsFolder, "sensor_bias_feature_dataset.csv");

features = readtable(featurePath, "TextType", "string");
features = rmmissing(features);
features = features(~features.TransitionWindow, :);

faultClasses = unique(features.FaultID, "stable");
testRun = strings(0, 1);

for classIndex = 1:numel(faultClasses)
    classRows = features.FaultID == faultClasses(classIndex);
    classRuns = unique(features.RunID(classRows), "stable");
    if numel(classRuns) >= 2
        testRun(end + 1, 1) = classRuns(end); %#ok<SAGROW>
    end
end

isTest = ismember(features.RunID, unique(testRun));
isTrain = ~isTest;

excluded = [ ...
    "Scenario", "RunID", "FaultName", ...
    "ExpectedFaultID", "FaultID", "ModeCommand", ...
    "Severity", "BiasSign", "WindowStart", ...
    "FaultActive", "TransitionWindow"];
predictorNames = setdiff( ...
    string(features.Properties.VariableNames), excluded, "stable");

trainX = features{isTrain, predictorNames};
trainY = features.FaultID(isTrain);
testX = features{isTest, predictorNames};
testY = features.FaultID(isTest);

predictorMean = mean(trainX, 1);
predictorScale = std(trainX, 0, 1);
predictorScale(predictorScale < eps) = 1;
trainZ = (trainX - predictorMean) ./ predictorScale;
testZ = (testX - predictorMean) ./ predictorScale;

classIDs = unique(trainY, "stable");
centroids = zeros(numel(classIDs), size(trainZ, 2));
for classIndex = 1:numel(classIDs)
    centroids(classIndex, :) = ...
        mean(trainZ(trainY == classIDs(classIndex), :), 1);
end

distance = zeros(size(testZ, 1), numel(classIDs));
for classIndex = 1:numel(classIDs)
    difference = testZ - centroids(classIndex, :);
    distance(:, classIndex) = sum(difference .^ 2, 2);
end
[~, predictedIndex] = min(distance, [], 2);
predictedY = classIDs(predictedIndex);

accuracy = mean(predictedY == testY);
classList = unique([trainY; testY], "stable");
precision = zeros(numel(classList), 1);
recall = zeros(numel(classList), 1);
f1 = zeros(numel(classList), 1);

for classIndex = 1:numel(classList)
    classID = classList(classIndex);
    truePositive = sum(predictedY == classID & testY == classID);
    falsePositive = sum(predictedY == classID & testY ~= classID);
    falseNegative = sum(predictedY ~= classID & testY == classID);
    precision(classIndex) = ...
        truePositive / max(truePositive + falsePositive, 1);
    recall(classIndex) = ...
        truePositive / max(truePositive + falseNegative, 1);
    f1(classIndex) = 2 * precision(classIndex) * ...
        recall(classIndex) / ...
        max(precision(classIndex) + recall(classIndex), eps);
end

metrics = table(accuracy, mean(f1), ...
    'VariableNames', {'Accuracy', 'MacroF1'});
classMetrics = table( ...
    classList, precision, recall, f1, ...
    'VariableNames', ...
    {'FaultID', 'Precision', 'Recall', 'F1'});

writetable(metrics, ...
    fullfile(resultsFolder, "sensor_classifier_metrics.csv"));
writetable(classMetrics, ...
    fullfile(resultsFolder, ...
    "sensor_classifier_class_metrics.csv"));

confusionMatrix = zeros(numel(classList), numel(classList));
for rowIndex = 1:numel(classList)
    for columnIndex = 1:numel(classList)
        confusionMatrix(rowIndex, columnIndex) = sum( ...
            testY == classList(rowIndex) & ...
            predictedY == classList(columnIndex));
    end
end

figureHandle = figure( ...
    "Color", "white", ...
    "Name", "Sensor-bias classifier confusion matrix");
imagesc(confusionMatrix);
axis equal tight;
colorbar;
xlabel("Predicted fault ID");
ylabel("True fault ID");
title("Run-grouped sensor-bias classification");
xticks(1:numel(classList));
yticks(1:numel(classList));
xticklabels(string(classList));
yticklabels(string(classList));

for rowIndex = 1:numel(classList)
    for columnIndex = 1:numel(classList)
        text(columnIndex, rowIndex, ...
            string(confusionMatrix(rowIndex, columnIndex)), ...
            "HorizontalAlignment", "center", ...
            "Color", "white", "FontWeight", "bold");
    end
end

exportgraphics(figureHandle, ...
    fullfile(resultsFolder, ...
    "sensor_classifier_confusion_matrix.png"), ...
    "Resolution", 180);

model = struct( ...
    "Type", "standardized_nearest_centroid", ...
    "PredictorNames", predictorNames, ...
    "PredictorMean", predictorMean, ...
    "PredictorScale", predictorScale, ...
    "ClassIDs", classIDs, ...
    "Centroids", centroids);

save(fullfile(resultsFolder, "sensor_bias_classifier.mat"), ...
    "model", "predictorNames", "metrics", ...
    "classMetrics", "testRun", "confusionMatrix");

disp(metrics);
disp(classMetrics);
fprintf("MODEL=%s\n", ...
    fullfile(resultsFolder, "sensor_bias_classifier.mat"));
