clear; clc;

projectRoot = fileparts(mfilename('fullpath'));
modelsDir = fullfile(projectRoot, 'simulink', 'models');
cd(modelsDir);
addpath(modelsDir);
model = getenv('ENERGY_STORAGE_MODEL');
if isempty(model)
    model = 'main_model_fd_v05_energyprotect';
end
modelFile = fullfile(modelsDir, [model '.slx']);
assert(isfile(modelFile), 'Model file does not exist: %s', modelFile);
load_system(modelFile);

targets = {
    [model '/Battery']
    [model '/DC Voltage Source']
    [model '/PWM Generator\n(DC-DC)']
    [model '/PWM Generator\n(DC-DC)1']
    [model '/IGBT//Diode']
    [model '/IGBT//Diode1']
    [model '/Step']
    [model '/Step1']
    [model '/Series RLC Branch']
    [model '/Series RLC Branch1']
    [model '/Series RLC Branch2']
    [model '/Series RLC Branch3']
};

fprintf('== Key block parameters ==\n');
for k = 1:numel(targets)
    target = targets{k};
    target = strrep(target, '\n', newline);
    try
        fprintf('\n-- %s --\n', target);
        maskNames = get_param(target, 'MaskNames');
        maskValues = get_param(target, 'MaskValues');
        if ~isempty(maskNames)
            for n = 1:numel(maskNames)
                fprintf('%s = %s\n', maskNames{n}, maskValues{n});
            end
        else
            objParams = get_param(target, 'ObjectParameters');
            names = intersect({'Time','Before','After','SampleTime','Amplitude','Frequency','Resistance','Inductance','Capacitance','BranchType'}, fieldnames(objParams), 'stable');
            for n = 1:numel(names)
                fprintf('%s = %s\n', names{n}, string(get_param(target, names{n})));
            end
        end
    catch ME
        fprintf('\n-- %s --\nERROR: %s\n', target, ME.message);
    end
end

bdclose(model);
