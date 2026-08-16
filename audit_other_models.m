clear; clc;

projectRoot = fileparts(mfilename('fullpath'));
modelsDir = fullfile(projectRoot, 'simulink', 'models');
cd(modelsDir);
addpath(modelsDir);

files = {
    'main_model_fd_v03_faultdiag.slx'
    'main_model_fd_v04_faultdiag.slx'
    'main_model_fd_v05_energyprotect.slx'
    'main_model_fd_v06_switchobservability.slx'
};
for m = 1:numel(files)
    file = fullfile(modelsDir, files{m});
    [~, requestedModel] = fileparts(file);
    fprintf('\n================ %s ================\n', files{m});
    try
        load_system(file);
        openModels = find_system('type', 'block_diagram');
        model = openModels{end};
        if any(strcmp(openModels, requestedModel))
            model = requestedModel;
        end
        fprintf('MODEL_NAME=%s\n', model);
        fprintf('LOAD=OK StopTime=%s Solver=%s SolverType=%s\n', ...
            get_param(model,'StopTime'), get_param(model,'Solver'), get_param(model,'SolverType'));
        rootBlocks = find_system(model, 'SearchDepth', 1, 'Type', 'Block');
        fprintf('Root blocks:\n');
        for k = 1:numel(rootBlocks)
            if strcmp(rootBlocks{k}, model), continue; end
            fprintf('  %s | %s | %s\n', rootBlocks{k}, get_param(rootBlocks{k}, 'BlockType'), get_param(rootBlocks{k}, 'MaskType'));
        end
        fprintf('findVars:\n');
        try
            vars = Simulink.findVars(model);
            for k = 1:numel(vars)
                fprintf('  %s | %s | %s | Users=%d\n', vars(k).Name, vars(k).Source, vars(k).SourceType, numel(vars(k).Users));
            end
        catch ME
            fprintf('  findVars failed: %s\n', ME.message);
        end
        try
            set_param(model, 'SimulationCommand', 'update');
            fprintf('UPDATE=OK\n');
        catch ME
            fprintf('UPDATE=ERROR: %s\n', ME.message);
        end
        bdclose(model);
    catch ME
        fprintf('LOAD=ERROR: %s\n', ME.message);
    end
end
