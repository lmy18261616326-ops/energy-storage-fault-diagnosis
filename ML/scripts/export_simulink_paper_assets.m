function export_simulink_paper_assets
%EXPORT_SIMULINK_PAPER_ASSETS Read-only export of the verified converter model.
% This script deliberately never calls save_system or modifies block parameters.

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
modelPath = fullfile(projectRoot, 'simulink', 'models', ...
    'main_model_fd_v06_switchobservability.slx');
outDir = fullfile(projectRoot, 'ML', 'reports', ...
    'ieee_paper_simulink_enhanced_2026-08-04', 'figures');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

[modelDir, model, ~] = fileparts(modelPath);
addpath(modelDir);
load_system(modelPath);
cleanupObj = onCleanup(@() close_system(model, 0)); %#ok<NASGU>

cfgNames = {'SolverType','Solver','FixedStep','StartTime','StopTime', ...
    'RelTol','AbsTol','MaxStep','SimulationMode'};
cfg = struct();
for k = 1:numel(cfgNames)
    name = cfgNames{k};
    try
        cfg.(name) = get_param(model, name);
    catch
        cfg.(name) = '';
    end
end

blocks = find_system(model, 'SearchDepth', 2, 'FollowLinks', 'on', ...
    'LookUnderMasks', 'all', 'Type', 'Block');
records = repmat(struct('path','','sid','','name','','block_type','', ...
    'mask_type','','parent',''), numel(blocks), 1);
for k = 1:numel(blocks)
    p = blocks{k};
    records(k).path = p;
    records(k).name = get_param(p, 'Name');
    records(k).block_type = get_param(p, 'BlockType');
    records(k).parent = get_param(p, 'Parent');
    try
        records(k).sid = Simulink.ID.getSID(p);
    catch
        records(k).sid = '';
    end
    try
        records(k).mask_type = get_param(p, 'MaskType');
    catch
        records(k).mask_type = '';
    end
end

payload = struct('model', model, 'model_path', modelPath, ...
    'configuration', cfg, 'blocks_depth_2', records);
fid = fopen(fullfile(outDir, 'simulink_model_inventory.json'), 'w');
fprintf(fid, '%s', jsonencode(payload, PrettyPrint=true));
fclose(fid);

% Export the actual Simulink canvas. The model is closed without saving.
open_system(model);
print(['-s' model], '-dpng', '-r180', ...
    fullfile(outDir, 'simulink_top_level_model.png'));

% Export important top-level subsystems separately when present.
patterns = {'control','fault','protect','sensor','measure','log'};
top = find_system(model, 'SearchDepth', 1, 'Type', 'Block');
exported = strings(0,1);
for k = 1:numel(top)
    p = top{k};
    if ~strcmp(get_param(p, 'BlockType'), 'SubSystem')
        continue;
    end
    label = lower(get_param(p, 'Name'));
    if ~any(contains(label, patterns))
        continue;
    end
    open_system(p);
    safe = regexprep(get_param(p, 'Name'), '[^a-zA-Z0-9_-]+', '_');
    outPath = fullfile(outDir, ['simulink_subsystem_' safe '.png']);
    try
        print(['-s' p], '-dpng', '-r180', outPath);
        exported(end+1,1) = string(outPath); %#ok<AGROW>
    catch err
        warning('Could not export %s: %s', p, err.message);
    end
end

fid = fopen(fullfile(outDir, 'simulink_export_summary.txt'), 'w');
fprintf(fid, 'Model: %s\n', modelPath);
fprintf(fid, 'Top-level figure: simulink_top_level_model.png\n');
fprintf(fid, 'Solver type: %s\n', cfg.SolverType);
fprintf(fid, 'Solver: %s\n', cfg.Solver);
fprintf(fid, 'Fixed step: %s s\n', cfg.FixedStep);
fprintf(fid, 'Stop time stored in model: %s s\n', cfg.StopTime);
fprintf(fid, 'Blocks inspected (depth <= 2): %d\n', numel(records));
fprintf(fid, 'Subsystem exports: %d\n', numel(exported));
for k = 1:numel(exported)
    fprintf(fid, '  %s\n', exported(k));
end
fclose(fid);

disp('SIMULINK_PAPER_ASSET_EXPORT_COMPLETE');
disp(fullfile(outDir, 'simulink_top_level_model.png'));
end
