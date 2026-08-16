function proj = configure_project()
%CONFIGURE_PROJECT Create or refresh the MATLAB Project definition.
%   The function keeps generated caches under the ignored repository-local
%   work folder, registers source artifacts, and adds only resolvable source
%   folders to the MATLAB path. It is safe to run repeatedly.

root = string(fileparts(mfilename('fullpath')));
projectFile = fullfile(root, "EnergyStorageFaultDiagnosis.prj");

try
    openProj = currentProject;
    if ~strcmpi(string(openProj.RootFolder), root)
        close(openProj);
    else
        proj = openProj;
    end
catch
    % No project is open.
end

if ~exist('proj', 'var')
    if isfile(projectFile)
        proj = openProject(projectFile);
    else
        proj = matlab.project.createProject( ...
            "Folder", root, "Name", "EnergyStorageFaultDiagnosis");
    end
end

workFolder = fullfile(root, "work");
if ~isfolder(workFolder)
    mkdir(workFolder);
end
proj.SimulinkCacheFolder = fullfile(workFolder, "simulink_cache");
proj.SimulinkCodeGenFolder = fullfile(workFolder, "simulink_codegen");
proj.DependencyCacheFile = fullfile(workFolder, "dependency_cache.graphml");

pathFolders = [ ...
    "simulink/models"
    "simulink/experiments/sensor_bias/scripts"
    "simulink/experiments/healthy_cycle/scripts"
    "simulink/experiments/current_performance/scripts"
    "ML/src"
    "ML/scripts"
];
for k = 1:numel(pathFolders)
    if isfolder(fullfile(root, pathFolders(k)))
        addPath(proj, pathFolders(k));
    end
end

rootFiles = [ ...
    "README.md"
    ".gitignore"
    ".gitattributes"
    "configure_project.m"
    "audit_main_model.m"
    "audit_main_model_params.m"
    "audit_main_model_topology.m"
    "audit_other_models.m"
    "start_project_matlab.cmd"
];
for k = 1:numel(rootFiles)
    if isfile(fullfile(root, rootFiles(k)))
        addFile(proj, rootFiles(k));
    end
end

safeTrees = ["docs/specs", "ML/src", "ML/scripts", "ML/tests", "ML/configs"];
for k = 1:numel(safeTrees)
    folder = fullfile(root, safeTrees(k));
    if isfolder(folder)
        addFolderIncludingChildFiles(proj, safeTrees(k));
    end
end

documentationFiles = [ ...
    "docs/PROJECT_STATUS.md"
    "docs/REPRODUCIBILITY.md"
    "docs/学习与面试应答攻略.md"
    "docs/学习与面试应答攻略.docx"
];
for k = 1:numel(documentationFiles)
    if isfile(fullfile(root, documentationFiles(k)))
        addFile(proj, documentationFiles(k));
    end
end

modelFiles = [ ...
    "simulink/models/main_model_fd_v03_faultdiag.slx"
    "simulink/models/main_model_fd_v04_faultdiag.slx"
    "simulink/models/main_model_fd_v05_energyprotect.slx"
    "simulink/models/main_model_fd_v06_switchobservability.slx"
    "simulink/README.md"
    "simulink/experiments/sensor_bias/README.md"
];
for k = 1:numel(modelFiles)
    if isfile(fullfile(root, modelFiles(k)))
        addFile(proj, modelFiles(k));
    end
end

experimentScripts = dir(fullfile(root, "simulink", "experiments", "**", "*.m"));
for k = 1:numel(experimentScripts)
    sourceFile = string(fullfile(experimentScripts(k).folder, experimentScripts(k).name));
    if contains(sourceFile, filesep + "dataset_output") || ...
            contains(sourceFile, filesep + "results" + filesep)
        continue;
    end
    relativeFile = extractAfter(sourceFile, strlength(root) + 1);
    addFile(proj, relativeFile);
end

fprintf('PROJECT_FILE=%s\n', projectFile);
fprintf('PROJECT_ROOT=%s\n', proj.RootFolder);
fprintf('PROJECT_FILES=%d\n', numel(proj.Files));
fprintf('CACHE_FOLDER=%s\n', proj.SimulinkCacheFolder);
fprintf('CODEGEN_FOLDER=%s\n', proj.SimulinkCodeGenFolder);

results = runChecks(proj);
fprintf('PROJECT_CHECKS=%d\n', numel(results));
updateDependencies(proj);
fprintf('PROJECT_DEPENDENCIES_UPDATED=1\n');
end
