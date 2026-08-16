param(
    [ValidateRange(30, 3600)]
    [int]$PollSeconds = 60
)

$datasetRoot = Join-Path $PSScriptRoot 'dataset_output_v13'
$monitorRoot = Join-Path $datasetRoot 'monitoring'
$logPath = Join-Path $monitorRoot 'disk_usage.csv'
$statusPath = Join-Path $monitorRoot 'latest_status.json'
$pidPath = Join-Path $monitorRoot 'monitor.pid'
$criticalFlagPath = Join-Path $monitorRoot 'DISK_SPACE_CRITICAL.flag'
$phaseNames = @(
    'phase2_core_health',
    'phase3_sensor_faults',
    'phase4_switch_full_open',
    'phase5_switch_partial_open',
    'phase6_switch_intermittent',
    'phase7_switch_high_resistance',
    'phase8_bridge_health',
    'phase9_bridge_sensor_faults',
    'phase10_bridge_switch_full_open',
    'phase11_bridge_switch_partial_open',
    'phase12_bridge_switch_intermittent',
    'phase13_bridge_switch_high_resistance'
)

New-Item -ItemType Directory -Path $monitorRoot -Force | Out-Null

if (Test-Path -LiteralPath $pidPath) {
    $existingPid = 0
    [void][int]::TryParse(
        (Get-Content -LiteralPath $pidPath -Raw).Trim(),
        [ref]$existingPid)
    if ($existingPid -gt 0 -and
            (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
        exit 0
    }
}

Set-Content -LiteralPath $pidPath -Value $PID -Encoding ascii
$sampleIndex = 0
$datasetSizeGB = 0.0

try {
    while ($true) {
        $sampleIndex++
        $systemDrive = [System.IO.DriveInfo]::new($env:SystemDrive)
        $projectDrive = [System.IO.DriveInfo]::new(
            (Split-Path -Qualifier $datasetRoot))
        $freeCGB = [math]::Round($systemDrive.AvailableFreeSpace / 1GB, 3)
        $freeDGB = [math]::Round($projectDrive.AvailableFreeSpace / 1GB, 3)

        # Recalculate the recursive dataset size every five samples. Free
        # space itself is still checked on every sample.
        if ($sampleIndex -eq 1 -or ($sampleIndex % 5) -eq 0) {
            $datasetBytes = (
                Get-ChildItem -LiteralPath $datasetRoot -File -Recurse `
                    -ErrorAction SilentlyContinue |
                Measure-Object -Property Length -Sum
            ).Sum
            $datasetSizeGB = [math]::Round($datasetBytes / 1GB, 3)
        }

        $phaseCounts = [ordered]@{}
        foreach ($phaseName in $phaseNames) {
            $rawPath = Join-Path $datasetRoot "$phaseName\raw_runs"
            $phaseCounts[$phaseName] = if (Test-Path -LiteralPath $rawPath) {
                (Get-ChildItem -LiteralPath $rawPath -File -Filter '*.mat' `
                    -ErrorAction SilentlyContinue).Count
            } else {
                0
            }
        }

        $state = 'NORMAL'
        if ($freeDGB -lt 40 -or $freeCGB -lt 20) {
            $state = 'WARNING'
        }
        if ($freeDGB -lt 20 -or $freeCGB -lt 10) {
            $state = 'CRITICAL'
        }

        $record = [pscustomobject]@{
            Timestamp = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssK')
            State = $state
            CFreeGB = $freeCGB
            DFreeGB = $freeDGB
            DatasetSizeGB = $datasetSizeGB
            CompletedRuns = ($phaseCounts.Values | Measure-Object -Sum).Sum
        }
        if (-not (Test-Path -LiteralPath $logPath)) {
            $record | Export-Csv -LiteralPath $logPath -NoTypeInformation `
                -Encoding UTF8
        } else {
            $record | Export-Csv -LiteralPath $logPath -NoTypeInformation `
                -Append -Encoding UTF8
        }

        $status = [ordered]@{
            Timestamp = $record.Timestamp
            State = $state
            CFreeGB = $freeCGB
            DFreeGB = $freeDGB
            DatasetSizeGB = $datasetSizeGB
            CompletedRuns = $record.CompletedRuns
            PhaseCounts = $phaseCounts
        }
        $status | ConvertTo-Json -Depth 4 |
            Set-Content -LiteralPath $statusPath -Encoding UTF8

        if ($state -eq 'CRITICAL') {
            Set-Content -LiteralPath $criticalFlagPath -Encoding UTF8 -Value (
                "Critical disk space at $($record.Timestamp): " +
                "C=$freeCGB GB, D=$freeDGB GB")
        } elseif (Test-Path -LiteralPath $criticalFlagPath) {
            Remove-Item -LiteralPath $criticalFlagPath -Force
        }

        Start-Sleep -Seconds $PollSeconds
    }
}
finally {
    if (Test-Path -LiteralPath $pidPath) {
        Remove-Item -LiteralPath $pidPath -Force
    }
}
