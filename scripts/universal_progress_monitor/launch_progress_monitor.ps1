param(
    [int]$Port = 8765,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$scriptDirectory = $PSScriptRoot
$workspaceRoot = Split-Path -Parent $scriptDirectory
$dashboardUrl = "http://127.0.0.1:$Port"

$pythonCandidates = @(
    (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe')
)
$python = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $python) {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $python = $pythonCommand.Source
    }
}
if (-not $python) {
    throw 'No Windows Python runtime was found for the progress monitor.'
}

try {
    $existing = Invoke-WebRequest -UseBasicParsing "$dashboardUrl/api/status" -TimeoutSec 4
    if ($existing.StatusCode -eq 200) {
        $null = $existing.Content | ConvertFrom-Json
        Write-Host "Progress Monitor is already running: $dashboardUrl"
        if (-not $NoBrowser) {
            Start-Process $dashboardUrl
        }
        exit 0
    }
} catch {
    # Do not kill an unknown listener automatically. An older broken dashboard
    # and an unrelated Python service are indistinguishable without elevated
    # process-command inspection.
    $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($listener) {
        $listenerProcess = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
        $processName = if ($listenerProcess) { $listenerProcess.ProcessName } else { 'unknown process' }
        throw "Port $Port is already occupied by $processName (PID $($listener.OwningProcess)). Close the old monitor window or start this monitor with -Port $($Port + 1)."
    }
}

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($url)
        Start-Sleep -Seconds 2
        Start-Process $url
    } -ArgumentList $dashboardUrl | Out-Null
}

Write-Host "Progress Monitor: $dashboardUrl"
Write-Host 'Leave this window open. Press Ctrl+C to stop the dashboard.'
& $python (Join-Path $scriptDirectory 'progress_dashboard.py') `
    --root $workspaceRoot `
    --state-dir (Join-Path $workspaceRoot '.progress_tasks') `
    --port $Port
