$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
$python = Join-Path $repo '.venv/Scripts/python.exe'
$logs = Join-Path $repo '.run'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '4'

foreach ($port in @(8501)) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $port is already in use. No existing process was stopped."
    }
}

$ui = Start-Process -FilePath $python `
    -ArgumentList @('-m', 'streamlit', 'run', 'app.py', '--server.address=127.0.0.1', '--server.port=8501', '--server.headless=true', '--server.runOnSave=true', '--server.fileWatcherType=poll', '--browser.gatherUsageStats=false') `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logs 'ui.stdout.log') -RedirectStandardError (Join-Path $logs 'ui.stderr.log')
@{ui_pid=$ui.Id; dashboard='http://127.0.0.1:8501'; entrypoint='app.py'} |
    ConvertTo-Json | Set-Content (Join-Path $logs 'processes.json')
Write-Output 'RiskRadar rainfall and flood analysis starting: http://127.0.0.1:8501'
