param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv-web\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Set up .venv-web using README_WEB.md first.'
}
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'frontend\dist\index.html'))) {
    throw 'Build the frontend using the instructions in README_WEB.md first.'
}
Write-Host "Current is available locally at http://127.0.0.1:$Port. Press Ctrl+C to stop."
& $taskPython -B -m uvicorn web_api:app --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
