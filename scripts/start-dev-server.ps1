param(
    [switch]$Restart
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Logs = Join-Path $Root "logs"
$OutLog = Join-Path $Logs "server.out.log"
$ErrLog = Join-Path $Logs "server.err.log"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtualenv Python not found at $Python. Run dependency setup first."
}

New-Item -ItemType Directory -Force -Path $Logs | Out-Null

$listeners = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue |
    Where-Object { $_.State -eq "Listen" }

if ($listeners -and -not $Restart) {
    $listeners | Select-Object LocalAddress, LocalPort, State, OwningProcess
    Write-Host "Server already appears to be listening on port 8000."
    exit 0
}

if ($listeners -and $Restart) {
    $listeners | ForEach-Object {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

Remove-Item -LiteralPath $OutLog, $ErrLog -ErrorAction SilentlyContinue

$proc = Start-Process `
    -FilePath $Python `
    -ArgumentList @("run.py", "--server") `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru

Start-Sleep -Seconds 5

$active = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue |
    Where-Object { $_.State -eq "Listen" }

if (-not $active) {
    Write-Host "Server did not start. Last stderr lines:"
    if (Test-Path -LiteralPath $ErrLog) {
        Get-Content -LiteralPath $ErrLog -Tail 80
    }
    exit 1
}

Write-Host "Dev server running on http://localhost:8000"
Write-Host "PID: $($proc.Id)"
Write-Host "Logs: $ErrLog"

