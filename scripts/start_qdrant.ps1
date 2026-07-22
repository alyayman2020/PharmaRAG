# Start the native Qdrant server (single exe, no Docker).
#
# Storage lives in data/qdrant_server - separate from data/qdrant, which is the
# old embedded-mode store. The server memory-maps its storage, so startup is
# instant and RAM stays flat regardless of collection size.
#
# Usage:  powershell -File scripts\start_qdrant.ps1
# Stop:   Stop-Process -Name qdrant

$root = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $root ".qdrant\qdrant.exe"
$storage = Join-Path $root "data\qdrant_server"

if (-not (Test-Path $exe)) {
    Write-Error "qdrant.exe not found at $exe - download the Windows asset from https://github.com/qdrant/qdrant/releases into .qdrant\"
    exit 1
}

try {
    $health = Invoke-RestMethod "http://localhost:6333/healthz" -TimeoutSec 2
    Write-Host "Qdrant already running on :6333 ($health)"
    exit 0
} catch {}

New-Item -ItemType Directory -Force $storage | Out-Null

$env:QDRANT__STORAGE__STORAGE_PATH = $storage
$env:QDRANT__TELEMETRY_DISABLED = "true"

Start-Process -FilePath $exe -WorkingDirectory $root -WindowStyle Hidden
Write-Host "Starting Qdrant..."
foreach ($i in 1..240) {
    Start-Sleep -Milliseconds 500
    try {
        Invoke-RestMethod "http://localhost:6333/healthz" -TimeoutSec 2 | Out-Null
        Write-Host "Qdrant up on http://localhost:6333 (storage: $storage)"
        exit 0
    } catch {}
}
Write-Error "Qdrant did not become healthy within 120s"
exit 1
