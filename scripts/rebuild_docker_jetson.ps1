# Force rebuild Docker container on Jetson without cache
param(
    [string]$JetsonHost = "jetson1",
    [string]$JetsonUser = "jonathan"
)

Write-Host "=== Force Rebuilding Docker Container on Jetson ===" -ForegroundColor Cyan
Write-Host ""

# First sync the latest code
Write-Host "Syncing latest code..." -ForegroundColor Yellow
& "$PSScriptRoot\setup_remote_jetson.ps1" -CodeOnly

if ($LASTEXITCODE -ne 0) {
    Write-Host "Code sync failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Building Docker container (no cache)..." -ForegroundColor Yellow

# Build without cache
$buildCmd = @"
cd /ssd/ml-audio-restoration && \
sudo docker build --no-cache --network=host -f Dockerfile.jetson -t ml-audio-restore:latest .
"@

ssh "${JetsonUser}@${JetsonHost}" $buildCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== Container rebuilt successfully! ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "To run the container:" -ForegroundColor Cyan
    Write-Host "ssh ${JetsonUser}@${JetsonHost}" -ForegroundColor White
    Write-Host "sudo docker run --gpus all --network=host ``" -ForegroundColor White
    Write-Host "  -v /ssd/ml-audio-restoration/data:/workspace/ml-audio-restoration/data ``" -ForegroundColor White
    Write-Host "  -v /ssd/ml-audio-restoration/models:/workspace/ml-audio-restoration/models ``" -ForegroundColor White
    Write-Host "  -v /ssd/ml-audio-restoration/outputs:/workspace/ml-audio-restoration/outputs ``" -ForegroundColor White
    Write-Host "  -v /ssd/ml-audio-restoration/runs:/workspace/ml-audio-restoration/runs ``" -ForegroundColor White
    Write-Host "  -v /ssd/ml-audio-restoration/test_audio:/workspace/ml-audio-restoration/test_audio ``" -ForegroundColor White
    Write-Host "  -it ml-audio-restore:latest" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "=== Build failed! ===" -ForegroundColor Red
    exit 1
}
