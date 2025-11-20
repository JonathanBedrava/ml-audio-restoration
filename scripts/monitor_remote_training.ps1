# Monitor Remote Training
# Run this on Windows to watch TensorBoard and sync status

$JETSON_USER = "jonathan"
$JETSON_IP = "jetson1"

Write-Host "=== ML Audio Restoration - Remote Training Monitor ===" -ForegroundColor Green
Write-Host ""

# Function to sync and show status
function Sync-Status {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Syncing from Jetson..." -ForegroundColor Cyan
    
    # Sync checkpoints using scp
    scp -r ${JETSON_USER}@${JETSON_IP}:/ssd/ml-audio-restoration/models/checkpoints/* models/checkpoints/ 2>$null
    
    # Sync outputs
    scp -r ${JETSON_USER}@${JETSON_IP}:/ssd/ml-audio-restoration/outputs/* outputs/ 2>$null
    
    # Sync TensorBoard logs
    scp -r ${JETSON_USER}@${JETSON_IP}:/ssd/ml-audio-restoration/runs/* runs/ 2>$null
    
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Sync complete!" -ForegroundColor Green
}

# Initial sync
Sync-Status

# Start TensorBoard in background (check if already running)
Write-Host ""
$tbProcess = Get-Process tensorboard -ErrorAction SilentlyContinue
if ($tbProcess) {
    Write-Host "TensorBoard already running at http://localhost:6006" -ForegroundColor Yellow
} else {
    Write-Host "Starting TensorBoard on http://localhost:6006" -ForegroundColor Yellow
    Start-Process python -ArgumentList "-m tensorboard.main --logdir=runs" -WindowStyle Hidden
}

# Monitor loop
Write-Host ""
Write-Host "Monitoring training... (Press Ctrl+C to stop)"
Write-Host "Syncing every 2 minutes..."
Write-Host ""

while ($true) {
    Start-Sleep -Seconds 120
    Sync-Status
    
    # Show latest checkpoint if it exists
    $latest = Get-ChildItem models/checkpoints -Filter "*.pth" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) {
        Write-Host "Latest checkpoint: $($latest.Name) ($(Get-Date $latest.LastWriteTime -Format 'HH:mm:ss'))" -ForegroundColor Magenta
    }
}
