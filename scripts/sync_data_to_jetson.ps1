# Sync Training Data to Jetson
# Run this to upload training data and test audio to Jetson

$JETSON_USER = "jonathan"
$JETSON_IP = "jetson1"
$LOCAL_PROJECT = "D:\source\repos\ml-audio-restoration"
$REMOTE_PROJECT = "/ssd/ml-audio-restoration"

Write-Host "=== Syncing Training Data to Jetson ===" -ForegroundColor Green
Write-Host ""
Write-Host "This will take several minutes (310+ audio files)..." -ForegroundColor Yellow
Write-Host ""

# Sync training data
Write-Host "Step 1: Creating training data archive..." -ForegroundColor Cyan
Push-Location "${LOCAL_PROJECT}\data\raw"
$tempDataTar = "$env:TEMP\ml-audio-data.tar.gz"
tar -czf $tempDataTar .

Write-Host "Step 2: Transferring training data to Jetson..." -ForegroundColor Cyan
scp $tempDataTar ${JETSON_USER}@${JETSON_IP}:/tmp/ml-audio-data.tar.gz

Write-Host "Step 3: Extracting on Jetson..." -ForegroundColor Cyan
ssh ${JETSON_USER}@${JETSON_IP} "mkdir -p ${REMOTE_PROJECT}/data/raw && cd ${REMOTE_PROJECT}/data/raw && tar -xzf /tmp/ml-audio-data.tar.gz && rm /tmp/ml-audio-data.tar.gz"

Remove-Item $tempDataTar -ErrorAction SilentlyContinue
Pop-Location

# Sync test audio
Write-Host ""
Write-Host "Step 4: Creating test audio archive..." -ForegroundColor Cyan
Push-Location "G:\raw\opera"
$tempTestTar = "$env:TEMP\ml-audio-test.tar.gz"
tar -czf $tempTestTar .

Write-Host "Step 5: Transferring test audio to Jetson..." -ForegroundColor Cyan
scp $tempTestTar ${JETSON_USER}@${JETSON_IP}:/tmp/ml-audio-test.tar.gz

Write-Host "Step 6: Extracting on Jetson..." -ForegroundColor Cyan
ssh ${JETSON_USER}@${JETSON_IP} "mkdir -p ${REMOTE_PROJECT}/test_audio/opera && cd ${REMOTE_PROJECT}/test_audio/opera && tar -xzf /tmp/ml-audio-test.tar.gz && rm /tmp/ml-audio-test.tar.gz"

Remove-Item $tempTestTar -ErrorAction SilentlyContinue
Pop-Location

# Verify
Write-Host ""
Write-Host "Verifying data sync..." -ForegroundColor Cyan
$fileCount = ssh ${JETSON_USER}@${JETSON_IP} "find ${REMOTE_PROJECT}/data/raw -type f | wc -l"
$testCount = ssh ${JETSON_USER}@${JETSON_IP} "find ${REMOTE_PROJECT}/test_audio/opera -type f | wc -l"

Write-Host ""
Write-Host "=== Data Sync Complete! ===" -ForegroundColor Green
Write-Host "Training files: $fileCount" -ForegroundColor Cyan
Write-Host "Test files: $testCount" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now start training on the Jetson!" -ForegroundColor Yellow
