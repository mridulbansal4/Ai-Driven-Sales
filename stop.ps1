Write-Host "Stopping agent..." -ForegroundColor Yellow
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'livekit_sales_agent\.py|agent\.py' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Write-Host "Done." -ForegroundColor Green
