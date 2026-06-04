# Sales trainer - wraps backend agent (same as Agent/ uv run ... console)
$projectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Get-Location }
$backend = Join-Path $projectRoot "backend"

function Set-AgentPath {
    $env:PATH = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
    $nvidia = Join-Path $backend ".venv\Lib\site-packages\nvidia"
    foreach ($sub in @("cudnn\bin", "cublas\bin")) {
        $dir = Join-Path $nvidia $sub
        if (Test-Path $dir) { $env:PATH = "$dir;$env:PATH" }
    }
    foreach ($dir in @("${env:ProgramFiles}\eSpeak NG", "${env:ProgramFiles(x86)}\eSpeak NG")) {
        if (Test-Path $dir) { $env:PATH = "$dir;$env:PATH" }
    }
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
}

Set-AgentPath
Set-Location $backend

try {
    Invoke-RestMethod http://localhost:11434 | Out-Null
    Write-Host "[OK] Ollama running" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Start Ollama first." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Install uv: irm https://astral.sh/uv/install.ps1 | iex" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path ".venv")) { uv sync --python 3.11 }

function Resolve-AudioDeviceId {
    param([string]$Name, [string]$Kind)
    if (-not $Name -or $Name -match '^\d+$') { return $Name }
    $id = uv run --python 3.11 python -m pipeline.audio_devices $Name $Kind 2>&1 | Select-Object -Last 1
    if ($LASTEXITCODE -ne 0) { Write-Host $id -ForegroundColor Red; exit 1 }
    return $id.ToString().Trim()
}

$args = @("livekit_sales_agent.py", "download-files")
uv run --python 3.11 python @args 2>$null | Out-Null

$args = @("livekit_sales_agent.py", "console")
if ($env:INPUT_DEVICE) {
    $id = Resolve-AudioDeviceId $env:INPUT_DEVICE "input"
    Write-Host "Mic: $env:INPUT_DEVICE -> ID $id" -ForegroundColor DarkGray
    $args += @("--input-device", $id)
}
if ($env:OUTPUT_DEVICE) {
    $id = Resolve-AudioDeviceId $env:OUTPUT_DEVICE "output"
    Write-Host "Speaker: $env:OUTPUT_DEVICE -> ID $id" -ForegroundColor DarkGray
    $args += @("--output-device", $id)
}

Write-Host ""
Write-Host "uv run python livekit_sales_agent.py console" -ForegroundColor Green
Write-Host ""

uv run --python 3.11 python @args
