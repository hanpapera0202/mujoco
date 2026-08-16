[CmdletBinding()]
param(
    [switch]$SkipExtensions
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Get-BootstrapPython {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        return @($launcher.Source, "-3")
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @($python.Source)
    }

    throw "Python 3 was not found. Install Python 3.11 or newer from python.org, then run this script again."
}

Push-Location $ProjectRoot
try {
    if (-not (Test-Path $VenvPython)) {
        $bootstrap = Get-BootstrapPython
        Write-Host "Creating .venv virtual environment..."
        if ($bootstrap.Length -gt 1) {
            & $bootstrap[0] $bootstrap[1] -m venv .venv
        }
        else {
            & $bootstrap[0] -m venv .venv
        }
    }

    Write-Host "Installing Python packages..."
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r requirements.txt
    & $VenvPython -c "import mujoco; print('MuJoCo Python package:', mujoco.__version__)"

    if (-not $SkipExtensions) {
        $codeCandidates = @(
            (Join-Path $env:LOCALAPPDATA "Programs\Microsoft VS Code\bin\code.cmd"),
            (Join-Path $env:ProgramFiles "Microsoft VS Code\bin\code.cmd")
        )
        $code = $codeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
        if ($code) {
            Write-Host "Installing VS Code Python extensions..."
            & $code --install-extension ms-python.python --force
            & $code --install-extension ms-python.vscode-pylance --force
        }
    }

    Write-Host ""
    Write-Host "Environment ready. Open this folder in VS Code: $ProjectRoot"
    Write-Host "Press F5 and select the dual-arm or single-arm launch profile."
}
finally {
    Pop-Location
}
