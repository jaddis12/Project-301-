$ErrorActionPreference = "Stop"

$python = "python"
$app = Join-Path $PSScriptRoot "app.py"

Set-Location $PSScriptRoot
& $python -m streamlit run $app

