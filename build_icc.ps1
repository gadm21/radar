$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $compilerPath = Join-Path $PSScriptRoot 'tmp/latex/tectonic.exe'
    if (-not (Test-Path -LiteralPath $compilerPath)) {
        throw 'Install Tectonic or compile icc_multimodal.tex with a full TeX distribution.'
    }
    & $compilerPath --keep-logs --keep-intermediates icc_multimodal.tex
    if ($LASTEXITCODE -ne 0) { throw "LaTeX compilation failed: $LASTEXITCODE" }
} finally {
    Pop-Location
}
