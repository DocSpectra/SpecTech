$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    docker build --platform linux/amd64 `
        --tag spectech-ko-specificity:36f8e835-official-release-v1 `
        --file ko_container/Dockerfile `
        .
}
finally {
    Pop-Location
}
