# Starts the existing Streamlit UI with an automatically assigned localhost port.
# Run from anywhere: powershell -NoProfile -File .\scripts\start-ui.ps1

$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

try {
    docker compose up -d --build --force-recreate ui

    if ($LASTEXITCODE -ne 0) {
        throw "Could not start the UI. Run: docker compose logs --tail=80 ui"
    }

    $publishedPort = docker compose port ui 8501

    if ($LASTEXITCODE -ne 0 -or -not $publishedPort) {
        throw "UI started but its host port could not be discovered. Run: docker compose ps"
    }

    $firstLine = @($publishedPort)[0].Trim()
    $portMatch = [regex]::Match($firstLine, ':(\d+)$')

    if (-not $portMatch.Success) {
        throw "Unexpected port mapping from Docker Compose: $firstLine"
    }

    $url = "http://127.0.0.1:$($portMatch.Groups[1].Value)"

    Write-Host ""
    Write-Host "LLM Sampling Lab: $url"
    Write-Host "Host port assigned automatically; other containers were not stopped."
    Write-Host "Troubleshooting: docker compose logs --tail=80 ui"
}
finally {
    Pop-Location
}
