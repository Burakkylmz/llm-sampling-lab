# Start the Streamlit UI on an automatically assigned localhost port.
# Usage from the repository root: .\scripts\start-ui.ps1

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

try {
    $entrypoint = Join-Path $projectRoot "streamlit_app.py"

    if (-not (Test-Path -LiteralPath $entrypoint -PathType Leaf)) {
        throw "streamlit_app.py was not found in $projectRoot. Open the project folder containing the Streamlit app."
    }

    docker compose config --quiet

    if ($LASTEXITCODE -ne 0) {
        throw "Invalid Docker Compose configuration. Run: docker compose config"
    }

    docker compose up -d --build --force-recreate ui

    if ($LASTEXITCODE -ne 0) {
        throw "Docker failed to start the UI. Run: docker compose logs --tail=80 ui"
    }

    $publishedPort = @(docker compose port ui 8501)

    if ($LASTEXITCODE -ne 0 -or $publishedPort.Count -eq 0) {
        throw "UI started but the host port could not be determined. Run: docker compose ps"
    }

    $firstLine = $publishedPort[0].Trim()
    $portMatch = [regex]::Match($firstLine, ':(\d+)$')

    if (-not $portMatch.Success) {
        throw "Unexpected Docker port mapping: $firstLine"
    }

    $url = "http://127.0.0.1:$($portMatch.Groups[1].Value)"
    $healthUrl = "$url/_stcore/health"
    $healthy = $false

    for ($attempt = 1; $attempt -le 30; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop

            if ($response.StatusCode -eq 200) {
                $healthy = $true
                break
            }
        }
        catch {
            # Docker can publish the port before Streamlit finishes startup.
        }

        Start-Sleep -Seconds 1
    }

    if (-not $healthy) {
        Write-Host ""
        Write-Warning "The UI container was created, but Streamlit is not reachable."
        docker compose ps -a
        docker compose logs --tail=60 ui
        throw "Streamlit health check failed. Inspect the Docker logs above."
    }

    Write-Host ""
    Write-Host "LLM Sampling Lab is ready: $url"
    Write-Host "Docker selected the available localhost port automatically."
}
finally {
    Pop-Location
}
