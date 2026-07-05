param(
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Resolve-Path (Join-Path $scriptDir "..")

if (-not $OutputPath) {
    $OutputPath = Join-Path $root "outputs\colab\insurance_capstone_colab_bundle.zip"
}

$outputFullPath = [System.IO.Path]::GetFullPath($OutputPath)
$outputDir = Split-Path -Parent $outputFullPath
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$tempDir = Join-Path $env:TEMP ("insurance_capstone_colab_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

$includePaths = @(
    "Insurance Data.csv",
    "insurance_modeling.py",
    "run_all.py",
    "generate_html_report.py",
    "app.py",
    "README.md",
    "requirements.txt",
    "scripts\prepare_colab_bundle.ps1",
    "scripts\colab_gpu_run.sh",
    "Milestone 1 Template.docx",
    "Milestone 2 Template.docx",
    "CapstoneProject_PresentationTemplate.pptx",
    "Problem statement Insurance Price Prediction.pdf",
    "Program  Structure.pdf",
    "notebooks\03_colab_gpu_training.ipynb"
)

try {
    foreach ($relativePath in $includePaths) {
        $source = Join-Path $root $relativePath
        if (-not (Test-Path -LiteralPath $source)) {
            Write-Warning "Skipping missing file: $relativePath"
            continue
        }

        $destination = Join-Path $tempDir $relativePath
        $destinationDir = Split-Path -Parent $destination
        New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }

    if (Test-Path -LiteralPath $outputFullPath) {
        Remove-Item -LiteralPath $outputFullPath -Force
    }

    Compress-Archive -Path (Join-Path $tempDir "*") -DestinationPath $outputFullPath -Force
    Write-Output $outputFullPath
}
finally {
    if (Test-Path -LiteralPath $tempDir) {
        Remove-Item -LiteralPath $tempDir -Recurse -Force
    }
}
