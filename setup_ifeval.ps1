# Download only instruction_following_eval files (not the entire repo)
# Run: powershell -ExecutionPolicy Bypass -File setup_ifeval.ps1

$ErrorActionPreference = "Stop"

$baseUrl = "https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval"
$destDir = "instruction_following_eval"

# Files needed for IFEval scoring
$files = @(
    "__init__.py",
    "instructions.py",
    "instructions_registry.py",
    "instructions_util.py"
)

Write-Host "Downloading instruction_following_eval files..." -ForegroundColor Cyan

# Create directory
if (Test-Path $destDir) {
    Remove-Item -Recurse -Force $destDir
}
New-Item -ItemType Directory -Path $destDir | Out-Null

# Download each file
foreach ($file in $files) {
    $url = "$baseUrl/$file"
    $dest = "$destDir\$file"
    Write-Host "  Downloading $file..."
    try {
        Invoke-WebRequest -Uri $url -OutFile $dest
    } catch {
        Write-Host "  Warning: Failed to download $file" -ForegroundColor Yellow
    }
}

Write-Host "Done! instruction_following_eval is ready." -ForegroundColor Green
Write-Host "You can now run: python eval_score.py --input <file>" -ForegroundColor Yellow
