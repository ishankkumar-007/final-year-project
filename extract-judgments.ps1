param(
    [Parameter(Mandatory=$true)]
    [int]$StartYear,
    
    [Parameter(Mandatory=$true)]
    [int]$EndYear
)

# Base paths
$dataBasePath = "judgments-data\data\tar"
$metadataBasePath = "judgments-data\metadata\tar"

Write-Host "Extracting tar files for years $StartYear to $EndYear..." -ForegroundColor Green

# Iterate through the year range
for ($year = $StartYear; $year -le $EndYear; $year++) {
    Write-Host "`nProcessing year $year..." -ForegroundColor Cyan
    
    # Process data tars (english only)
    $dataYearPath = Join-Path $dataBasePath "year=$year"
    $englishPath = Join-Path $dataYearPath "english"
    if (Test-Path $englishPath) {
        # Find english.tar file
        $englishTar = Join-Path $englishPath "english.tar"
        
        if (Test-Path $englishTar) {
            $tarFileName = [System.IO.Path]::GetFileNameWithoutExtension($englishTar)
            $extractPath = Join-Path $englishPath $tarFileName
            
            Write-Host "  Extracting: $englishTar" -ForegroundColor Yellow
            Write-Host "  To: $extractPath" -ForegroundColor Yellow
            
            # Remove existing directory if it exists (to ensure clean extraction)
            # Safety check: only remove if it's a directory and not a tar file
            if ((Test-Path $extractPath) -and (Test-Path $extractPath -PathType Container) -and ($extractPath -notlike "*.tar")) {
                Write-Host "  Removing existing directory..." -ForegroundColor Yellow
                Remove-Item -Path $extractPath -Recurse -Force
            }
            
            # Create extraction directory
            New-Item -Path $extractPath -ItemType Directory -Force | Out-Null
            
            # Extract tar file
            try {
                tar -xf $englishTar -C $extractPath
                Write-Host "  ✓ Extracted successfully" -ForegroundColor Green
            }
            catch {
                Write-Host "  ✗ Error extracting: $_" -ForegroundColor Red
            }
        }
        else {
            Write-Host "  English tar not found: $englishTar" -ForegroundColor Gray
        }
    }
    else {
        Write-Host "  English path not found: $englishPath" -ForegroundColor Gray
    }
    
    # Process metadata tar
    $metadataYearPath = Join-Path $metadataBasePath "year=$year"
    if (Test-Path $metadataYearPath) {
        $tarFiles = Get-ChildItem -Path $metadataYearPath -Filter "*.tar"
        
        foreach ($tarFile in $tarFiles) {
            $tarFileName = [System.IO.Path]::GetFileNameWithoutExtension($tarFile.Name)
            $extractPath = Join-Path $tarFile.DirectoryName $tarFileName
            
            Write-Host "  Extracting: $($tarFile.FullName)" -ForegroundColor Yellow
            Write-Host "  To: $extractPath" -ForegroundColor Yellow
            
            # Remove existing directory if it exists (to ensure clean extraction)
            # Safety check: only remove if it's a directory and not a tar file
            if ((Test-Path $extractPath) -and (Test-Path $extractPath -PathType Container) -and ($extractPath -notlike "*.tar")) {
                Write-Host "  Removing existing directory..." -ForegroundColor Yellow
                Remove-Item -Path $extractPath -Recurse -Force
            }
            
            # Create extraction directory
            New-Item -Path $extractPath -ItemType Directory -Force | Out-Null
            
            # Extract tar file
            try {
                tar -xf $tarFile.FullName -C $extractPath
                Write-Host "  ✓ Extracted successfully" -ForegroundColor Green
            }
            catch {
                Write-Host "  ✗ Error extracting: $_" -ForegroundColor Red
            }
        }
    }
    else {
        Write-Host "  Metadata path not found: $metadataYearPath" -ForegroundColor Gray
    }
}

Write-Host "`nExtraction complete!" -ForegroundColor Green
