[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$RepoRoot = "",
    [string]$OpenCvVersion = "4.13.0",
    [string]$DownloadUrl = "",
    [string]$DestinationDir = "",
    [string]$InstallerPath = "",
    [switch]$ForceDownload,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[opencv-dml] $Message" -ForegroundColor Cyan
}

function Resolve-NormalizedPath {
    param([string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function New-DirectoryIfMissing {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Get-LatestVcDir {
    param([string]$BuildDir)
    $x64Dir = Join-Path $BuildDir "x64"
    if (-not (Test-Path -LiteralPath $x64Dir)) {
        return $null
    }

    return Get-ChildItem -Path $x64Dir -Directory -Filter "vc*" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        Select-Object -First 1
}

function Test-OpenCvBuildLayout {
    param([string]$BuildDir)

    $header = Join-Path $BuildDir "include\opencv2\opencv.hpp"
    if (-not (Test-Path -LiteralPath $header)) {
        return $false
    }

    $vcDir = Get-LatestVcDir -BuildDir $BuildDir
    if (-not $vcDir) {
        return $false
    }

    $libPath = Join-Path $vcDir.FullName "lib\opencv_world4130.lib"
    $dllPath = Join-Path $vcDir.FullName "bin\opencv_world4130.dll"
    return (Test-Path -LiteralPath $libPath) -and (Test-Path -LiteralPath $dllPath)
}

function Find-OpenCvBuildDir {
    param([string]$Root)

    $direct = Join-Path $Root "opencv\build"
    if (Test-OpenCvBuildLayout -BuildDir $direct) {
        return $direct
    }

    $candidates = Get-ChildItem -Path $Root -Recurse -Directory -Filter "build" -ErrorAction SilentlyContinue
    foreach ($candidate in $candidates) {
        if (Test-OpenCvBuildLayout -BuildDir $candidate.FullName) {
            return $candidate.FullName
        }
    }

    return $null
}

function Wait-OpenCvBuildDir {
    param(
        [string]$Root,
        [int]$TimeoutSeconds = 180,
        [int]$PollSeconds = 2
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $buildDir = Find-OpenCvBuildDir -Root $Root
        if ($buildDir) {
            return $buildDir
        }
        Start-Sleep -Seconds $PollSeconds
    } while ((Get-Date) -lt $deadline)

    return $null
}

function Invoke-Download {
    param(
        [string]$Uri,
        [string]$OutFile
    )

    Write-Step "Downloading OpenCV package: $Uri"
    if ($DryRun) {
        Write-Host ">> Invoke-WebRequest -Uri `"$Uri`" -OutFile `"$OutFile`" -MaximumRedirection 10"
        return
    }

    Invoke-WebRequest -Uri $Uri -OutFile $OutFile -MaximumRedirection 10
}

function Test-ValidInstallerFile {
    param(
        [string]$Path,
        [int64]$MinSizeBytes = 50MB
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }

    $fileInfo = Get-Item -LiteralPath $Path
    if ($fileInfo.Length -lt $MinSizeBytes) {
        return $false
    }

    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $header = New-Object byte[] 2
        $read = $stream.Read($header, 0, 2)
    }
    finally {
        $stream.Dispose()
    }
    if ($read -lt 2) {
        return $false
    }

    return ($header[0] -eq 0x4D) -and ($header[1] -eq 0x5A) # "MZ"
}

function Invoke-DownloadWithFallback {
    param(
        [string[]]$Uris,
        [string]$OutFile
    )

    foreach ($uri in $Uris) {
        try {
            Invoke-Download -Uri $uri -OutFile $OutFile
            if ($DryRun) {
                return $true
            }
            if (Test-ValidInstallerFile -Path $OutFile) {
                return $true
            }

            if (Test-Path -LiteralPath $OutFile) {
                Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue
            }
            Write-Warning "[opencv-dml] Downloaded file is not a valid OpenCV installer, trying next URL."
        }
        catch {
            if (Test-Path -LiteralPath $OutFile) {
                Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue
            }
            Write-Warning "[opencv-dml] Download failed from '$uri': $($_.Exception.Message)"
        }
    }

    return $false
}

function Invoke-InstallerExtraction {
    param(
        [string]$ExePath,
        [string]$ExtractRoot
    )

    $methods = @(
        @("-o$ExtractRoot", "-y"),
        @("/S", "/D=$ExtractRoot")
    )

    foreach ($extractArgs in $methods) {
        $printable = $ExePath + " " + (($extractArgs | ForEach-Object {
                    if ($_ -match "\s") { '"' + $_ + '"' } else { $_ }
                }) -join " ")
        Write-Step "Trying installer extraction: $printable"

        if ($DryRun) {
            continue
        }

        $process = Start-Process -FilePath $ExePath -ArgumentList $extractArgs -PassThru -Wait
        $exitCode = 0
        if ($null -ne $process) {
            $exitCode = $process.ExitCode
        }
        elseif (Test-Path Variable:global:LASTEXITCODE) {
            $exitCode = $global:LASTEXITCODE
        }
        if ($exitCode -ne 0) {
            continue
        }

        $buildDir = Wait-OpenCvBuildDir -Root $ExtractRoot
        if ($buildDir) {
            return $buildDir
        }
    }

    return $null
}

try {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
        $RepoRoot = Resolve-NormalizedPath (Join-Path $scriptDir "..")
    }
    else {
        $RepoRoot = Resolve-NormalizedPath $RepoRoot
    }

    if ([string]::IsNullOrWhiteSpace($DestinationDir)) {
        $DestinationDir = Join-Path $RepoRoot "sunone_aimbot_cpp\modules\opencv\prebuilt"
    }
    $DestinationDir = Resolve-NormalizedPath $DestinationDir

    $defaultGithubUrl = "https://github.com/opencv/opencv/releases/download/$OpenCvVersion/opencv-$OpenCvVersion-windows.exe"
    $defaultSourceForgeUrl = "https://sourceforge.net/projects/opencvlibrary/files/$OpenCvVersion/opencv-$OpenCvVersion-windows.exe/download"
    $downloadUrls = @()
    if (-not [string]::IsNullOrWhiteSpace($DownloadUrl)) {
        $downloadUrls += $DownloadUrl
    }
    $downloadUrls += $defaultGithubUrl
    $downloadUrls += $defaultSourceForgeUrl
    $downloadUrls = $downloadUrls | Select-Object -Unique

    if ([string]::IsNullOrWhiteSpace($InstallerPath)) {
        $InstallerPath = Join-Path $DestinationDir ("opencv-" + $OpenCvVersion + "-windows.exe")
    }
    $InstallerPath = Resolve-NormalizedPath $InstallerPath

    $stableOpenCvRoot = Join-Path $DestinationDir "opencv"
    $stableBuildDir = Join-Path $stableOpenCvRoot "build"
    $extractRoot = Join-Path $DestinationDir "_extract"

    if ((-not $ForceDownload) -and (Test-OpenCvBuildLayout -BuildDir $stableBuildDir)) {
        Write-Step "Prebuilt OpenCV is already prepared: $stableBuildDir"
        $vcDir = Get-LatestVcDir -BuildDir $stableBuildDir
        $includeDir = Join-Path $stableBuildDir "include"
        $libPath = Join-Path $vcDir.FullName "lib\opencv_world4130.lib"
        $dllPath = Join-Path $vcDir.FullName "bin\opencv_world4130.dll"
        Write-Host "AIMBOT_OPENCV_INCLUDE_DIR=$includeDir"
        Write-Host "AIMBOT_OPENCV_LIBRARY=$libPath"
        Write-Host "AIMBOT_OPENCV_DLL=$dllPath"
        exit 0
    }

    New-DirectoryIfMissing $DestinationDir

    if ($ForceDownload -and (Test-Path -LiteralPath $InstallerPath)) {
        Write-Step "Removing existing installer: $InstallerPath"
        if (-not $DryRun) {
            Remove-Item -LiteralPath $InstallerPath -Force
        }
    }

    $hasInstaller = Test-Path -LiteralPath $InstallerPath
    $hasValidInstaller = $hasInstaller -and (Test-ValidInstallerFile -Path $InstallerPath)
    if ((-not $ForceDownload) -and $hasInstaller -and (-not $hasValidInstaller)) {
        if ($DryRun) {
            Write-Step "Cached installer is invalid, would re-download."
        }
        else {
            Write-Step "Cached installer is invalid, re-downloading."
            Remove-Item -LiteralPath $InstallerPath -Force
            $hasInstaller = $false
            $hasValidInstaller = $false
        }
    }

    if (-not $hasValidInstaller) {
        if (-not (Invoke-DownloadWithFallback -Uris $downloadUrls -OutFile $InstallerPath)) {
            throw "Could not download a valid OpenCV installer. Tried URLs: $($downloadUrls -join '; ')"
        }
    }
    else {
        Write-Step "Using cached installer: $InstallerPath"
    }

    if ($DryRun) {
        Write-Step "Dry-run mode: skipping extraction."
        Write-Host "InstallerPath=$InstallerPath"
        Write-Host "TargetBuildDir=$stableBuildDir"
        Write-Host "Configure DML with:"
        Write-Host "  cmake -S . -B build/dml -G `"Visual Studio 18 2026`" -A x64 -DAIMBOT_USE_CUDA=OFF"
        exit 0
    }

    if (-not $DryRun -and -not (Test-ValidInstallerFile -Path $InstallerPath)) {
        throw "Downloaded installer is not valid: $InstallerPath"
    }

    if (Test-Path -LiteralPath $extractRoot) {
        Write-Step "Cleaning extraction directory: $extractRoot"
        if (-not $DryRun) {
            Remove-Item -LiteralPath $extractRoot -Recurse -Force
        }
    }
    New-DirectoryIfMissing $extractRoot

    $detectedBuildDir = Invoke-InstallerExtraction -ExePath $InstallerPath -ExtractRoot $extractRoot
    if (-not $detectedBuildDir) {
        throw "Could not extract OpenCV package automatically. Extract manually and place OpenCV as: $stableBuildDir"
    }

    $detectedOpenCvRoot = Split-Path -Parent $detectedBuildDir
    if (Test-Path -LiteralPath $stableOpenCvRoot) {
        Write-Step "Replacing existing prebuilt OpenCV directory: $stableOpenCvRoot"
        if (-not $DryRun) {
            Remove-Item -LiteralPath $stableOpenCvRoot -Recurse -Force
        }
    }

    Write-Step "Installing extracted OpenCV to: $stableOpenCvRoot"
    if (-not $DryRun) {
        Copy-Item -LiteralPath $detectedOpenCvRoot -Destination $stableOpenCvRoot -Recurse
    }

    if (-not (Test-OpenCvBuildLayout -BuildDir $stableBuildDir)) {
        throw "Installed OpenCV layout is invalid: $stableBuildDir"
    }

    $latestVcDir = Get-LatestVcDir -BuildDir $stableBuildDir
    $includeDirOut = Join-Path $stableBuildDir "include"
    $libPathOut = Join-Path $latestVcDir.FullName "lib\opencv_world4130.lib"
    $dllPathOut = Join-Path $latestVcDir.FullName "bin\opencv_world4130.dll"

    Write-Step "Done."
    Write-Host "AIMBOT_OPENCV_INCLUDE_DIR=$includeDirOut"
    Write-Host "AIMBOT_OPENCV_LIBRARY=$libPathOut"
    Write-Host "AIMBOT_OPENCV_DLL=$dllPathOut"
    Write-Host "Now configure DML build:"
    Write-Host "  cmake -S . -B build/dml -G `"Visual Studio 18 2026`" -A x64 -DAIMBOT_USE_CUDA=OFF"
}
catch {
    Write-Error $_
    exit 1
}
