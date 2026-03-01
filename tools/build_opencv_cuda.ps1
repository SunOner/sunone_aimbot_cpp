[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$RepoRoot = "",
    [string]$OpenCvVersion = "4.13.0",
    [string]$OpenCvModulesDir = "",
    [string]$OpenCvSourceDir = "",
    [string]$OpenCvContribDir = "",
    [string]$OpenCvBuildDir = "",
    [string]$InstallPrefix = "",
    [string]$Generator = "Visual Studio 18 2026",
    [string]$Architecture = "x64",
    [ValidateSet("Release", "RelWithDebInfo", "MinSizeRel", "Debug")]
    [string]$Configuration = "Release",
    [string]$CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
    [string]$CudnnRoot = "C:\Program Files\NVIDIA\CUDNN\v9.17",
    [string]$CudaArchBin = "",
    [switch]$AutoDetectCudaArch,
    [switch]$DisableCuDNN,
    [switch]$SkipDownload,
    [switch]$ConfigureOnly,
    [switch]$BuildOnly,
    [switch]$NoInstall,
    [ValidateRange(0, 256)]
    [int]$MaxCpuCount = 0,
    [switch]$CleanBuild,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[opencv-cuda] $Message" -ForegroundColor Cyan
}

function Test-Tool {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required tool '$Name' was not found in PATH."
    }
}

function Resolve-NormalizedPath {
    param([string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function ConvertTo-CMakePath {
    param([string]$Path)
    return ($Path -replace "\\", "/")
}

function Invoke-External {
    param(
        [string]$Exe,
        [string[]]$ArgumentList
    )

    $printable = $Exe + " " + (($ArgumentList | ForEach-Object {
                if ($_ -match "\s") { '"' + $_ + '"' } else { $_ }
            }) -join " ")
    Write-Host ">> $printable"

    if ($DryRun) {
        return
    }

    & $Exe @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Exe"
    }
}

function New-DirectoryIfMissing {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Find-FirstExistingFile {
    param([string[]]$Candidates)
    foreach ($candidate in $Candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    return $null
}

function Find-FirstFileRecursive {
    param(
        [string]$Root,
        [string[]]$Patterns
    )

    if (-not (Test-Path -LiteralPath $Root)) {
        return $null
    }

    foreach ($pattern in $Patterns) {
        $found = Get-ChildItem -Path $Root -Recurse -File -Filter $pattern -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) {
            return $found.FullName
        }
    }

    return $null
}

function Test-CMakeCacheContains {
    param(
        [string]$BuildDir,
        [string]$Pattern
    )

    $cachePath = Join-Path $BuildDir "CMakeCache.txt"
    if (-not (Test-Path -LiteralPath $cachePath)) {
        return $false
    }

    return $null -ne (Select-String -Path $cachePath -Pattern $Pattern -ErrorAction SilentlyContinue | Select-Object -First 1)
}

function Get-CudaArchFromNvidiaSmi {
    if (-not (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue)) {
        return $null
    }

    try {
        $raw = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $raw) {
            return $null
        }

        $line = ($raw | Select-Object -First 1).ToString().Trim()
        $line = $line -replace ",.*$", ""
        if ($line -match "^\d+(\.\d+)?$") {
            return $line
        }
    }
    catch {
        return $null
    }

    return $null
}

function Get-AvailableVisualStudioGenerators {
    $helpText = & cmake --help 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $helpText) {
        return @()
    }

    $generators = $helpText | ForEach-Object {
        if ($_ -match "^\s*\*?\s*(Visual Studio \d+ \d{4})\s*=") {
            $Matches[1]
        }
    } | Select-Object -Unique

    if (-not $generators) {
        return @()
    }

    return $generators | Sort-Object { [int](($_ -split "\s+")[2]) } -Descending
}

function Initialize-OpenCvRepo {
    param(
        [string]$TargetDir,
        [string]$RemoteUrl,
        [string]$Branch
    )

    if (Test-Path -LiteralPath $TargetDir) {
        return
    }

    if ($SkipDownload) {
        throw "Directory not found and -SkipDownload is set: $TargetDir"
    }

    Test-Tool "git"
    New-DirectoryIfMissing ([System.IO.Path]::GetDirectoryName($TargetDir))
    Write-Step "Cloning $RemoteUrl ($Branch) -> $TargetDir"
    Invoke-External "git" @(
        "-c", "advice.detachedHead=false",
        "clone",
        "--depth", "1",
        "--branch", $Branch,
        $RemoteUrl,
        $TargetDir
    )
}

try {
    Test-Tool "cmake"

    $requiredGenerator = "Visual Studio 18 2026"
    if ($Generator -ne $requiredGenerator) {
        throw "Only '$requiredGenerator' is supported by this project. Remove -Generator override or set -Generator `"$requiredGenerator`"."
    }

    $availableVsGenerators = Get-AvailableVisualStudioGenerators
    if ($availableVsGenerators.Count -eq 0) {
        throw "CMake does not report any Visual Studio generators. Install Visual Studio 2026 with C++ tools and update CMake."
    }
    if ($availableVsGenerators -notcontains $requiredGenerator) {
        $availableText = ($availableVsGenerators -join ", ")
        throw "Required CMake generator '$requiredGenerator' is not available. Detected generators: $availableText"
    }
    $Generator = $requiredGenerator

    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
        $RepoRoot = Resolve-NormalizedPath (Join-Path $scriptDir "..")
    }
    else {
        $RepoRoot = Resolve-NormalizedPath $RepoRoot
    }

    if ([string]::IsNullOrWhiteSpace($OpenCvModulesDir)) {
        $OpenCvModulesDir = Join-Path $RepoRoot "sunone_aimbot_2\modules\opencv"
    }
    $OpenCvModulesDir = Resolve-NormalizedPath $OpenCvModulesDir

    if ([string]::IsNullOrWhiteSpace($OpenCvSourceDir)) {
        $OpenCvSourceDir = Join-Path $OpenCvModulesDir ("opencv-" + $OpenCvVersion)
    }
    $OpenCvSourceDir = Resolve-NormalizedPath $OpenCvSourceDir

    if ([string]::IsNullOrWhiteSpace($OpenCvContribDir)) {
        $OpenCvContribDir = Join-Path $OpenCvModulesDir ("opencv_contrib-" + $OpenCvVersion)
    }
    $OpenCvContribDir = Resolve-NormalizedPath $OpenCvContribDir

    if ([string]::IsNullOrWhiteSpace($OpenCvBuildDir)) {
        $OpenCvBuildDir = Join-Path $OpenCvModulesDir "build"
    }
    $OpenCvBuildDir = Resolve-NormalizedPath $OpenCvBuildDir

    if ([string]::IsNullOrWhiteSpace($InstallPrefix)) {
        $InstallPrefix = Join-Path $OpenCvBuildDir "install"
    }
    $InstallPrefix = Resolve-NormalizedPath $InstallPrefix

    $CudaPath = Resolve-NormalizedPath $CudaPath
    $CudnnRoot = Resolve-NormalizedPath $CudnnRoot

    $runConfigure = $true
    $runBuild = $true
    if ($ConfigureOnly) {
        $runBuild = $false
    }
    if ($BuildOnly) {
        $runConfigure = $false
    }
    $runInstall = $runBuild -and (-not $NoInstall)
    $msbuildArgs = @("/nodeReuse:false")
    if ($MaxCpuCount -gt 0) {
        $msbuildArgs += "/m:$MaxCpuCount"
    }
    else {
        $msbuildArgs += "/m"
    }

    Write-Step "Repo root: $RepoRoot"
    Write-Step "OpenCV source: $OpenCvSourceDir"
    Write-Step "OpenCV contrib: $OpenCvContribDir"
    Write-Step "OpenCV build: $OpenCvBuildDir"
    Write-Step "Install prefix: $InstallPrefix"
    Write-Step "CMake generator: $Generator"
    Write-Step "MSBuild args: $($msbuildArgs -join ' ')"

    New-DirectoryIfMissing $OpenCvModulesDir
    Initialize-OpenCvRepo -TargetDir $OpenCvSourceDir -RemoteUrl "https://github.com/opencv/opencv.git" -Branch $OpenCvVersion
    Initialize-OpenCvRepo -TargetDir $OpenCvContribDir -RemoteUrl "https://github.com/opencv/opencv_contrib.git" -Branch $OpenCvVersion

    $nvccPath = Join-Path $CudaPath "bin\nvcc.exe"
    if (-not (Test-Path -LiteralPath $nvccPath)) {
        throw "CUDA compiler was not found: $nvccPath"
    }

    if ([string]::IsNullOrWhiteSpace($CudaArchBin) -and $AutoDetectCudaArch) {
        $CudaArchBin = Get-CudaArchFromNvidiaSmi
        if ($CudaArchBin) {
            Write-Step "Detected CUDA_ARCH_BIN from nvidia-smi: $CudaArchBin"
        }
    }
    if (-not [string]::IsNullOrWhiteSpace($CudaArchBin) -and $CudaArchBin.Trim().ToLowerInvariant() -eq "all") {
        # Popular consumer NVIDIA architectures: GTX 16/RTX 20/30/40/50 series.
        $CudaArchBin = "7.5;8.6;8.9;12.0"
        Write-Step "Using CUDA_ARCH_BIN preset 'all': $CudaArchBin"
    }
    if ([string]::IsNullOrWhiteSpace($CudaArchBin)) {
        $CudaArchBin = "8.6"
        Write-Warning "[opencv-cuda] CUDA_ARCH_BIN is not set; using default value '$CudaArchBin'. Use -CudaArchBin to override."
    }

    $cudaVersion = [System.IO.Path]::GetFileName($CudaPath.TrimEnd("\", "/")) -replace "^[vV]", ""

    $cudnnIncludeDir = ""
    $cudnnLibPath = ""
    if (-not $DisableCuDNN) {
        $cudnnHeader = Find-FirstExistingFile @(
            (Join-Path $CudnnRoot ("include\" + $cudaVersion + "\cudnn.h")),
            (Join-Path $CudnnRoot "include\13.1\cudnn.h"),
            (Join-Path $CudnnRoot "include\12.9\cudnn.h"),
            (Join-Path $CudnnRoot "include\cudnn.h")
        )
        if (-not $cudnnHeader) {
            $cudnnHeader = Find-FirstFileRecursive -Root (Join-Path $CudnnRoot "include") -Patterns @("cudnn.h")
        }
        if (-not $cudnnHeader) {
            throw "cuDNN header was not found under '$CudnnRoot'. Use -CudnnRoot or -DisableCuDNN."
        }
        $cudnnIncludeDir = Split-Path -Parent $cudnnHeader

        $cudnnLibPath = Find-FirstExistingFile @(
            (Join-Path $CudnnRoot ("lib\" + $cudaVersion + "\x64\cudnn.lib")),
            (Join-Path $CudnnRoot ("lib\" + $cudaVersion + "\cudnn.lib")),
            (Join-Path $CudnnRoot ("lib\" + $cudaVersion + "\x64\cudnn64_9.lib")),
            (Join-Path $CudnnRoot "lib\13.1\cudnn.lib"),
            (Join-Path $CudnnRoot "lib\13.1\x64\cudnn.lib"),
            (Join-Path $CudnnRoot "lib\13.1\x64\cudnn64_9.lib"),
            (Join-Path $CudnnRoot "lib\12.9\x64\cudnn.lib"),
            (Join-Path $CudnnRoot "lib\12.9\x64\cudnn64_9.lib"),
            (Join-Path $CudnnRoot "lib\x64\cudnn.lib")
        )
        if (-not $cudnnLibPath) {
            $cudnnLibPath = Find-FirstFileRecursive -Root (Join-Path $CudnnRoot "lib") -Patterns @("cudnn.lib", "cudnn64_*.lib")
        }
        if (-not $cudnnLibPath) {
            throw "cuDNN library was not found under '$CudnnRoot'. Use -CudnnRoot or -DisableCuDNN."
        }

        Write-Step "cuDNN include: $cudnnIncludeDir"
        Write-Step "cuDNN library: $cudnnLibPath"
    }

    if ($CleanBuild -and (Test-Path -LiteralPath $OpenCvBuildDir)) {
        Write-Step "Cleaning build directory: $OpenCvBuildDir"
        if ($DryRun) {
            Write-Host ">> Remove-Item -LiteralPath `"$OpenCvBuildDir`" -Recurse -Force"
        }
        else {
            Remove-Item -LiteralPath $OpenCvBuildDir -Recurse -Force
        }
    }

    New-DirectoryIfMissing $OpenCvBuildDir

    $cmakeConfigureArgs = @(
        "-S", (ConvertTo-CMakePath $OpenCvSourceDir),
        "-B", (ConvertTo-CMakePath $OpenCvBuildDir),
        "-G", $Generator,
        "-A", $Architecture,
        "-DCMAKE_INSTALL_PREFIX=$(ConvertTo-CMakePath $InstallPrefix)",
        "-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler",
        "-DCUDA_NVCC_FLAGS=--allow-unsupported-compiler",
        "-DOPENCV_EXTRA_MODULES_PATH=$(ConvertTo-CMakePath (Join-Path $OpenCvContribDir 'modules'))",
        "-DWITH_CUDA=ON",
        "-DWITH_CUBLAS=ON",
        "-DENABLE_FAST_MATH=ON",
        "-DCUDA_FAST_MATH=ON",
        "-DOPENCV_DNN_CUDA=ON",
        "-DBUILD_opencv_world=ON",
        "-DWITH_NVCUVENC=OFF",
        "-DWITH_NVCUVID=OFF",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_PERF_TESTS=OFF",
        "-DBUILD_EXAMPLES=OFF",
        "-DCUDA_ARCH_BIN=$CudaArchBin"
    )

    if ($DisableCuDNN) {
        $cmakeConfigureArgs += "-DWITH_CUDNN=OFF"
    }
    else {
        $cmakeConfigureArgs += @(
            "-DWITH_CUDNN=ON",
            "-DCUDNN_INCLUDE_DIR=$(ConvertTo-CMakePath $cudnnIncludeDir)",
            "-DCUDNN_LIBRARY=$(ConvertTo-CMakePath $cudnnLibPath)"
        )
    }

    if ($runConfigure) {
        Write-Step "Configuring OpenCV..."
        Invoke-External "cmake" $cmakeConfigureArgs
    }

    if ($runBuild -and -not $DryRun) {
        $hasNvccBypass = Test-CMakeCacheContains -BuildDir $OpenCvBuildDir -Pattern "^CUDA_NVCC_FLAGS:.*allow-unsupported-compiler"
        if (-not $hasNvccBypass) {
            throw "CMake cache does not contain CUDA_NVCC_FLAGS with --allow-unsupported-compiler. Re-run configure (use script without -BuildOnly, or with -ConfigureOnly first)."
        }
    }

    if ($runBuild) {
        Write-Step "Building ALL_BUILD ($Configuration)..."
        $cmakeBuildAllArgs = @(
            "--build", (ConvertTo-CMakePath $OpenCvBuildDir),
            "--config", $Configuration,
            "--target", "ALL_BUILD",
            "--"
        )
        $cmakeBuildAllArgs += $msbuildArgs
        Invoke-External "cmake" $cmakeBuildAllArgs
    }

    if ($runInstall) {
        Write-Step "Building INSTALL ($Configuration)..."
        $cmakeBuildInstallArgs = @(
            "--build", (ConvertTo-CMakePath $OpenCvBuildDir),
            "--config", $Configuration,
            "--target", "INSTALL",
            "--"
        )
        $cmakeBuildInstallArgs += $msbuildArgs
        Invoke-External "cmake" $cmakeBuildInstallArgs
    }

    if (-not $DryRun) {
        $installedLibDir = Join-Path $InstallPrefix "x64\vc18\lib"
        if (-not (Test-Path -LiteralPath $installedLibDir)) {
            $x64InstallDir = Join-Path $InstallPrefix "x64"
            if (Test-Path -LiteralPath $x64InstallDir) {
                $vcDir = Get-ChildItem -Path $x64InstallDir -Directory -Filter "vc*" -ErrorAction SilentlyContinue |
                    Sort-Object Name -Descending |
                    Select-Object -First 1
                if ($vcDir) {
                    $installedLibDir = Join-Path $vcDir.FullName "lib"
                }
            }
        }
        if (Test-Path -LiteralPath $installedLibDir) {
            $opencvWorld = Get-ChildItem -Path $installedLibDir -Filter "opencv_world*.lib" -ErrorAction SilentlyContinue
            if ($opencvWorld) {
                Write-Step "Done. Installed: $($opencvWorld[0].FullName)"
            }
            else {
                Write-Warning "[opencv-cuda] Build completed, but opencv_world*.lib was not found in $installedLibDir"
            }
        }
    }
}
catch {
    Write-Error $_
    exit 1
}
