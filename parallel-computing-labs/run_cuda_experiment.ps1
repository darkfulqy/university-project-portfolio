param(
    [string]$Sizes = "512,1024,2048",
    [int]$Runs = 3
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutputDir = Join-Path $Root "output"
$VsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
$CudaSource = Join-Path $Root "matrix_cuda.cu"
$CudaExe = Join-Path $OutputDir "matrix_cuda.exe"
$CudaCsv = Join-Path $Root "matrix_cuda_perf.csv"
$OpenMpSource = Join-Path $Root "juzhenchengfa.cpp"
$OpenMpExe = Join-Path $OutputDir "juzhenchengfa_openmp.exe"
$OpenMpCsv = Join-Path $Root "matrix_openmp_current_perf.csv"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (-not (Test-Path -LiteralPath $VsDevCmd)) {
    throw "VS Build Tools not found: $VsDevCmd"
}

$cudaBuild = "chcp 65001 >nul && `"$VsDevCmd`" -arch=x64 >nul && nvcc -std=c++17 -O2 -arch=sm_86 -diag-suppress=192 -Xcompiler=/utf-8,/wd4129 -o `"$CudaExe`" `"$CudaSource`""
cmd.exe /d /s /c $cudaBuild
if ($LASTEXITCODE -ne 0) {
    throw "CUDA build failed."
}

$gpp = (Get-Command g++ -ErrorAction Stop).Source
& $gpp -std=c++17 -O2 -fopenmp -fdiagnostics-color=always $OpenMpSource -o $OpenMpExe
if ($LASTEXITCODE -ne 0) {
    throw "OpenMP build failed."
}

& $CudaExe --sizes $Sizes --runs $Runs --csv $CudaCsv
if ($LASTEXITCODE -ne 0) {
    throw "CUDA benchmark failed."
}

function Get-MatchValue {
    param(
        [string]$Text,
        [string]$Pattern,
        [string]$Description
    )

    $match = [regex]::Match($Text, $Pattern)
    if (-not $match.Success) {
        throw "Cannot parse $Description from OpenMP output."
    }
    return $match.Groups[1].Value
}

"mode,n,run,threads,chunk,static_time_s,dynamic_time_s,guided_time_s,best_schedule,checksum" | Set-Content -LiteralPath $OpenMpCsv -Encoding UTF8
$sizeList = $Sizes.Split(",") | ForEach-Object { [int]$_.Trim() }
foreach ($n in $sizeList) {
    for ($run = 1; $run -le $Runs; $run++) {
        $inputFile = Join-Path $env:TEMP "matrix_openmp_input_$PID.txt"
        [System.IO.File]::WriteAllText($inputFile, "$n`r`n8`r`n4`r`n", [System.Text.Encoding]::ASCII)
        $runCommand = "`"$OpenMpExe`" < `"$inputFile`""
        $output = cmd.exe /d /s /c $runCommand
        Remove-Item -LiteralPath $inputFile -Force -ErrorAction SilentlyContinue
        if ($LASTEXITCODE -ne 0) {
            throw "OpenMP benchmark failed for n=$n run=$run."
        }

        $text = $output -join "`n"
        $static = Get-MatchValue $text "schedule\(static,4\) time = ([0-9.]+) s" "static time"
        $dynamic = Get-MatchValue $text "schedule\(dynamic,4\) time = ([0-9.]+) s" "dynamic time"
        $guided = Get-MatchValue $text "schedule\(guided,4\) time = ([0-9.]+) s" "guided time"
        $best = Get-MatchValue $text "Best schedule = (\w+)" "best schedule"
        $checksum = Get-MatchValue $text "Result checksum = ([0-9]+)" "checksum"
        "openmp,$n,$run,8,4,$static,$dynamic,$guided,$best,$checksum" | Add-Content -LiteralPath $OpenMpCsv -Encoding UTF8
    }
}

Write-Host "CUDA CSV: $CudaCsv"
Write-Host "OpenMP CSV: $OpenMpCsv"
