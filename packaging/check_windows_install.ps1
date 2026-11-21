# Run on the clean Windows release runner with the production installer.
$ErrorActionPreference = 'Stop'
$dir = Join-Path $env:RUNNER_TEMP 'ClampTrace-install-test'
if (Test-Path -LiteralPath $dir) { throw 'Expected a fresh test installation' }
$exe = Join-Path $dir 'PatchClampAnalysis.exe'
$setup = (Resolve-Path 'ClampTrace-Windows-Setup.exe').Path
function Install-Build {
    $proc = Start-Process -FilePath $setup -ArgumentList @('/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART',('/DIR="'+$dir+'"')) -PassThru -Wait -WindowStyle Hidden
    if ($proc.ExitCode -ne 0) { throw "Setup exited with $($proc.ExitCode)" }
}
function Assert-Shortcut($expected) {
    $lnk = Join-Path ([Environment]::GetFolderPath('Programs')) 'ClampTrace.lnk'
    $shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
    if ($shortcut.TargetPath -ne $expected -or !(Test-Path -LiteralPath $expected)) {
        throw "Unexpected launch target: $($shortcut.TargetPath)"
    }
}
Install-Build
Assert-Shortcut $exe
Install-Build
Assert-Shortcut $exe
$env:QT_QPA_PLATFORM = 'offscreen'
$old = Start-Process -FilePath $exe -PassThru -WindowStyle Hidden
try {
    Start-Sleep -Seconds 8
    if ($old.HasExited) { throw 'Original GUI exited before the upgrade' }
    Install-Build
    if ($old.HasExited) { throw 'Setup closed the running GUI' }
    $spare = Join-Path $dir 'ClampTrace-runtime-1'
    Assert-Shortcut "$spare\PatchClampAnalysis.exe"
    if (!(Test-Path "$spare\_internal\ClampTraceDirectML\ClampTraceDirectML.exe")) { throw 'Incomplete GPU runtime' }
    $new = Start-Process -FilePath "$spare\PatchClampAnalysis.exe" -PassThru -WindowStyle Hidden
    try {
        Start-Sleep -Seconds 8
        if ($new.HasExited) { throw 'Updated GUI failed to start' }
    } finally {
        if (!$new.HasExited) { $new.Kill(); $new.WaitForExit() }
    }
} finally {
    if (!$old.HasExited) { $old.Kill(); $old.WaitForExit() }
}
$marker = Join-Path $dir 'user-recording.txt'
Set-Content -LiteralPath $marker -Value 'user data'
$uninstaller = @(Get-ChildItem -LiteralPath $dir -Filter 'unins*.exe')
if ($uninstaller.Count -ne 1) { throw 'Expected a single uninstall entry' }
$proc = Start-Process $uninstaller[0].FullName -ArgumentList '/VERYSILENT','/NORESTART' -PassThru -Wait -WindowStyle Hidden
if ($proc.ExitCode -ne 0 -or (Test-Path $exe) -or (Test-Path "$spare\PatchClampAnalysis.exe") -or !(Test-Path $marker)) {
    throw 'Uninstall cleanup or user data preservation failed'
}
