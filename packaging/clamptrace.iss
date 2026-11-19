#define AppVersion Trim(FileRead(FileOpen(SourcePath + "\..\base_version.txt")))
[Setup]
AppId=org.patchclamp.analysis
AppName=ClampTrace
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\ClampTrace
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..
OutputBaseFilename=ClampTrace-Windows-Setup
SetupIconFile=clamptrace.ico
UninstallDisplayIcon={app}\PatchClampAnalysis.exe
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
[Files]
Source: "..\dist\PatchClampAnalysis\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: unchecked
[Icons]
Name: "{userprograms}\ClampTrace"; Filename: "{app}\PatchClampAnalysis.exe"
Name: "{userdesktop}\ClampTrace"; Filename: "{app}\PatchClampAnalysis.exe"; Tasks: desktopicon
