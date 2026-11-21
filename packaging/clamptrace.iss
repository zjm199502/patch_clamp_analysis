#define AppVersion Trim(FileRead(FileOpen(SourcePath + "\..\base_version.txt")))
[Setup]
AppId=org.patchclamp.analysis
SetupMutex=ClampTraceInstaller
AppName=ClampTrace
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\ClampTrace
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..
OutputBaseFilename=ClampTrace-Windows-Setup
SetupIconFile=clamptrace.ico
UninstallDisplayIcon={code:RuntimeDir}\PatchClampAnalysis.exe
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
[Files]
Source: "..\dist\PatchClampAnalysis\*"; DestDir: "{code:RuntimeDir}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: unchecked
[Icons]
Name: "{userprograms}\ClampTrace"; Filename: "{code:RuntimeDir}\PatchClampAnalysis.exe"
Name: "{userdesktop}\ClampTrace"; Filename: "{code:RuntimeDir}\PatchClampAnalysis.exe"; Tasks: desktopicon

[Code]
var
  SelectedRoot, SelectedRuntime: String;

function CreateFileForProbe(const Name: String; Access, Share, Security, Disposition,
  Flags, Template: LongWord): THandle;
  external 'CreateFileW@kernel32.dll stdcall';
function CloseProbe(Handle: THandle): Boolean;
  external 'CloseHandle@kernel32.dll stdcall';

function Replaceable(const Name: String): Boolean;
var
  Handle: THandle;
begin
  { OPEN_EXISTING with write/delete access checks Windows share modes without modifying the file. }
  Handle := CreateFileForProbe(Name, $40010000, 7, 0, 3, 0, 0);
  Result := Handle <> THandle(-1);
  if Result then CloseProbe(Handle)
  else Log('Cannot replace ' + Name + ': ' + SysErrorMessage(DLLGetLastError));
end;

function TreeReplaceable(const Dir: String): Boolean;
var
  Entry: TFindRec;
  Name: String;
begin
  Result := True;
  if not FindFirst(AddBackslash(Dir) + '*', Entry) then Exit;
  try
    repeat
      if (Entry.Name <> '.') and (Entry.Name <> '..') then begin
        Name := AddBackslash(Dir) + Entry.Name;
        if (Entry.Attributes and $400) <> 0 then Result := False
        else if (Entry.Attributes and FILE_ATTRIBUTE_DIRECTORY) <> 0 then
          Result := TreeReplaceable(Name)
        else Result := ((Entry.Attributes and FILE_ATTRIBUTE_READONLY) = 0) and Replaceable(Name);
        if not Result then Exit;
      end;
    until not FindNext(Entry);
  finally
    FindClose(Entry);
  end;
end;

function RuntimeDir(Param: String): String;
var
  Root, Candidate, Exe: String;
  Slot: Integer;
begin
  Root := ExpandConstant('{app}');
  if (SelectedRuntime = '') or (CompareText(Root, SelectedRoot) <> 0) then begin
    SelectedRoot := Root;
    Candidate := Root;
    Slot := 0;
    while True do begin
      Exe := AddBackslash(Candidate) + 'PatchClampAnalysis.exe';
      if not DirExists(Candidate) and not FileExists(Candidate) then Break;
      if ((Slot = 0) or FileExists(Exe)) and DirExists(Candidate) then begin
        if (not FileExists(Exe) or Replaceable(Exe)) and
          TreeReplaceable(AddBackslash(Candidate) + '_internal') then Break;
      end;
      Slot := Slot + 1;
      Candidate := AddBackslash(Root) + 'ClampTrace-runtime-' + IntToStr(Slot);
    end;
    SelectedRuntime := Candidate;
    Log('ClampTrace runtime destination: ' + Candidate);
  end;
  Result := SelectedRuntime;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  if CurPageID = wpReady then SelectedRuntime := '';
  Result := True;
end;
