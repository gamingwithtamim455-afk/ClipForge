; ClipForge installer — Inno Setup script (Windows).
; Built by the GitHub Actions Windows job after PyInstaller produces
; dist\ClipForge.exe. Requires Inno Setup 6 (choco install innosetup).

#define AppVersion "0.2.2"
#define AppName "ClipForge"
#define ExeName "ClipForge.exe"

[Setup]
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=ClipForge
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=Output
OutputBaseFilename=ClipForge-Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
DisableProgramGroupPage=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\{#ExeName}"; DestDir: "{app}"; Flags: ignoreversion
; Bundled FFmpeg (downloaded by tools/download_ffmpeg.py before the build)
Source: "..\clipforge\bin\ffmpeg.exe"; DestDir: "{app}\bin"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\clipforge\bin\ffprobe.exe"; DestDir: "{app}\bin"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#ExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#ExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#ExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
