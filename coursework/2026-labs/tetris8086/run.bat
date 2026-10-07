@echo off
setlocal
cd /d "%~dp0"
where nasm.exe >nul 2>nul
if not errorlevel 1 (
  call build.bat
  if errorlevel 1 exit /b 1
)
if not exist build\TETRIS.COM (
  echo Missing build\TETRIS.COM. Run build.bat first.
  pause
  exit /b 1
)
set "TETRIS_EMULATOR="
for /f "delims=" %%I in ('where dosbox-x.exe 2^>nul') do if not defined TETRIS_EMULATOR set "TETRIS_EMULATOR=%%I"
if not defined TETRIS_EMULATOR if exist "%ProgramFiles%\DOSBox-X\dosbox-x.exe" set "TETRIS_EMULATOR=%ProgramFiles%\DOSBox-X\dosbox-x.exe"
if not defined TETRIS_EMULATOR if exist "%ProgramFiles(x86)%\DOSBox-X\dosbox-x.exe" set "TETRIS_EMULATOR=%ProgramFiles(x86)%\DOSBox-X\dosbox-x.exe"
if not defined TETRIS_EMULATOR (
  echo Install DOSBox-X from https://dosbox-x.com/ and add its folder to PATH.
  pause
  exit /b 1
)
"%TETRIS_EMULATOR%" -conf dosbox-x.conf -fastlaunch -c "mount c .\build" -c "c:" -c "TETRIS.COM" -c "exit"
