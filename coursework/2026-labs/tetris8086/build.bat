@echo off
setlocal
cd /d "%~dp0"
where nasm.exe >nul 2>nul
if errorlevel 1 (
  echo NASM was not found. Install from https://www.nasm.us/ and add it to PATH.
  pause
  exit /b 1
)
if not exist build mkdir build
nasm -f bin -Werror -l build\tetris.lst main.asm -o build\TETRIS.COM
if errorlevel 1 (
  pause
  exit /b 1
)
echo Built: build\TETRIS.COM
