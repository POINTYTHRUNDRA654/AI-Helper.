echo Dashboard should be open in your browser at http://127.0.0.1:8765
@echo off
setlocal
set BASE=%~dp0..
pushd %BASE%
set EXE=%BASE%\dist\AIHelper.exe

if not exist "%EXE%" (
  echo Could not find AIHelper.exe at %EXE%
  echo Build it first (from project root): .\.venv\Scripts\python.exe -m PyInstaller ai_helper/__main__.py --name AIHelper --onefile --console
  pause
  popd
  exit /b 1
)

echo Starting AI Helper with web dashboard (running in background)...
start "AIHelper" /B "%EXE%" --web-ui --daemon
rem Give it a moment to start, then open the dashboard.
ping -n 2 127.0.0.1 >nul
start "" http://127.0.0.1:8765

echo Dashboard should be open in your browser at http://127.0.0.1:8765
pause
popd
endlocal
