@echo off
setlocal
rem One-click launcher: starts the backend (port 8000) and frontend (port 3000)
rem in their own windows, then opens the app in the browser.
rem Close those two windows to stop the servers.

set "ROOT=%~dp0"
set "BACKEND=%ROOT%backend"
set "FRONTEND=%ROOT%frontend"

title Language Evolution Simulator - launcher

where python >nul 2>nul || (echo [error] Python not found on PATH. & pause & exit /b 1)
where npm >nul 2>nul || (echo [error] Node.js/npm not found on PATH. & pause & exit /b 1)

python -m uv --version >nul 2>nul || (
    echo [setup] Installing uv...
    python -m pip install --user uv || (echo [error] Could not install uv. & pause & exit /b 1)
)

rem Ollama is only needed for local models; warn instead of failing.
curl -s -o nul http://127.0.0.1:11434 || echo [warn] Ollama does not seem to be running. Start it with "ollama serve" if you use a local model.

echo [setup] Syncing backend dependencies...
pushd "%BACKEND%"
python -m uv sync || (popd & echo [error] uv sync failed. & pause & exit /b 1)
popd

if not exist "%FRONTEND%\node_modules" (
    echo [setup] Installing frontend dependencies...
    pushd "%FRONTEND%"
    call npm install || (popd & echo [error] npm install failed. & pause & exit /b 1)
    popd
)

echo [start] Backend on http://127.0.0.1:8000
start "LangEvo backend" /D "%BACKEND%" cmd /k python -m uv run uvicorn app.main:app --reload --reload-dir app --timeout-graceful-shutdown 3 --port 8000

echo [start] Frontend on http://localhost:3000
start "LangEvo frontend" /D "%FRONTEND%" cmd /k npx next dev -p 3000

echo [wait] Waiting for the frontend to come up...
set /a TRIES=0
:waitloop
curl -s -o nul http://localhost:3000 && goto ready
set /a TRIES+=1
if %TRIES% geq 90 (
    echo [warn] Frontend did not respond after 90s; opening the browser anyway.
    goto ready
)
timeout /t 1 /nobreak >nul
goto waitloop

:ready
start "" http://localhost:3000
echo [done] App opened. Close the "LangEvo backend" and "LangEvo frontend" windows to stop.
timeout /t 5 >nul
endlocal
