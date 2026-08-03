@echo off
setlocal

rem Change this path if MuJoCo was extracted elsewhere.
set "MUJOCO_HOME=C:\mujoco\mujoco-3.11.0-windows-x86_64"
set "MODEL=%~dp0models\falling_box.xml"

if not exist "%MUJOCO_HOME%\bin\simulate.exe" (
    echo [ERROR] simulate.exe was not found:
    echo %MUJOCO_HOME%\bin\simulate.exe
    echo.
    echo Edit MUJOCO_HOME in run_windows.bat to match your installation path.
    pause
    exit /b 1
)

if not exist "%MODEL%" (
    echo [ERROR] Model file was not found:
    echo %MODEL%
    pause
    exit /b 1
)

"%MUJOCO_HOME%\bin\simulate.exe" "%MODEL%"

endlocal
