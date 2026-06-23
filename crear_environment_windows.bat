@echo off
setlocal

REM ================================================================
REM Crea un entorno virtual local para ejecutar el notebook.
REM
REM IMPORTANTE:
REM TensorFlow instala algunos archivos con rutas muy largas. Para evitar
REM errores de Windows Long Path, el entorno se crea en una ruta corta:
REM   %USERPROFILE%\.venvs\miax-b4-t1
REM y no dentro de la carpeta profunda del proyecto.
REM ================================================================

cd /d "%~dp0"
set "PROJECT_DIR=%CD%"
set "VENV_DIR=%USERPROFILE%\.venvs\miax-b4-t1"

echo.
echo === Creacion del entorno MIAX B4-T1 ===
echo Carpeta del proyecto:
echo %PROJECT_DIR%
echo.
echo Carpeta del entorno virtual:
echo %VENV_DIR%
echo.

where py >nul 2>nul
if errorlevel 1 (
    echo ERROR: No se encontro el lanzador de Python "py".
    echo Instala Python desde https://www.python.org/ y marca "Add python.exe to PATH".
    goto error
)

if not exist requirements.txt (
    echo ERROR: No se encontro requirements.txt en esta carpeta.
    goto error
)

if not exist "%USERPROFILE%\.venvs" (
    mkdir "%USERPROFILE%\.venvs"
    if errorlevel 1 goto error
)

if not exist "%VENV_DIR%" (
    echo Creando entorno virtual en ruta corta ...
    py -m venv "%VENV_DIR%"
    if errorlevel 1 goto error
) else (
    echo Ya existe el entorno. Se reutilizara:
    echo %VENV_DIR%
)

echo.
echo Activando entorno ...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 goto error

echo.
echo Actualizando pip ...
python -m pip install --upgrade pip
if errorlevel 1 goto error

echo.
echo Instalando paquetes de requirements.txt ...
python -m pip install -r "%PROJECT_DIR%\requirements.txt"
if errorlevel 1 goto error

echo.
echo Registrando kernel de Jupyter ...
python -m ipykernel install --user --name miax-b4-t1 --display-name "Python (MIAX B4-T1)"
if errorlevel 1 goto error

echo.
echo ================================================================
echo Entorno creado correctamente.
echo.
echo Para activarlo en otra consola:
echo   "%VENV_DIR%\Scripts\activate.bat"
echo.
echo Para volver a la carpeta del proyecto:
echo   cd /d "%PROJECT_DIR%"
echo.
echo Para abrir Jupyter desde la carpeta del proyecto:
echo   jupyter notebook
echo.
echo En el notebook selecciona el kernel:
echo   Python (MIAX B4-T1)
echo ================================================================
echo.
pause
exit /b 0

:error
echo.
echo ================================================================
echo El script se detuvo por un error.
echo Revisa el mensaje que aparece justo encima de este bloque.
echo.
echo Si el error vuelve a mencionar Windows Long Path, activa rutas largas
echo en Windows o mueve el proyecto a una ruta mas corta, por ejemplo C:\MIAX.
echo ================================================================
echo.
pause
exit /b 1
