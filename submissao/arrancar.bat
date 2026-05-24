@echo off
title Warehouse Vision AI - Demo
color 0B

echo ========================================================
echo     INICIANDO WAREHOUSE VISION AI (MODO PORTABLE)
echo ========================================================
echo.

set PYTHON_CMD=

IF EXIST "..\.venv\Scripts\python.exe" (
    set PYTHON_CMD="..\.venv\Scripts\python.exe"
    goto found_python
)

python --version >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=python
    goto found_python
)

py --version >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=py
    goto found_python
)

:no_python
color 0C
echo [ERRO CRITICO] Python nao encontrado no PATH do Windows!
echo.
echo O teu computador precisa de ter o Python instalado para correr esta App.
echo Por favor, instala em https://www.python.org/downloads/
echo.
echo IMPORTANTE: Durante a instalacao, na primeira janela, 
echo tens de marcar OBRIGATORIAMENTE a caixa ca em baixo que diz:
echo "Add Python to PATH" ou "Add Python to environment variables".
echo.
pause
exit /b

:found_python
IF NOT EXIST ".venv_portable\Scripts\python.exe" (
    echo [1/3] A criar um ambiente virtual isolado - apenas na primeira vez...
    %PYTHON_CMD% -m venv .venv_portable
)

IF NOT EXIST ".venv_portable\Scripts\python.exe" (
    color 0C
    echo [ERRO CRITICO] Falha ao criar ou localizar o ambiente virtual em .venv_portable!
    echo Por favor, apaga a pasta .venv_portable caso exista e corre este script de novo.
    echo.
    pause
    exit /b
)


echo [2/3] A carregar ambiente e a instalar bibliotecas pesadas...
echo Isto pode demorar alguns minutos na primeira execucao (especialmente se for a primeira vez).
echo.
.venv_portable\Scripts\python.exe -m pip install --upgrade pip
.venv_portable\Scripts\python.exe -m pip install -r app\requirements.txt

echo.
echo [3/3] Tudo pronto! A iniciar a aplicacao no teu browser...
echo Podes fechar esta janela preta quando quiseres desligar a app.
echo.
.venv_portable\Scripts\python.exe -m streamlit run app\app.py

pause

