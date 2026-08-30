@echo off
setlocal EnableDelayedExpansion

rem Scheduled entry point for the daily pipeline.
rem   - runs Update.py with the repo's own venv
rem   - writes a dated log
rem   - leaves ONE marker file whose NAME is the result, so the outcome is
rem     visible in Explorer without opening anything
rem Any arguments are passed straight through to Update.py.

set "HERE=%~dp0"
set "REPO=%HERE%.."
set "PY=%REPO%\.venv\Scripts\python.exe"
set "LOGDIR=%HERE%logs"
set "PS=powershell -NoProfile -ExecutionPolicy Bypass -Command"

if not exist "%PY%" (
    echo Cannot find the interpreter at "%PY%".
    exit /b 2
)
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

for /f %%t in ('%PS% "(Get-Date).ToString('yyyy-MM-dd_HHmmss')"') do set "STAMP=%%t"
set "LOG=%LOGDIR%\run_%STAMP%.log"

del /q "%LOGDIR%\LAST_RUN_SUCCESS.txt" 2>nul
del /q "%LOGDIR%\LAST_RUN_FAILED.txt"  2>nul

set "T0=%TIME%"
pushd "%REPO%"
"%PY%" Live\Update.py %* > "%LOG%" 2>&1
set "RC=!ERRORLEVEL!"
popd

if "!RC!"=="0" (set "RESULT=SUCCESS") else (set "RESULT=FAILED")
set "MARKER=%LOGDIR%\LAST_RUN_!RESULT!.txt"

rem The marker carries the outcome, when it finished, how long the site says it
rem is current to, and -- when it failed -- the line that actually failed, so
rem the marker alone usually answers "what went wrong" without opening the log.
%PS% ^
  "$rc=%RC%; $log='%LOG%'; $marker='%MARKER%';" ^
  "$asof='unknown';" ^
  "try{ $asof=(Get-Content '%REPO%\docs\data\latest.json' -Raw | ConvertFrom-Json).meta.as_of }catch{}" ^
  "$why='';" ^
  "if($rc -ne 0){ $why=(Select-String -Path $log -Pattern '\[ABORT\]|\[FAIL\]|Traceback|Error:' | Select-Object -Last 3 | ForEach-Object { $_.Line.Trim() }) -join \"`r`n         \" }" ^
  "$lines=@(('{0}  exit {1}' -f (('%RESULT%')), $rc), ('finished : ' + (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')), ('site as of: ' + $asof), ('log      : ' + $log));" ^
  "if($why){ $lines += ('why      : ' + $why) }" ^
  "[IO.File]::WriteAllLines($marker, $lines, (New-Object Text.UTF8Encoding $false))"

rem Keep a fortnight of logs, drop the rest.
%PS% "Get-ChildItem '%LOGDIR%\run_*.log' | Sort-Object LastWriteTime -Descending | Select-Object -Skip 14 | Remove-Item -Force -ErrorAction SilentlyContinue"

type "%MARKER%"
exit /b !RC!
