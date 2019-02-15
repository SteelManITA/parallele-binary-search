@ECHO OFF
setlocal enabledelayedexpansion

SET FILE_NAME=natural-merge
SET /A rand=%RANDOM%

set "commands[0]=rm -rf compile\* vc140.pdb"
set "commands[1]=nvcc -g -O3 -arch=sm_35 -rdc=true -lineinfo .\src\%FILE_NAME%.cu -o compile\%FILE_NAME%-%rand%"
set "commands[2]=copy compile\%FILE_NAME%-%rand%.exe dist\%FILE_NAME%-%rand%.exe"
set "commands[3]=.\dist\%FILE_NAME%-%rand%.exe %1"

set "command=%commands[0]%"
set "i=1"
:loop
if defined commands[%i%] (
    set "command=%command% && !commands[%i%]!"
    set /a "i+=1"
    GOTO :loop
)

echo !command!
%command%