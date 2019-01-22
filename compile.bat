@ECHO OFF
setlocal enabledelayedexpansion

SET FILE_NAME=natural-merge

set "commands[0]=rm -rf compile\* dist\* vc140.pdb"
set "commands[1]=nvcc -g -O3 .\src\%FILE_NAME%.cu -o compile\%FILE_NAME%"
set "commands[2]=copy compile\%FILE_NAME%.exe dist\%FILE_NAME%.exe"
set "commands[3]=.\dist\%FILE_NAME%.exe %1"

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