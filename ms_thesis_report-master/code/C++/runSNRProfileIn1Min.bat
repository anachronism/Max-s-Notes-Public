@echo off
for /F "tokens=1-3 delims=:." %%a in ("%time%") do (
   set timeHour=%%a
   set timeMinute=%%b
   set timeSeconds=%%c
)
rem Convert HH:MM to minutes + 1 + 4*60
set /A newTime=timeHour*60 + timeMinute + 1 + 4*60
rem Convert new time back to HH:MM
set /A timeHour=newTime/60, timeMinute=newTime%%60
rem Adjust new hour and minute
if %timeHour% gtr 23 set timeHour=0
if %timeHour% lss 10 set timeHour=0%timeHour%
if %timeMinute% lss 10 set timeMinute=0%timeMinute%

echo RunAttenProfile.exe ..\spocexp\Data8-normalized.txt %timeHour%:%timeMinute%:%timeSeconds% 100