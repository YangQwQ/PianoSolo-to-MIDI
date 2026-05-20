CD /D %~dp0
mkdir workspace\mp3s 2>nul
mkdir workspace\midis 2>nul

if exist .venv\Scripts\python.exe (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)

%PY% audios_to_midis.py transcribe_file ^
    --input ./workspace/mp3s ^
    --output ./workspace/midis ^
    --align-strength 1.0 ^
    --align-threshold 0.005
pause