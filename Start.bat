CD /D %~dp0
mkdir workspace\mp3s 2>nul
mkdir workspace\midis 2>nul
python audios_to_midis.py transcribe_file ^
    --input=./workspace/mp3s ^
    --output=./workspace/midis
pause
