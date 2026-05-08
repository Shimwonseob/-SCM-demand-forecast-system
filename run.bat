@echo off
cd /d "C:\Users\simwo\OneDrive\Desktop\수요예측시스템\SCM_Claude Code"
echo Streamlit 시작 중...
start "" cmd /k "streamlit run app.py --server.port 8502"
timeout /t 4 /nobreak > nul
start "" "http://localhost:8502"
exit
