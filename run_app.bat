@echo off

cd /d "D:\Web application\SkinLesionResearchPlatform"

echo ==========================================
echo   Skin Lesion Research Platform
echo ==========================================
echo.
echo Activating virtual environment...
echo.

call venv\Scripts\activate.bat

echo.
echo Starting Streamlit application...
echo.

python -m streamlit run app.py

pause