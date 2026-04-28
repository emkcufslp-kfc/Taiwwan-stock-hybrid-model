@echo off
echo Pushing to GitHub...
git push origin main
if %ERRORLEVEL% == 0 (
    echo.
    echo SUCCESS: Files pushed to GitHub!
) else (
    echo.
    echo Push failed. Error code: %ERRORLEVEL%
)
pause
