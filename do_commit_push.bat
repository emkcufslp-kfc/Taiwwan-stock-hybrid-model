@echo off
cd /d "%~dp0"
echo ============================================
echo  台股估值系統 - Auto Commit and Push
echo ============================================

echo Removing stale lock file if present...
if exist ".git\index.lock" (
    del /f /q ".git\index.lock"
    echo Lock file removed.
) else (
    echo No lock file found.
)

echo.
echo Staging all changes...
git add -A
if %ERRORLEVEL% neq 0 (echo ERROR: git add failed & pause & exit /b 1)

echo.
echo Committing...
git commit -m "fix: rename worker to -2, serve data/, fix html truncation, prefill Groq key"
if %ERRORLEVEL% neq 0 (echo NOTE: Nothing new to commit or commit failed & goto push)

:push
echo.
echo Pushing to GitHub...
git push origin main
if %ERRORLEVEL% == 0 (
    echo.
    echo ============================================
    echo  SUCCESS: Pushed to GitHub!
    echo  Cloudflare will auto-deploy in ~30 seconds
    echo ============================================
) else (
    echo.
    echo ERROR: Push failed. Error code: %ERRORLEVEL%
)
pause
