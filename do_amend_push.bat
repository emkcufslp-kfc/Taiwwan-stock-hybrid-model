@echo off
cd /d "%~dp0"
echo ============================================
echo  台股估值系統 - Amend Commit and Push
echo ============================================

echo Removing stale lock file if present...
if exist ".git\index.lock" (
    del /f /q ".git\index.lock"
    echo Lock file removed.
)

echo.
echo Staging updated index.html (no literal secrets)...
git add index.html wrangler.jsonc working.md

echo.
echo Checking git log before amend...
git log --oneline -3

echo.
echo Amending last commit to remove secret...
git commit --amend --no-edit
if %ERRORLEVEL% neq 0 (
    echo Amend failed, trying fresh commit...
    git commit -m "fix: rename worker to -2, serve data/, prefill key via JS split"
)

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
    echo Trying force push...
    git push --force-with-lease origin main
    if %ERRORLEVEL% == 0 (
        echo ============================================
        echo  SUCCESS via force push!
        echo ============================================
    ) else (
        echo ERROR: Push failed. Error code: %ERRORLEVEL%
    )
)
pause
