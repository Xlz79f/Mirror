@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ═══════════════════════════════════
echo   镜 · 情感对话助手 - 安装
echo ═══════════════════════════════════
echo.

if exist "python\python.exe" (
    echo [跳过] Python 已安装
    goto run
)

echo [1/3] 下载嵌入式 Python 3.12...
curl -L -o python.zip "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip"
echo [2/3] 解压...
mkdir python
tar -xf python.zip -C python
del python.zip

echo [3/3] 安装依赖（约 3-5 分钟）...
echo python312.zip>python\python312._pth
echo .>>python\python312._pth
echo Lib\site-packages>>python\python312._pth
echo import site>>python\python312._pth

curl -sL https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python\python.exe get-pip.py --no-warn-script-location
del get-pip.py
python\python.exe -m pip install -r requirements.txt

echo.
echo [完成] 安装完毕！
echo.

:run
set PYTHONUTF8=1
python\python.exe jing_demo.py
pause
