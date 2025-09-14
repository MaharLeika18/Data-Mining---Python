@echo off
echo Installing required Python modules...

python -m pip install --upgrade pip

pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn

echo.
echo Installation complete!
pause
