rmdir joatmon-egg-info /s
rmdir build /s
pip uninstall joatmon -y
pip install .
python scripts\assistant.py
