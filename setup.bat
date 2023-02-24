python setup.py bdist_wheel
cd dist
pip install joatmon-1.0.1rc1-py3-none-any.whl --force-reinstall
cd ..
rmdir build /s /q
rmdir dist /s /q
rmdir joatmon.egg-info /s /q
