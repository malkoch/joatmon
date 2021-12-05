pip uninstall joatmon -y

python setup.py bdist_wheel
cd dist
pip install joatmon-1.0.1rc0-py3-none-any.whl
copy joatmon-1.0.1rc0-py3-none-any.whl ..\joatmon-1.0.1rc0-py3-none-any.whl

cd ..
rmdir build /s /q
rmdir dist /s /q
rmdir joatmon.egg-info /s /q
