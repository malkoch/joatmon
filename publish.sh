bumpversion patch
python setup.py sdist bdist_wheel
twine upload dist/*

sudo rm -r build
sudo rm -r dist
sudo rm -r *.egg-info
