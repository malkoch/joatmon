import os

count = 0
for path, folders, files in os.walk('joatmon'):
    for file in files:
        count += len(open(os.path.join(path, file), 'rb').readlines())
for path, folders, files in os.walk('scripts'):
    for file in files:
        count += len(open(os.path.join(path, file), 'rb').readlines())
for path, folders, files in os.walk('tests'):
    for file in files:
        count += len(open(os.path.join(path, file), 'rb').readlines())
print(count)
