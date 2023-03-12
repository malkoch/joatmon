import psutil

from joatmon.system.screen import grab

print(psutil.virtual_memory())
print(psutil.swap_memory())
print(psutil.cpu_percent())
print(psutil.cpu_count(logical=True))
print(psutil.cpu_count(logical=False))
# print(psutil.sensors_temperatures())
print(psutil.sensors_battery())
# print(psutil.sensors_fans())
# for p in psutil.pids():
#     process = psutil.Process(p)
#     print(process)
