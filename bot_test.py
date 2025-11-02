import copy

from joatmon.bot.memory import process


proc = process.Process.from_name("Raid")

base_address = proc.get_base_address()
print(f'{base_address=}')

# app_model_addr_value = proc.read_value(base_address.address + 55_079_968, 'ulong')
# print(f'{app_model_addr_value=}')

# add_model_addr_value = proc.read_value(app_model_addr_value + 0x18, 'ulong')
# print(f'{add_model_addr_value=}')
# add_model_addr_value = proc.read_value(app_model_addr_value + 0xc0, 'ulong')
# print(f'{add_model_addr_value=}')
# add_model_addr_value = proc.read_value(app_model_addr_value + 0x0, 'ulong')
# print(f'{add_model_addr_value=}')
# add_model_addr_value = proc.read_value(app_model_addr_value + 0xb8, 'ulong')
# print(f'{add_model_addr_value=}')
# add_model_addr_value = proc.read_value(app_model_addr_value + 0x8, 'ulong')
# print(f'{add_model_addr_value=}')

# proc.dump('raid_dumps')

addresses = proc.find_value(6691, 'int')
print(addresses)
while addresses:
    req = int(input('enter new value: '))
    loop_addresses = copy.deepcopy(addresses)
    for address in loop_addresses:
        value = address.read('int')
        if value == req:
            addresses = list(filter(lambda x: x.address != address.address, addresses))
    print(addresses)
