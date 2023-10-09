from joatmon.structure.dictionary import CustomDictionary

d = {'item1': 2, 'item2': 3, 'item3': [1, 2, 3, 4, 5], 'item4': {'item5': 1, 'item6': [1, 2], 'item7': {'item8': 1}}}
cd = CustomDictionary(d)

print(f"{d=:}")
print(f"{cd=:}")
print(f"{cd['item1']=:}")
print(f"{cd['item2']=:}")
print(f"{cd['item3']=:}")
print(f"{cd['item4']=:}")
print(f"{cd['item4.item5']=:}")
print(f"{cd['item4.item6']=:}")
print(f"{cd['item4.item7']=:}")
print(f"{cd['item4.item7.item8']=:}")
print(f"{cd.item1=:}")
print(f"{cd.item2=:}")
print(f"{cd.item3=:}")
print(f"{cd.item4=:}")
print(f"{cd.item4.item5=:}")
print(f"{cd.item4.item6=:}")
print(f"{cd.item4.item7=:}")
print(f"{cd.item4.item7.item8=:}")

cd['item1'] = 11
cd['item4.item5'] = 11
cd['item4.item7.item8'] = 11
print(f"{cd=:}")

cd.item1 = 22
cd.item4.item5 = 22
cd.item4.item7.item8 = 22
print(f"{cd=:}")
