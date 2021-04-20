import os,sys
fold=sys.argv[1]
ll=[]
for ff in os.listdir(fold):
    ff0=ff.split('_')[0]
    if ff0 not in ll:
        ll.append(ff0)

print(sorted(ll))

#with open('../input/va_cnty_list.txt','r') as f:
#    va_ll=[line.rstrip('\n') for line in f]
#
#print(len(ll))
#print('Remaining counties')
#print(set(va_ll)-set(ll))
