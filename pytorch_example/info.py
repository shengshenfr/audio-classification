import psutil
import commands
import os
def getRAMinfo():
    p = os.popen('free')
    i = 0
    while 1:
        i = i + 1
        line = p.readline()
        if i==2:
            return(line.split()[1:4])

def getDiskSpace():
    p = os.popen("df -h /")
    i = 0
    while 1:
        i = i +1
        line = p.readline()
        if i==2:
            return(line.split()[1:5])

print (u"cpu number: %s" % psutil.cpu_count(logical=False))

total = str(round(psutil.virtual_memory().total/(1024.0*1024.0*1024.0), 2))
print(total)

RAM_stats = getRAMinfo()
RAM_total = round(int(RAM_stats[0]) / (1024.0*1024.0),1)
print(RAM_total)


DISK_stats = getDiskSpace()
DISK_total = DISK_stats[0]
print(DISK_total)
