import time
import sys

print('Hello')
print(int(10/101 * 100))
for i in range(100):
    time.sleep(0.2)
    res = int(i / 100 * 100)
    sys.stdout.write("\r%d%%" % res)
    sys.stdout.flush()
