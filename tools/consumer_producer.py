import time
import random
from multiprocessing import Queue,Process
 
def consumer(name,q):
    while True:
        res=q.get()
        time.sleep(random.randint(1,3))
        print('消费者》》%s 准备开吃%s。'%(name,res))
 
def producer(name,q):
    for i in range(5):
        time.sleep(random.randint(1,2))
        res='大虾%s'%i
        q.put(res)
        print('生产者》》》%s 生产了%s'%(name,res))
 
if __name__ == '__main__':
    q=Queue()#一个队列
 
    p1=Process(target=producer,args=('monicx',q))
    c1=Process(target=consumer,args=('lili',q))
 
    p1.start()

    c1.start()