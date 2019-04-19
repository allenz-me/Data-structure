
# 定义单链表的节点类
class Node(object):
    def __init__(self, item, next_=None):
        self.item = item
        self.next = next_

class SLList(object):
    # 初始化方法，可以不接收、接收一个或多个初始项
    def __init__(self, *items):
        self.__sentinel = Node(0) # sentinel 哨兵
        self.__size = 0 
        if len(items) > 0:
            for item in items[::-1]:
                self.addFirst(item)

    @property # 方法变属性而绑定起来，size是不能修改的属性
    def size(self):
        return self.__size

    def addFirst(self, item):
        self.__sentinel.next = Node(item, self.__sentinel.next)
        self.__size += 1

    def addLast(self, item):
        p = self.__sentinel
        while p.next is not None:
            p = p.next
        p.next = Node(item)
        self.__size += 1
    
    def getFirst(self):
        return self.__sentinel.next.item

    def getLast(self):
        p = self.__sentinel
        while p.next is not None:
            p = p.next
        return p.item

    def get(self, index:int):
        if index >= self.size:
            raise IndexError("Index out of range.")
        p = self.__sentinel
        for i in range(index):
            p = p.next
        return p.next.item

    # 定义__getitem__方法
    def __getitem__(self, index:int):
        return self.get(index)

    def set(self, index:int, item):
        if index >= self.size:
            raise IndexError("Index out of range.")
        p = self.__sentinel
        for i in range(index):
            p = p.next
        p.next = item
    
    # 定义__setitem__方法
    def __setitem__(self, index:int, item):
        self.set(index, item)

    def reverse(self):
        if self.__sentinel.next is None:
            return
        cur = self.__sentinel.next
        p = self.__sentinel.next.next
        while p is not None:
            cur.next = p.next
            p.next = self.__sentinel.next
            self.__sentinel.next = p
            p = cur.next
        


    def __iter__(self):
        self.__pointer = self.__sentinel
        return self

    def __next__(self):
        while self.__pointer.next is not None:
            self.__pointer = self.__pointer.next
            return self.__pointer.item
        else:
            raise StopIteration()



if __name__ == "__main__":
    sl = SLList(1, 2)
    sl.addFirst(3)
    for i in sl:
        print(i)
    print(sl[2])
        


