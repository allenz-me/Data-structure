package src.main.java.List;

import java.util.Iterator;

/* Singly linked list with generics*/
public class SLList<T> implements List61b<T>,Iterable<T> {
    private Node sentinel = new Node(null, null);
    private int size;

    // 单链表的节点类
    private class Node{
        T item;
        Node next;

        public Node(T item, Node next){
            this.item = item;
            this.next = next;
        }
    }
    // 初始化方法，可以不接收、接收一个或多个初始项
    public SLList() {
        size = 0;
        sentinel.next = null;
    }
    public SLList(T item) {
        size = 1;
        sentinel.next = new Node(item, null);
    }
    public SLList(T[] items){
        int l = items.length;
        for (int i=l-1;i<0;i--){
            addFirst(items[i]);
        }
    }

    public int size() {
        return size;
    }
    
    public void addFirst(T item) {
        sentinel.next = new Node(item, sentinel.next);
        size += 1;
    }
    
    public void addLast(T item){
        Node p = this.sentinel;
        while (p.next != null){
            p = p.next;
        }
        p.next = new Node(item, null);
        size += 1;
    }

    public T getFirst() {
        return sentinel.next.item;
    }

    
    public T getLast() {
        Node p = this.sentinel;
        while (p.next != null) {
            p = p.next;
        }
        return p.item;
    }

    public T get(int index) {
        if (index >= size) {
            throw new IndexOutOfBoundsException();
        }
        Node p = this.sentinel;
        for (int i=0;i<index;i++) {
            p = p.next;
        }
        return p.next.item;
    }

    @Override
    public void print() {
        this.forEach(each -> System.out.println(each));
        System.out.println("==========");
    }

    @Override
    public T removeLast() {
        Node p = this.sentinel;
        while (p.next.next != null) {
            p = p.next;
        }
        T item = p.next.item;
        p.next = null;
        size -= 1;
        return item;
    }

    public void deleteFirst() {
        size -= 1;
        sentinel.next = sentinel.next.next;
    }

    /* 向表中的第i个位置插入item */
    public void insert(int pos, T x) {
        if (pos > size || pos < 0){
            throw new IndexOutOfBoundsException();
        }
        Node p = sentinel;
        for (int i=0;i<pos;i++) {
            p = p.next;
        }
        p.next = new Node(x, p.next);
        size += 1;
    }

    /* Reverse the elements iteratively. */
    public void reverse() {
        if (sentinel.next == null) {
            return;
        }
        Node cur = sentinel.next;
        Node p = sentinel.next.next;
        while ( p != null) {
            cur.next = p.next;
            p.next = sentinel.next;
            sentinel.next = p;
            p = cur.next;
        }
    }

    public void reverseRecursively() {
        sentinel.next = reverseRecursiveHelper(sentinel.next);
    }
    private Node reverseRecursiveHelper(Node front) {
        // 逻辑短路，保证判断front.next时front是非null的
        if (front == null || front.next == null) {
            return front;
        } else {
            Node reversed = reverseRecursiveHelper(front.next);
            front.next.next = front;
            front.next = null;
            return reversed;
        }
    }
    
    private class SLListIterator implements Iterator<T>{
        private Node pointer = sentinel;

        @Override
        public boolean hasNext() {
            return pointer.next != null;
        }

        @Override
        public T next() {
            pointer = pointer.next;
            return pointer.item;
        }
    }
    @Override
    public Iterator<T> iterator() {
        return new SLListIterator();
    }


    public static void main(String[] args) {
        SLList<String> sll = new SLList<>();
        sll.addFirst("ahj");
        sll.addFirst("kas8");
        sll.addLast("iks");
        sll.insert(0, "afa");
        sll.print();
        sll.reverse();
        sll.print();
    }

}