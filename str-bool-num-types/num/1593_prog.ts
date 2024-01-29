interface QueueNode<T> {
  value: T;
  next: QueueNode<T> | null;
}

interface Queue<T extends any> {
  enqueue: (value: T) => void;
  dequeue: () => T | undefined;
  peek: () => T | undefined;
  clear: () => void;
  size: () => void;
  forEach: (callback?: (value: T, index: <FILL>) => void) => void;
  toArray: () => T[];
}

const createNode = <T>(value: T): QueueNode<T> => {
  return {
    value,
    next: null
  }
}
export const createQueue = <T extends any>(): Queue<T> => {
  let head: QueueNode<T> | null = null;
  let tail: QueueNode<T> | null = null;
  let size = 0;
  return {
    enqueue(value: T) {
      const node = createNode<T>(value);
      if (head && tail) {
        tail.next = node
        tail = node
      } else {
        head = node
        tail = node
      }
      size++
    },
    dequeue() {
      const current = head
      if (!current) {
        return undefined
      }
      head = current.next
      size--
      return current.value
    },
    peek() {
      if (!head) {
        return
      }
      return head.value
    },
    clear() {
      head = null
      tail = null
      size = 0
    },
    size() {
      return size
    },
    forEach(callback) {
      let current = head
      let index = 0
      while(current) {
        if (typeof callback === "function") {
          callback(current.value, index)
        }
        current = current.next
        index++
      }
    },
    toArray() {
      const arr: T[] = []
      this.forEach((value) => {
        arr.push(value)
      })
      return arr
    }
  }
}