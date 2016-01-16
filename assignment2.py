from __future__ import division
import math
import doctest


class SinglyLinkedNode(object):
    """
    >>> node = SinglyLinkedNode(1)
    >>> node.item
    1

    >>> node1 = SinglyLinkedNode(1)
    >>> node2 = SinglyLinkedNode(2, node1)
    >>> node2.next
    1

    """
    def __init__(self, item=None, next_link=None):
        super(SinglyLinkedNode, self).__init__()
        self._item = item
        self._next = next_link

    @property
    def item(self):
        return self._item

    @item.setter
    def item(self, item):
        self._item = item

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, next):
        self._next = next

    def __repr__(self):
        return repr(self.item)


class LinkedListIterator(object):
    def __init__(self, node):
        self.pointer = node

    def next(self):
        if self.pointer is None:
            raise StopIteration()

        result = self.pointer
        self.pointer = self.pointer.next
        return result


class SinglyLinkedList(object):
    """
    >>> l = SinglyLinkedList()
    >>> print l.head
    None

    >>> l = SinglyLinkedList()
    >>> l.prepend(1)
    >>> len(l)
    1

    >>> l = SinglyLinkedList()
    >>> l.prepend(1)
    >>> l.prepend(2)
    >>> print l
    List:2->1

    >>> l = SinglyLinkedList()
    >>> l.prepend(1)
    >>> print l.head
    1

    >>> l = SinglyLinkedList()
    >>> l.prepend(1)
    >>> l.__contains__(11)
    False

    >>> l = SinglyLinkedList()
    >>> l.prepend(1)
    >>> l.__contains__(1)
    True

    >>> l = SinglyLinkedList()
    >>> l.prepend(1)
    >>> l.prepend(2)
    >>> l.remove(1)
    True

    >>> l = SinglyLinkedList()
    >>> l.prepend(1)
    >>> l.prepend(2)
    >>> l.remove(2)
    True
    >>> print l.head
    1

    """
    def __init__(self):
        super(SinglyLinkedList, self).__init__()
        self.head = None

    def __len__(self):
        counter = 0
        for _ in self:
            counter += 1
        return counter

    def __iter__(self):
        return LinkedListIterator(self.head)

    def __contains__(self, item):
        for node in self:
            if node.item == item:
                return True
        return False

    def remove(self, item):
        prev_node = None
        for node in self:
            if node.item == item:
                if prev_node is None:
                    # This is the head node
                    self.head = node.next
                else:
                    prev_node.next = node.next
                del node
                return True
            prev_node = node
        return False

    def prepend(self, item):
        node = SinglyLinkedNode(item)
        if self.head is not None:
            node.next = self.head
        self.head = node

    def __repr__(self):
        s = "List:" + "->".join([str(node.item) for node in self])
        return s


class KeyValuePair(object):
    """
    >>> pair = KeyValuePair(1, 2)
    >>> print pair
    {key: 1, value: 2}

    """
    def __init__(self, key, value):
        super(KeyValuePair, self).__init__()
        self.key = key
        self.value = value

    def __repr__(self):
        s = "{key: " + str(self.key) + ", value: " + str(self.value) + "}"
        return str(s)


class ChainedHashDict(object):
    """
    >>> dict = ChainedHashDict(15)
    >>> print dict.bin_count
    15

    >>> dict = ChainedHashDict()
    >>> dict.__setitem__(1, 2)
    >>> len(dict)
    1

    >>> dict = ChainedHashDict(5)
    >>> dict.__setitem__(1, 2)
    >>> dict.__setitem__("abcd", 3)
    >>> dict.display()
    Index | Value
    0  |  {key: abcd, value: 3}
    1  |  None
    2  |  {key: 1, value: 2}
    3  |  None
    4  |  None

    >>> dict = ChainedHashDict(5)
    >>> dict.__setitem__(1, 2)
    >>> dict.__setitem__(1, 3)
    >>> dict.display()
    Index | Value
    0  |  None
    1  |  None
    2  |  List:{key: 1, value: 3}->{key: 1, value: 2}
    3  |  None
    4  |  None

    >>> dict = ChainedHashDict(5, 0.7, terrible_hash(1))
    >>> dict.__setitem__(1, 2)
    >>> dict.__setitem__("abcd", 3)
    >>> dict.display()
    Index | Value
    0  |  None
    1  |  None
    2  |  List:{key: abcd, value: 3}->{key: 1, value: 2}
    3  |  None
    4  |  None

    >>> dict = ChainedHashDict(5)
    >>> dict.__setitem__(1, 2)
    >>> dict.__getitem__(1)
    2

    >>> dict = ChainedHashDict(5)
    >>> dict.__setitem__(1, 2)
    >>> dict.__setitem__(1, 3)
    >>> dict.__getitem__(1)
    [3, 2]

    >>> dict = ChainedHashDict(5)
    >>> dict.__setitem__(1, 2)
    >>> dict.__getitem__("abcd")
    Traceback (most recent call last):
     ...
    Exception: Key not found in Hash Table!

    >>> dict = ChainedHashDict(5)
    >>> dict.__setitem__(1, 2)
    >>> dict.__delitem__(1)
    True

    >>> dict = ChainedHashDict()
    >>> dict.__setitem__(1, 2)
    >>> dict.__delitem__("abcd")
    False

    >>> dict = ChainedHashDict()
    >>> dict.__setitem__(1, 2)
    >>> dict.__setitem__("abcd", 3)
    >>> dict.__delitem__(1)
    True
    >>> len(dict)
    1

    >>> dict = ChainedHashDict()
    >>> dict.__setitem__(1, 2)
    >>> dict.__setitem__(1, 3)
    >>> dict.__delitem__(1)
    True
    >>> len(dict)
    0

    >>> dict = ChainedHashDict()
    >>> dict.__setitem__(1, 2)
    >>> dict.__contains__(1)
    True

    >>> dict = ChainedHashDict()
    >>> dict.__setitem__(1, 2)
    >>> dict.__contains__("abcd")
    False

    >>> dict = ChainedHashDict(5)
    >>> dict.__setitem__(1, 2)
    >>> dict.__setitem__("abcd", 3)
    >>> dict.__setitem__(4, 5)
    >>> print dict.load_factor
    0.6

    >>> dict = ChainedHashDict(5)
    >>> dict.__setitem__(1, 2)
    >>> dict.__setitem__("abcd", 3)
    >>> dict.__setitem__(4, 5)
    >>> print dict.bin_count
    5
    >>> print dict.load_factor
    0.6
    >>> dict.__setitem__("efgh", 2)
    >>> print dict.bin_count
    10
    >>> print dict.load_factor
    0.4

    """
    def __init__(self, bin_count=10, max_load=0.7, hashfunc=hash):
        super(ChainedHashDict, self).__init__()
        self.table = [None] * bin_count
        self.max_load = max_load
        self.total_bins = bin_count
        self.hash_func = hashfunc
        self.max_elements = math.floor(max_load * bin_count)
        self.total_elements = 0

    @property
    def load_factor(self):
        return self.total_elements / self.total_bins

    @property
    def bin_count(self):
        return self.total_bins

    def rebuild(self, bincount):
        # initializing all relevant variables
        self.old_table = self.table
        self.table = [None] * bincount
        self.total_bins = bincount
        self.max_elements = math.floor(self.max_load * bincount)
        self.total_elements = 0

        for table_value in self.old_table:
            if type(table_value).__name__ == "SinglyLinkedList":
                for node in table_value:
                    self.__setitem__(node.item.key, node.item.value)
            elif table_value is not None:
                self.__setitem__(table_value.key, table_value.value)

    def __compression_func__(self, key):
        a, b = 7, 20  # randomly chosen positive integers
        prime = 31
        return ((a * self.hash_func(key) + b) % prime) % self.bin_count

    def __getitem__(self, key):
        index = self.__compression_func__(key)
        if type(self.table[index]).__name__ == "SinglyLinkedList":
            arr = []
            for node in self.table[index]:
                if node.item.key == key:
                    arr += [node.item.value]
            return arr
        elif self.table[index] is not None:
            return self.table[index].value

        #  Control comes here when Key was Not found!
        raise Exception('Key not found in Hash Table!')

    def __setitem__(self, key, value):
        index = self.__compression_func__(key)
        if type(self.table[index]).__name__ == "NoneType":
            self.table[index] = KeyValuePair(key, value)
        elif type(self.table[index]).__name__ == "SinglyLinkedList":
            self.table[index].prepend(KeyValuePair(key, value))
        else:
            list = SinglyLinkedList()
            list.prepend(self.table[index])
            list.prepend(KeyValuePair(key, value))
            self.table[index] = list
        self.total_elements += 1

        # Check load factor
        if self.total_elements > self.max_elements:
            # Doubling the size of hash table
            self.rebuild(self.bin_count * 2)

    def __delitem__(self, key):
        index = self.__compression_func__(key)
        if type(self.table[index]).__name__ == "SinglyLinkedList":
            #  Find all nodes for given key
            results = [node for node in self.table[index] if node.item.key == key]
            for r in results:
                self.table[index].remove(r.item)
                self.total_elements -= 1
            if len(self.table[index]) == 0:
                self.table[index] = None
        elif self.table[index] is not None:
            self.total_elements -= 1
            self.table[index] = None
        else:
            return False
        return True

    def __contains__(self, key):
        index = self.__compression_func__(key)
        if type(self.table[index]).__name__ == "SinglyLinkedList":
            for node in self.table[index]:
                if node.item.key == key:
                    return True
        elif self.table[index] is not None and self.table[index].key == key:
            return True
        return False  # Key not found

    def __len__(self):
        return self.total_elements

    def display(self):
        print "Index | Value"  # Header
        for i in range(self.total_bins):
            print i, " | ", self.table[i]


class OpenAddressHashDict(object):
    """
    >>> dict = OpenAddressHashDict(15)
    >>> print dict.bin_count
    15

    >>> dict = OpenAddressHashDict()
    >>> dict.__setitem__(1, 2)
    >>> len(dict)
    1

    >>> dict = OpenAddressHashDict(5)
    >>> dict.__setitem__(1, 2)
    >>> dict.__setitem__("abcd", 3)
    >>> dict.display()
    Index | Value
    0  |  None
    1  |  {key: 1, value: 2}
    2  |  {key: abcd, value: 3}
    3  |  None
    4  |  None

    >>> dict = OpenAddressHashDict(5)
    >>> dict.__setitem__(1, 2)
    >>> dict.__setitem__(1, 3)
    >>> dict.display()
    Index | Value
    0  |  None
    1  |  {key: 1, value: 2}
    2  |  {key: 1, value: 3}
    3  |  None
    4  |  None

    >>> dict = OpenAddressHashDict(5, 0.7, terrible_hash(1))
    >>> dict.__setitem__(1, 2)
    >>> dict.__setitem__("abcd", 3)
    >>> dict.display()
    Index | Value
    0  |  None
    1  |  {key: 1, value: 2}
    2  |  {key: abcd, value: 3}
    3  |  None
    4  |  None

    >>> dict = OpenAddressHashDict(5)
    >>> dict.__setitem__(1, 2)
    >>> dict.__getitem__(1)
    2

    >>> dict = OpenAddressHashDict(5)
    >>> dict.__setitem__(1, 2)
    >>> dict.__getitem__("abcd")
    Traceback (most recent call last):
     ...
    Exception: Key not found in Hash Table!

    >>> dict = OpenAddressHashDict(5)
    >>> dict.__setitem__(1, 2)
    >>> dict.__delitem__(1)
    True
    >>> dict.display()
    Index | Value
    0  |  None
    1  |  DELETED
    2  |  None
    3  |  None
    4  |  None

    >>> dict = OpenAddressHashDict()
    >>> dict.__setitem__(1, 2)
    >>> dict.__delitem__("abcd")
    False

    >>> dict = OpenAddressHashDict()
    >>> dict.__setitem__(1, 2)
    >>> dict.__setitem__("abcd", 3)
    >>> dict.__delitem__(1)
    True
    >>> len(dict)
    1

    >>> dict = OpenAddressHashDict()
    >>> dict.__setitem__(1, 2)
    >>> dict.__contains__(1)
    True

    >>> dict = OpenAddressHashDict()
    >>> dict.__setitem__(1, 2)
    >>> dict.__contains__("abcd")
    False

    >>> dict = OpenAddressHashDict(5)
    >>> dict.__setitem__(1, 2)
    >>> dict.__setitem__("abcd", 3)
    >>> dict.__setitem__(4, 5)
    >>> print dict.load_factor
    0.6

    >>> dict = OpenAddressHashDict(5)
    >>> dict.__setitem__(1, 2)
    >>> dict.__setitem__("abcd", 3)
    >>> dict.__setitem__(4, 5)
    >>> print dict.bin_count
    5
    >>> print dict.load_factor
    0.6
    >>> dict.__setitem__("efgh", 2)
    >>> print dict.bin_count
    10
    >>> print dict.load_factor
    0.4

    """
    def __init__(self, bin_count=10, max_load=0.7, hashfunc=hash):
        # Using Linear Probe strategy for Open Addressing
        super(OpenAddressHashDict, self).__init__()
        self.table = [None] * bin_count
        self.max_load = max_load
        self.total_bins = bin_count
        self.hash_func = hashfunc
        self.max_elements = math.floor(max_load * bin_count)
        self.total_elements = 0

    @property
    def load_factor(self):
        return self.total_elements / self.total_bins

    @property
    def bin_count(self):
        return self.total_bins

    def rebuild(self, bincount):
        # initializing all relevant variables
        self.old_table = self.table
        self.table = [None] * bincount
        self.total_bins = bincount
        self.max_elements = math.floor(self.max_load * bincount)
        self.total_elements = 0

        for table_value in self.old_table:
            if table_value is not None:
                self.__setitem__(table_value.key, table_value.value)

    def __compression_func__(self, key, offset=0):
        # Linear Probing
        return (self.hash_func(key) + offset) % self.bin_count

    def __getitem__(self, key, offset=0):
        index = self.__compression_func__(key, offset)
        if self.table[index] is not None:
            if self.table[index] != "DELETED" and self.table[index].key == key:
                return self.table[index].value
            else:
                return self.__getitem__(key, offset + 1)
        else:
            #  Control comes here when Key was Not found!
            raise Exception('Key not found in Hash Table!')

    def __setitem__(self, key, value, offset=0):
        index = self.__compression_func__(key, offset)
        if self.table[index] is None or self.table[index] == "DELETED":
            self.table[index] = KeyValuePair(key, value)
            self.total_elements += 1

            # Check load factor
            if self.total_elements > self.max_elements:
                # Doubling the size of hash table
                self.rebuild(self.bin_count * 2)
        else:
            # Bin not empty. Try for consecutive bin
            self.__setitem__(key, value, offset + 1)

    def __delitem__(self, key, offset=0):
        index = self.__compression_func__(key, offset)
        if self.table[index] is not None:
            if self.table[index] != "DELETED" and self.table[index].key == key:
                self.table[index] = "DELETED"
                self.total_elements -= 1
                return True
            else:
                self.__delitem__(key, offset + 1)
        return False

    def __contains__(self, key, offset=0):
        index = self.__compression_func__(key, offset)
        if self.table[index] is not None:
            if self.table[index] != "DELETED" and self.table[index].key == key:
                return True
            else:
                return self.__contains__(key, offset + 1)
        return False  # Key not found

    def __len__(self):
        return self.total_elements

    def display(self):
        print "Index | Value"  # Header
        for i in range(self.total_bins):
            print i, " | ", self.table[i]


class BinaryTreeNode(object):
    def __init__(self, data=None, left=None, right=None, parent=None):
        super(BinaryTreeNode, self).__init__()
        self.data = data
        self.left = left
        self.right = right
        self.parent = parent
        self.height = 0


class BinarySearchTreeDict(object):
    """
    >>> tree = BinarySearchTreeDict()
    >>> tree.__setitem__(1, 2)
    >>> len(tree)
    1

    >>> tree = BinarySearchTreeDict()
    >>> tree.__setitem__(1, 2)
    >>> tree.__setitem__(4, 3)
    >>> tree.root.data
    {key: 1, value: 2}

    >>> tree = BinarySearchTreeDict()
    >>> tree.__setitem__(1, 2)
    >>> tree.height
    1

    >>> tree = BinarySearchTreeDict()
    >>> tree.__setitem__(1, 2)
    >>> tree.__setitem__(4, 3)
    >>> tree.height
    2

    >>> tree = BinarySearchTreeDict()
    >>> tree.__setitem__(2, 2)
    >>> tree.__setitem__(4, 3)
    >>> tree.__setitem__(1, 3)
    >>> tree.height
    2

    >>> tree = BinarySearchTreeDict()
    >>> tree.__setitem__(2, 2)
    >>> tree.__setitem__(4, 3)
    >>> tree.__setitem__(1, 3)
    >>> tree.__setitem__(5, 3)
    >>> tree.height
    3

    >>> tree = BinarySearchTreeDict()
    >>> tree.__setitem__(2, 2)
    >>> type(tree.inorder_keys()).__name__
    'generator'

    >>> tree = BinarySearchTreeDict()
    >>> tree.__setitem__(2, 2)
    >>> type(tree.postorder_keys()).__name__
    'generator'

    >>> tree = BinarySearchTreeDict()
    >>> tree.__setitem__(2, 2)
    >>> type(tree.preorder_keys()).__name__
    'generator'

    >>> tree = BinarySearchTreeDict()
    >>> tree.__setitem__(2, 2)
    >>> type(tree.items()).__name__
    'generator'

    >>> tree = BinarySearchTreeDict()
    >>> tree.__setitem__(2, 2)
    >>> type(tree.search_node()).__name__
    'generator'

    >>> tree = BinarySearchTreeDict()
    >>> tree.__setitem__(2, 2)
    >>> tree.__setitem__(4, 3)
    >>> tree.__setitem__(1, 3)
    >>> tree.display()
    In-order Traversal: 1 2 4
    Pre-order Traversal: 2 1 4

    >>> tree = BinarySearchTreeDict()
    >>> tree.__setitem__(2, 10)
    >>> tree.__getitem__(2)
    10

    >>> tree = BinarySearchTreeDict()
    >>> tree.__setitem__(2, 10)
    >>> tree.__getitem__(11111)
    Traceback (most recent call last):
     ...
    Exception: Key not found!

    >>> tree = BinarySearchTreeDict()
    >>> tree.__setitem__(2, 10)
    >>> tree.__setitem__(3, 20)
    >>> tree.minimum_node().data
    {key: 2, value: 10}

    >>> tree = BinarySearchTreeDict()
    >>> tree.__setitem__(2, 10)
    >>> tree.__setitem__(2, 101)
    >>> tree.__setitem__(3, 20)
    >>> tree.__delitem__(2)
    Deleting node:  {key: 2, value: 10}
    Deleting node:  {key: 2, value: 101}

    >>> tree = BinarySearchTreeDict()
    >>> tree.__setitem__(2, 10)
    >>> tree.__setitem__(3, 101)
    >>> tree.__setitem__(4, 20)
    >>> tree.display()
    In-order Traversal: 2 3 4
    Pre-order Traversal: 2 3 4
    >>> tree.__delitem__(2)
    Deleting node:  {key: 2, value: 10}
    >>> tree.display()
    In-order Traversal: 3 4
    Pre-order Traversal: 3 4

    >>> tree = BinarySearchTreeDict()
    >>> tree.__setitem__(4, 10)
    >>> tree.__setitem__(3, 101)
    >>> tree.__setitem__(2, 20)
    >>> tree.__setitem__(1, 10)
    >>> tree.__setitem__(5, 101)
    >>> tree.__setitem__(6, 20)
    >>> tree.display()
    In-order Traversal: 1 2 3 4 5 6
    Pre-order Traversal: 4 3 2 1 5 6
    >>> tree.__delitem__(4)
    Deleting node:  {key: 4, value: 10}
    >>> tree.display()
    In-order Traversal: 1 2 3 5 6
    Pre-order Traversal: 5 3 2 1 6

    >>> tree = BinarySearchTreeDict()
    >>> tree.__setitem__(4, 10)
    >>> tree.__setitem__(3, 101)
    >>> tree.__setitem__(2, 20)
    >>> tree.__setitem__(1, 10)
    >>> tree.__setitem__(6, 101)
    >>> tree.__setitem__(5, 20)
    >>> tree.display()
    In-order Traversal: 1 2 3 4 5 6
    Pre-order Traversal: 4 3 2 1 6 5
    >>> tree.__delitem__(6)
    Deleting node:  {key: 6, value: 101}
    >>> tree.display()
    In-order Traversal: 1 2 3 4 5
    Pre-order Traversal: 4 3 2 1 5

    """
    def __init__(self):
        super(BinarySearchTreeDict, self).__init__()
        self.root = None
        self.total_nodes = 0
        self._height = 0

    @property
    def height(self):
        return self._height

    def __set_height__(self, h):
        if h > self._height:
            self._height = h

    def inorder_keys(self, pointer=None):
        if pointer is None:
            if self.root is not None:
                pointer = self.root
            else:
                raise StopIteration()

        if pointer.left is not None:
            for x in self.inorder_keys(pointer.left):
                yield x

        yield pointer.data.key

        if pointer.right is not None:
            for x in self.inorder_keys(pointer.right):
                yield x

    def postorder_keys(self, pointer=None):
        if pointer is None:
            if self.root is not None:
                pointer = self.root
            else:
                raise StopIteration()

        if pointer.left is not None:
            for x in self.postorder_keys(pointer.left):
                yield x

        if pointer.right is not None:
            for x in self.postorder_keys(pointer.right):
                yield x

        yield pointer.data.key

    def preorder_keys(self, pointer=None):
        if pointer is None:
            if self.root is not None:
                pointer = self.root
            else:
                raise StopIteration()

        yield pointer.data.key

        if pointer.left is not None:
            for x in self.preorder_keys(pointer.left):
                yield x

        if pointer.right is not None:
            for x in self.preorder_keys(pointer.right):
                yield x

    def items(self, pointer=None):
        # Using in-order traversal to yeild key,value pairs
        if pointer is None:
            if self.root is not None:
                pointer = self.root
            else:
                raise StopIteration()

        if pointer.left is not None:
            for x in self.items(pointer.left):
                yield x

        yield pointer.data

        if pointer.right is not None:
            for x in self.items(pointer.right):
                yield x

    def search_node(self, pointer=None):
        # Using in-order traversal to yeild ENTIRE NODE
        if pointer is None:
            if self.root is not None:
                pointer = self.root
            else:
                raise StopIteration()

        if pointer.left is not None:
            for x in self.search_node(pointer.left):
                yield x

        yield pointer

        if pointer.right is not None:
            for x in self.search_node(pointer.right):
                yield x

    def __getitem__(self, key):
        items = self.items()
        for i in items:
            if i.key == key:
                return i.value

        # Control comes over here when the key was not found.
        # Hence, exception raised.
        raise Exception("Key not found!")

    def __setitem__(self, key, value, pointer=None):
        if pointer is None:
            pointer = self.root
        pair = KeyValuePair(key, value)
        node = BinaryTreeNode(pair)
        self.total_nodes += 1
        if self.root is None:
            node.height = 1
            self.root = node
            self.__set_height__(node.height)
        else:
            if pointer.data.key <= key:
                if pointer.right is None:
                    pointer.right = node
                    node.parent = pointer
                    node.height = pointer.height + 1
                    self.__set_height__(node.height)
                else:
                    self.__setitem__(key, value, pointer.right)
            else:
                if pointer.left is None:
                    pointer.left = node
                    node.parent = pointer
                    node.height = pointer.height + 1
                    self.__set_height__(node.height)
                else:
                    self.__setitem__(key, value, pointer.left)

    def transplant(self, u, v):
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        if v is not None:
            v.parent = u.parent

    def minimum_node(self, node=None):
        if node is None:
            node = self.root
        while node.left is not None:
            node = node.left
        return node

    def __delitem__(self, key):
        nodes = self.search_node()
        delete_nodes = []
        for i in nodes:
            if i.data.key == key:
                delete_nodes += [i]

        for node in delete_nodes:
            # Delete all nodes with matching keys
            print "Deleting node: ", node.data
            if node.left is None:
                self.transplant(node, node.right)
            elif node.right is None:
                self.transplant(node, node.left)
            else:
                min = self.minimum_node(node.right)
                if min.parent != node:
                    self.transplant(min, min.right)
                    min.right = node.right
                    min.right.parent = min

                self.transplant(node, min)
                min.left = node.left
                min.left.parent = min

    def __contains__(self, key):
        items = self.items()
        for i in items:
            if i.key == key:
                return True

        # Control comes over here when the key was not found.
        return False

    def __len__(self):
        return self.total_nodes

    def display(self):
        inorder_keys = []
        preorder_keys = []
        for i in self.inorder_keys():
            inorder_keys += [str(i)]
        for i in self.preorder_keys():
            preorder_keys += [str(i)]

        print "In-order Traversal: " + " ".join(inorder_keys)
        print "Pre-order Traversal: " + " ".join(preorder_keys)


def terrible_hash(bin):
    """A terrible hash function that can be used for testing.

    A hash function should produce unpredictable results,
    but it is useful to see what happens to a hash table when
    you use the worst-possible hash function.  The function
    returned from this factory function will always return
    the same number, regardless of the key.

    :param bin:
        The result of the hash function, regardless of which
        item is used.

    :return:
        A python function that can be passes into the constructor
        of a hash table to use for hashing objects.
    """
    def hashfunc(item):
        return bin
    return hashfunc


def main():
    doctest.testmod()


if __name__ == '__main__':
    main()
