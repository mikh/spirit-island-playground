

class A():
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def match(self, X):
        if self.a == X.a and self.b == X.b and self.c ==  X.c:
            return True
        return False

A1 = A(1,2,3)
A2 = A(2,3,4)
A3 = A(3,4,5)

L = [A1, A2, A3]

A4 = A(1,2,3)

print("A4 in L", A4 in L)
print("A1 in L", A1 in L)

L1 = [1,2,3]
L2 = [1,2,3]
print('L1 == L2', L1 == L2)

L3 = [[1,2], [2, 3], [3,4]]
L4 = [3,5]
L5 = [2,3]
print('L4 in L3', L4 in L3)
print('L5 in L3', L5 in L3)
