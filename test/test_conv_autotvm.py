import numpy as np
import tvm
from tvm import te


def matmul(n, m, l):
    k = te.reduce_axis((0, l), name="k")
    A = te.placeholder((n, l), name="A")
    B = te.placeholder((l, m), name="B")
    C = te.compute((n, m),
                   lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    return A, B, C


n = 100
A, B, C = matmul(n, n, n)
s = te.create_schedule(C.op)
print(tvm.lower(s, [A, B], simple_mode=True))
mod = tvm.build(s, [A, B, C])
