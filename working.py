import qcfpga
import time

s = qcfpga.State(20)
h = qcfpga.gate.h()

s.apply_all(h)



print(s.measure(1000))
# print(s.backend.measure(samples=10000))
