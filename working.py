import qcfpga
import time

s = qcfpga.State(3)
h = qcfpga.gate.h()
sqrt_x = qcfpga.gate.sqrt_x()
#s.apply_all(h)
s.apply_all(sqrt_x)

print(s.measure(1000))
# print(s.backend.measure(samples=10000))
