import cProfile
import pstats
import numpy as np
import mnist_conv_snn as m

pr = cProfile.Profile()
pr.enable()
for s in range(10):
    x = m.tr_flat[s]
    yv = m.y_onehot[m.trl[s]]
    for _ in range(10):
        m.spiking_step(x, yv, True)
pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(14)
