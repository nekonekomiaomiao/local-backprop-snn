import numpy as np
import mnist_conv_snn as m

acc = np.zeros(m.N_NEURONS)
m.u[:] = 0
m.r_est[:] = 0
m.f_est[:] = 0
m.E1[:] = 0
m.E2[:] = 0
m.E3[:] = 0
m.E4[:] = 0

x = m.tr_flat[5]
yv = m.y_onehot[m.trl[5]]
for _ in range(200):
    m.spiking_step(x, yv, learn=False)

rates = m.r_est / m.TAU_R
print("label:", m.trl[5])
print("L1: mean %.1f  max %.1f  active %d/%d" % (rates[m.L1].mean(), rates[m.L1].max(), (rates[m.L1] > 1).sum(), m.N_L1))
print("P1: mean %.1f  max %.1f  active %d/%d" % (rates[m.P1].mean(), rates[m.P1].max(), (rates[m.P1] > 1).sum(), m.NP1))
print("L2: mean %.1f  max %.1f  active %d/%d" % (rates[m.L2].mean(), rates[m.L2].max(), (rates[m.L2] > 1).sum(), m.N_L2))
print("P2: mean %.1f  max %.1f  active %d/%d" % (rates[m.P2].mean(), rates[m.P2].max(), (rates[m.P2] > 1).sum(), m.NP2))
print("FC: mean %.1f  max %.1f  active %d/%d" % (rates[m.FC].mean(), rates[m.FC].max(), (rates[m.FC] > 1).sum(), m.NFC))
print("OUT:", np.round(rates[m.OUT], 1))
print("f_est/tau_f:", np.round(m.f_est / m.TAU_F, 1))
print("E1 nonzero:", int((m.E1 > 0.01).sum()), "/", m.G1)
print("E4 nonzero:", int((m.E4 > 0.01).sum()), "/", m.G4 - m.G3)
