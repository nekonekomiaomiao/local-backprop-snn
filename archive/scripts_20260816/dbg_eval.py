import numpy as np
import mnist_conv_snn as m

z = np.load("mnist_checkpoint.npz")
m.P[:] = z["P"]
m.u[:] = 0
m.r_est[:] = 0
m.f_est[:] = 0
m.E1[:] = 0
m.E2[:] = 0
m.E3[:] = 0
m.E4[:] = 0

# 训练序列表的前 10 张 + 各自 25 步
for ii in m.order[:10]:
    x = m.tr_flat[ii]
    y = m.trl[ii]
    for _ in range(m.steps):
        m.spiking_step(x, m.y_onehot[y], learn=False)
    f = m.f_est / m.TAU_F
    print(f"img {ii} label {y}: out = {np.round(f,1)}  pred {np.argmax(f)}")
