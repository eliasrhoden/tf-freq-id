# Transfer function identification from frequency data

Fits Transfer function of the form

$$
G(s) = k \dfrac{\prod G_2(s)}{\prod G_2(s)}\dfrac{\prod G_1(s)}{\prod G_1(s)}
$$

$$
G_1(s) = s \tau + 1
$$

$$
G_2(s) = \frac{s^2 + 2\delta\omega_c s + \omega_c^2}{\omega_c^2}
$$

