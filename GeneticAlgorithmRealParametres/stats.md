# Srovnání variant GA na reálných funkcích

10 nezávislých běhů na variantu, D = 10, rozpočet 10000 ohodnocení.

## Nastavení

- **Společné:** populace 20, elitismus 20%, rank selekce, jednobodové křížení vždy
- **Binární varianty (IEEE 754, Fixed-point, BCD):** 32 bitů na dimenzi, chromozom 320 bitů, pravděpodobnost překlopení bitu 0.01
- **Real-valued:** pravděpodobnost mutace souřadnice 0.1, normální mutace se σ = 1% rozsahu funkce

## Sphere

Rozsah [-5.12, 5.12], globální minimum f = 0 v x_i = 0.

| Varianta | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |
|:---------|---------:|---------:|-------:|-------:|----------:|
| IEEE 754 | 3.219e-83 | 1.194e-80 | 2.435e-81 | 9.820e-82 | 3.571e-81 |
| Fixed-point | 3.853e-09 | 2.450e-07 | 5.980e-08 | 1.677e-08 | 8.975e-08 |
| BCD | 7.062e-08 | 2.118e-06 | 1.008e-06 | 1.249e-06 | 7.043e-07 |
| Real + N(0, σ²) | 3.353e-06 | 3.696e-05 | 1.181e-05 | 9.813e-06 | 9.544e-06 |
| Real + náhodná | 3.343e-03 | 2.811e-02 | 1.110e-02 | 7.213e-03 | 8.872e-03 |

## Rosenbrock

Rozsah [-5, 10], globální minimum f = 0 v x_i = 1.

| Varianta | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |
|:---------|---------:|---------:|-------:|-------:|----------:|
| IEEE 754 | 8.667e+00 | 8.933e+00 | 8.736e+00 | 8.698e+00 | 1.036e-01 |
| Fixed-point | 8.961e+00 | 8.255e+02 | 1.212e+02 | 1.238e+01 | 2.525e+02 |
| BCD | 5.590e+01 | 2.704e+05 | 8.129e+04 | 5.484e+04 | 1.016e+05 |
| Real + N(0, σ²) | 1.105e+00 | 2.176e+02 | 2.750e+01 | 7.230e+00 | 6.687e+01 |
| Real + náhodná | 7.540e+00 | 8.181e+01 | 3.432e+01 | 1.795e+01 | 3.074e+01 |

## Schwefel

Rozsah [-500, 500], globální minimum f = 0 v x_i = 420.969.

| Varianta | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |
|:---------|---------:|---------:|-------:|-------:|----------:|
| IEEE 754 | 1.790e+03 | 2.264e+03 | 2.020e+03 | 2.025e+03 | 1.280e+02 |
| Fixed-point | 2.497e+02 | 9.938e+02 | 6.885e+02 | 7.154e+02 | 2.039e+02 |
| BCD | 8.630e+02 | 1.244e+03 | 9.855e+02 | 9.565e+02 | 1.290e+02 |
| Real + N(0, σ²) | 7.699e+02 | 1.304e+03 | 1.011e+03 | 9.969e+02 | 2.039e+02 |
| Real + náhodná | 1.459e+00 | 2.155e+01 | 6.884e+00 | 5.013e+00 | 6.256e+00 |

Nejlepší = nejmenší hodnota f. Směrodatná odchylka je výběrová (ddof=1).
