# Srovnání variant PSO na reálných funkcích

10 nezávislých běhů na variantu, D = 10, NP = 40, MAXFES = 10000.

## Nastavení

- **w 0.8→0.3, hvězda:** setrvačnost lineárně klesá z 0.8 na 0.3, c1 = c2 = 2, globální topologie
- **w = 0.7, hvězda:** c1 = c2 = 1.49618, globální topologie
- **w = 0.7, kruh (N = 3):** c1 = c2 = 1.49618, kruhová topologie, 3 sousedé na každou stranu
- Všechny varianty: v(0) = 0, vmax = 0.2 · rozsah, částice mimo oblast se neohodnocují.

## Sphere

Rozsah [-5.12, 5.12], globální minimum f = 0 v x_i = 0.

| Varianta | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |
|:---------|---------:|---------:|-------:|-------:|----------:|
| w 0.8→0.3, hvězda | 2.850e-13 | 1.756e-11 | 4.052e-12 | 1.830e-12 | 5.375e-12 |
| w = 0.7, hvězda | 1.577e-16 | 3.170e-15 | 9.911e-16 | 8.573e-16 | 8.677e-16 |
| w = 0.7, kruh (N = 3) | 2.841e-12 | 9.677e-11 | 2.755e-11 | 1.583e-11 | 3.193e-11 |

## Rosenbrock

Rozsah [-5, 10], globální minimum f = 0 v x_i = 1.

| Varianta | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |
|:---------|---------:|---------:|-------:|-------:|----------:|
| w 0.8→0.3, hvězda | 2.665e-02 | 6.932e+00 | 4.644e+00 | 5.690e+00 | 2.514e+00 |
| w = 0.7, hvězda | 1.222e+00 | 9.076e+00 | 4.604e+00 | 4.290e+00 | 1.949e+00 |
| w = 0.7, kruh (N = 3) | 2.151e+00 | 7.389e+00 | 5.153e+00 | 5.574e+00 | 1.465e+00 |

## Schwefel

Rozsah [-500, 500], globální minimum f = 0 v x_i = 420.969.

| Varianta | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |
|:---------|---------:|---------:|-------:|-------:|----------:|
| w 0.8→0.3, hvězda | 4.738e+02 | 1.303e+03 | 9.317e+02 | 9.080e+02 | 2.437e+02 |
| w = 0.7, hvězda | 6.909e+02 | 1.599e+03 | 1.080e+03 | 1.056e+03 | 3.027e+02 |
| w = 0.7, kruh (N = 3) | 5.922e+02 | 1.520e+03 | 1.196e+03 | 1.204e+03 | 2.542e+02 |

Nejlepší = nejmenší hodnota f. Směrodatná odchylka je výběrová (ddof=1).
