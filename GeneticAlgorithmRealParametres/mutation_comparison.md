# Experimenty s normální mutací

## Vliv σ normální mutace

Real-valued GA, p_mut = 0.1, 10 běhů, D = 10, 10000 ohodnocení. σ je zadané jako podíl rozsahu funkce, aby šly funkce porovnat.

### Sphere

Rozsah [-5.12, 5.12], globální minimum f = 0 v x_i = 0.

| Varianta | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |
|:---------|---------:|---------:|-------:|-------:|----------:|
| σ = 0.001·rozsah (malá) | 5.273e-02 | 8.200e+00 | 2.431e+00 | 2.092e+00 | 2.503e+00 |
| σ = 0.01·rozsah (střední) | 3.353e-06 | 3.696e-05 | 1.181e-05 | 9.813e-06 | 9.544e-06 |
| σ = 0.1·rozsah (velká) | 3.418e-04 | 2.080e-03 | 9.707e-04 | 9.253e-04 | 5.769e-04 |

### Rosenbrock

Rozsah [-5, 10], globální minimum f = 0 v x_i = 1.

| Varianta | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |
|:---------|---------:|---------:|-------:|-------:|----------:|
| σ = 0.001·rozsah (malá) | 1.296e+02 | 2.708e+04 | 4.511e+03 | 1.752e+03 | 8.154e+03 |
| σ = 0.01·rozsah (střední) | 1.105e+00 | 2.176e+02 | 2.750e+01 | 7.230e+00 | 6.687e+01 |
| σ = 0.1·rozsah (velká) | 6.822e-01 | 1.457e+01 | 6.396e+00 | 7.026e+00 | 3.740e+00 |

### Schwefel

Rozsah [-500, 500], globální minimum f = 0 v x_i = 420.969.

| Varianta | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |
|:---------|---------:|---------:|-------:|-------:|----------:|
| σ = 0.001·rozsah (malá) | 7.699e+02 | 1.540e+03 | 1.128e+03 | 1.086e+03 | 2.412e+02 |
| σ = 0.01·rozsah (střední) | 7.699e+02 | 1.304e+03 | 1.011e+03 | 9.969e+02 | 2.039e+02 |
| σ = 0.1·rozsah (velká) | 1.192e+02 | 8.302e+02 | 4.155e+02 | 3.569e+02 | 1.955e+02 |

Nejlepší = nejmenší hodnota f. Směrodatná odchylka je výběrová (ddof=1).

## Vliv pravděpodobnosti mutace

Real-valued GA, σ = 0.01·rozsah, 10 běhů, D = 10, 10000 ohodnocení.

### Sphere

Rozsah [-5.12, 5.12], globální minimum f = 0 v x_i = 0.

| Varianta | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |
|:---------|---------:|---------:|-------:|-------:|----------:|
| p_mut = 0.05 | 4.936e-06 | 1.867e-05 | 1.100e-05 | 9.048e-06 | 4.848e-06 |
| p_mut = 0.1 | 3.353e-06 | 3.696e-05 | 1.181e-05 | 9.813e-06 | 9.544e-06 |
| p_mut = 0.3 | 6.790e-05 | 3.786e-04 | 2.080e-04 | 2.130e-04 | 9.716e-05 |

### Rosenbrock

Rozsah [-5, 10], globální minimum f = 0 v x_i = 1.

| Varianta | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |
|:---------|---------:|---------:|-------:|-------:|----------:|
| p_mut = 0.05 | 8.282e-01 | 2.379e+02 | 4.006e+01 | 8.683e+00 | 7.643e+01 |
| p_mut = 0.1 | 1.105e+00 | 2.176e+02 | 2.750e+01 | 7.230e+00 | 6.687e+01 |
| p_mut = 0.3 | 1.268e+00 | 9.608e+00 | 6.862e+00 | 7.474e+00 | 2.567e+00 |

### Schwefel

Rozsah [-500, 500], globální minimum f = 0 v x_i = 420.969.

| Varianta | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |
|:---------|---------:|---------:|-------:|-------:|----------:|
| p_mut = 0.05 | 8.094e+02 | 1.342e+03 | 1.116e+03 | 1.136e+03 | 1.780e+02 |
| p_mut = 0.1 | 7.699e+02 | 1.304e+03 | 1.011e+03 | 9.969e+02 | 2.039e+02 |
| p_mut = 0.3 | 8.688e+02 | 1.305e+03 | 1.134e+03 | 1.166e+03 | 1.459e+02 |

Nejlepší = nejmenší hodnota f. Směrodatná odchylka je výběrová (ddof=1).
