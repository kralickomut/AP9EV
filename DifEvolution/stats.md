# Srovnání variant DE na reálných funkcích

10 nezávislých běhů na variantu, D = 10, NP = 20, rozpočet 10000 ohodnocení.

## Nastavení

- **DE/rand/1/bin, DE/best/1/bin:** F = 0.5, CR = 0.8
- **jDE:** rand/1/bin, počáteční F = 0.5, CR = 0.9, τ1 = τ2 = 0.1, F ∈ [0.1, 0.9], CR ∈ [0, 1]
- Selekce f(u) ≤ f(x), souřadnice mimo meze se ořežou na mez.

## Sphere

Rozsah [-5.12, 5.12], globální minimum f = 0 v x_i = 0.

| Varianta | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |
|:---------|---------:|---------:|-------:|-------:|----------:|
| DE/rand/1/bin | 8.032e-25 | 7.695e-22 | 1.506e-22 | 2.603e-23 | 2.740e-22 |
| DE/best/1/bin | 1.658e-02 | 1.530e+00 | 4.442e-01 | 2.676e-01 | 5.077e-01 |
| jDE | 3.597e-25 | 8.386e-22 | 8.823e-23 | 2.423e-24 | 2.637e-22 |

## Rosenbrock

Rozsah [-5, 10], globální minimum f = 0 v x_i = 1.

| Varianta | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |
|:---------|---------:|---------:|-------:|-------:|----------:|
| DE/rand/1/bin | 3.124e-01 | 8.304e+00 | 6.192e+00 | 6.479e+00 | 2.190e+00 |
| DE/best/1/bin | 4.486e+00 | 1.612e+03 | 3.858e+02 | 1.324e+02 | 5.488e+02 |
| jDE | 1.718e-01 | 7.285e+00 | 4.206e+00 | 4.097e+00 | 2.162e+00 |

## Schwefel

Rozsah [-500, 500], globální minimum f = 0 v x_i = 420.969.

| Varianta | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |
|:---------|---------:|---------:|-------:|-------:|----------:|
| DE/rand/1/bin | 1.733e+02 | 8.399e+02 | 4.718e+02 | 4.062e+02 | 2.557e+02 |
| DE/best/1/bin | 4.555e+02 | 1.594e+03 | 1.090e+03 | 1.073e+03 | 3.269e+02 |
| jDE | 1.273e-04 | 2.369e+02 | 1.421e+02 | 1.184e+02 | 9.343e+01 |

Nejlepší = nejmenší hodnota f. Směrodatná odchylka je výběrová (ddof=1).
