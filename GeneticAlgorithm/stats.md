# Statistiky experimentů

## Nastavení

- **Počet běhů:** 10 nezávislých běhů na konfiguraci
- **Rozpočet:** 100·D vyhodnocení účelové funkce
- **Velikost populace:** 20
- **Elitismus:** 20%
- **Pravděpodobnost mutace:** 0.01
- **Pravděpodobnost křížení:** 1.0

## OneMax

| D | Selekce | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |
|--:|:--------|---------:|---------:|-------:|-------:|----------:|
| 10 | roulette | 10 | 10 | 10.00 | 10.0 | 0.000 |
| 10 | rank | 10 | 10 | 10.00 | 10.0 | 0.000 |
| 30 | roulette | 30 | 30 | 30.00 | 30.0 | 0.000 |
| 30 | rank | 30 | 30 | 30.00 | 30.0 | 0.000 |
| 100 | roulette | 100 | 100 | 100.00 | 100.0 | 0.000 |
| 100 | rank | 100 | 100 | 100.00 | 100.0 | 0.000 |

## LeadingOnes

| D | Selekce | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |
|--:|:--------|---------:|---------:|-------:|-------:|----------:|
| 10 | roulette | 10 | 10 | 10.00 | 10.0 | 0.000 |
| 10 | rank | 10 | 10 | 10.00 | 10.0 | 0.000 |
| 30 | roulette | 30 | 26 | 29.50 | 30.0 | 1.269 |
| 30 | rank | 30 | 25 | 29.20 | 30.0 | 1.751 |
| 100 | roulette | 93 | 68 | 82.30 | 82.0 | 8.056 |
| 100 | rank | 99 | 70 | 82.00 | 82.0 | 8.769 |

Směrodatná odchylka je výběrová (ddof=1). Optimum je u obou úloh rovno D.
