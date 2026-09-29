# Srovnání nastavení

30 nezávislých běhů na konfiguraci, rozpočet 100·D vyhodnocení účelové funkce. Všechna nastavení běží se stejnými seedy.

## Srovnávaná nastavení

| Nastavení | pop_size | elite_ratio | p_mut | selection |
|:----------|--:|--:|--:|--:|
| A malá pop. + rank | 20 | 0.2 | 0.01 | rank |
| B větší populace | 50 | 0.1 | 0.01 | rank |
| C mutace 0.5 % | 20 | 0.2 | 0.005 | rank |
| D ruletová selekce | 20 | 0.2 | 0.01 | roulette |

## OneMax

| D | Nastavení | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |
|--:|:----------|---------:|---------:|-------:|-------:|----------:|
| 10 | A malá pop. + rank | 10 | 10 | 10.00 | 10.0 | 0.000 |
| 10 | B větší populace | 10 | 10 | 10.00 | 10.0 | 0.000 |
| 10 | C mutace 0.5 % | 10 | 10 | 10.00 | 10.0 | 0.000 |
| 10 | D ruletová selekce | 10 | 10 | 10.00 | 10.0 | 0.000 |
| 30 | A malá pop. + rank | 30 | 30 | 30.00 | 30.0 | 0.000 |
| 30 | B větší populace | 30 | 30 | 30.00 | 30.0 | 0.000 |
| 30 | C mutace 0.5 % | 30 | 30 | 30.00 | 30.0 | 0.000 |
| 30 | D ruletová selekce | 30 | 30 | 30.00 | 30.0 | 0.000 |
| 100 | A malá pop. + rank | 100 | 100 | 100.00 | 100.0 | 0.000 |
| 100 | B větší populace | 100 | 100 | 100.00 | 100.0 | 0.000 |
| 100 | C mutace 0.5 % | 100 | 100 | 100.00 | 100.0 | 0.000 |
| 100 | D ruletová selekce | 100 | 100 | 100.00 | 100.0 | 0.000 |

## LeadingOnes

| D | Nastavení | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |
|--:|:----------|---------:|---------:|-------:|-------:|----------:|
| 10 | A malá pop. + rank | 10 | 10 | 10.00 | 10.0 | 0.000 |
| 10 | B větší populace | 10 | 10 | 10.00 | 10.0 | 0.000 |
| 10 | C mutace 0.5 % | 10 | 9 | 9.87 | 10.0 | 0.346 |
| 10 | D ruletová selekce | 10 | 10 | 10.00 | 10.0 | 0.000 |
| 30 | A malá pop. + rank | 30 | 25 | 29.63 | 30.0 | 1.159 |
| 30 | B větší populace | 30 | 21 | 28.50 | 30.0 | 2.556 |
| 30 | C mutace 0.5 % | 30 | 14 | 25.40 | 26.5 | 4.917 |
| 30 | D ruletová selekce | 30 | 20 | 28.93 | 30.0 | 2.243 |
| 100 | A malá pop. + rank | 100 | 70 | 85.70 | 84.0 | 9.259 |
| 100 | B větší populace | 85 | 53 | 66.83 | 65.5 | 8.631 |
| 100 | C mutace 0.5 % | 88 | 52 | 68.73 | 68.5 | 9.399 |
| 100 | D ruletová selekce | 100 | 64 | 81.57 | 81.5 | 10.163 |

Směrodatná odchylka je výběrová (ddof=1). Optimum je u obou úloh rovno D.
