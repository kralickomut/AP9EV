"""GA pro funkce s reálnými parametry - binárně kódovaný i real-valued.

Binární varianta nepřináší nový algoritmus: bere run_ga z minulého úkolu
(GeneticAlgorithm/ga.py) a mění jen fitness - jedinec má 32*D bitů, před
výpočtem se po 32 bitech rozkóduje na D reálných čísel a dosadí do funkce.

Real-valued varianta má jedince rovnou jako D reálných čísel. Selekci,
elitismus i jednobodové křížení přebírá z ga.py beze změny, mění se jen
mutace.

Všechny testovací funkce se minimalizují, GA maximalizuje - fitness je
proto -f(x). Funguje to, protože pořadová selekce bere jen pořadí jedinců,
ne velikost fitness; ruleta by -f neunesla (potřebuje kladná čísla) a pro
ni by se hodilo 1 / (1 + f).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ga.py z minulého úkolu leží v sousední složce. Přidává se na konec cesty,
# aby experiment.py z této složky nepřebil ten minulý se stejným jménem.
sys.path.append(str(Path(__file__).resolve().parent.parent / "GeneticAlgorithm"))

from ga import one_point_crossover, run_ga, select_parents  # noqa: E402

from functions import TestFunction  # noqa: E402

D = 10
MAX_EVALS = 10_000


@dataclass
class RealResult:
    """Výsledek jednoho běhu, už v řeči minimalizace."""

    best_x: np.ndarray  # nejlepší nalezený bod
    best_f: float  # hodnota funkce v něm
    curve: np.ndarray  # curve[i] = nejlepší f po i+1 vyhodnoceních
    generations: int


def decode_in_bounds(rep, bits: np.ndarray, f: TestFunction) -> np.ndarray:
    """Rozkóduje bity na body uvnitř mezí funkce.

    Hodnoty mimo rozsah se ořežou na nejbližší mez. NaN (u IEEE 754 exponent
    samé jedničky s nenulovou mantisou) se bere jako nekonečno se svým
    znaménkem, takže skončí na příslušné mezi - nespadne to a jedinec
    jen dostane špatnou fitness.
    """
    x = rep.decode(bits)
    x = np.where(np.isnan(x), np.copysign(np.inf, x), x)
    return np.clip(x, f.lower, f.upper)


def run_binary_ga(
    f: TestFunction,
    rep,
    dim: int = D,
    max_evals: int = MAX_EVALS,
    pop_size: int = 20,
    elite_ratio: float = 0.2,
    p_mut: float | None = None,
    selection: str = "rank",
    rng: np.random.Generator | None = None,
) -> RealResult:
    """Binární GA z ga.py na reálné funkci v zadané reprezentaci.

    Args:
        f: Testovací funkce i s mezemi.
        rep: Instance reprezentace z representations.py (IEEE754, FixedPoint, BCD).
        dim: Počet reálných parametrů; chromozom má 32 * dim bitů.
        max_evals: Rozpočet vyhodnocení funkce.
        pop_size, elite_ratio, selection: Jako u ga.run_ga.
        p_mut: Pravděpodobnost překlopení bitu; None znamená 1 / délka chromozomu.
        rng: Generátor náhodných čísel.

    Returns:
        RealResult s nejlepším bodem a konvergenční křivkou hodnot f.
    """
    def fitness(bits: np.ndarray) -> np.ndarray:
        return -f(decode_in_bounds(rep, bits, f))

    result = run_ga(
        fitness,
        D=32 * dim,
        pop_size=pop_size,
        elite_ratio=elite_ratio,
        p_mut=p_mut,
        selection=selection,
        max_evals=max_evals,
        rng=rng,
    )
    return RealResult(
        best_x=decode_in_bounds(rep, result.best, f),
        best_f=-result.best_fitness,
        curve=-result.curve,
        generations=result.generations,
    )


def mutate_normal(
    x: np.ndarray,
    p_mut: float,
    sigma: float,
    f: TestFunction,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gaussovská mutace: každá souřadnice s pravděpodobností p_mut dostane x + N(0, sigma^2).

    Malé kroky jsou časté, velké vzácné - mutace hledá hlavně v okolí
    rodiče. Výsledek se ořízne do mezí funkce.
    """
    mask = rng.random(x.shape) < p_mut
    noise = rng.normal(0.0, sigma, x.shape)
    return np.clip(np.where(mask, x + noise, x), f.lower, f.upper)


def mutate_uniform(
    x: np.ndarray,
    p_mut: float,
    f: TestFunction,
    rng: np.random.Generator,
) -> np.ndarray:
    """Náhodná mutace: souřadnice s pravděpodobností p_mut dostane úplně novou hodnotu z celého rozsahu.

    Nová hodnota vůbec nezávisí na té staré - mutace neumí "doladit",
    jen přeskočit jinam.
    """
    mask = rng.random(x.shape) < p_mut
    fresh = rng.uniform(f.lower, f.upper, x.shape)
    return np.where(mask, fresh, x)


def run_real_ga(
    f: TestFunction,
    mutation: str = "normal",
    dim: int = D,
    max_evals: int = MAX_EVALS,
    pop_size: int = 20,
    elite_ratio: float = 0.2,
    p_mut: float = 0.1,
    sigma: float | None = None,
    selection: str = "rank",
    rng: np.random.Generator | None = None,
) -> RealResult:
    """Real-valued GA: jedinec je pole dim reálných čísel, bez kódování.

    Smyčka je stejná jako u ga.run_ga - elitismus, selekce dvou rodičů,
    jednobodové křížení mezi dimenzemi (řez padne mezi dvě souřadnice,
    hodnoty se nemíchají), mutace potomků, dokud není nová populace plná.

    Args:
        f: Testovací funkce i s mezemi.
        mutation: "normal" pro x + N(0, sigma^2), "uniform" pro náhodnou
            hodnotu z celého rozsahu.
        dim: Počet parametrů.
        max_evals: Rozpočet vyhodnocení funkce.
        pop_size, elite_ratio, selection: Jako u ga.run_ga.
        p_mut: Pravděpodobnost mutace jedné souřadnice.
        sigma: Směrodatná odchylka normální mutace (absolutní); None znamená
            1 % rozsahu funkce.
        rng: Generátor náhodných čísel.

    Returns:
        RealResult s nejlepším bodem a konvergenční křivkou hodnot f.

    Raises:
        ValueError: Pro neznámý typ mutace.
    """
    if mutation not in ("normal", "uniform"):
        raise ValueError(f"neznámá mutace {mutation!r}, znám: normal, uniform")

    rng = np.random.default_rng() if rng is None else rng
    sigma = 0.01 * f.range if sigma is None else sigma

    def mutate(x: np.ndarray) -> np.ndarray:
        if mutation == "normal":
            return mutate_normal(x, p_mut, sigma, f, rng)
        return mutate_uniform(x, p_mut, f, rng)

    n_elite = min(int(round(elite_ratio * pop_size)), pop_size - 2)

    population = rng.uniform(f.lower, f.upper, (pop_size, dim))
    values = f(population)
    evaluations = pop_size

    curve = np.empty(max_evals)
    curve[:pop_size] = np.minimum.accumulate(values)
    best_idx = int(np.argmin(values))
    best_x, best_f = population[best_idx].copy(), float(values[best_idx])

    generations = 0
    while evaluations < max_evals:
        elite_idx = np.argsort(values, kind="stable")[:n_elite]
        new_population = [population[i].copy() for i in elite_idx]
        new_values = [float(values[i]) for i in elite_idx]

        while len(new_population) < pop_size and evaluations < max_evals:
            # Selekce maximalizuje, proto -f.
            parents = select_parents(population, -values, 2, selection, rng)
            child1, child2 = one_point_crossover(parents[0], parents[1], rng)

            for child in (mutate(child1), mutate(child2)):
                if len(new_population) >= pop_size or evaluations >= max_evals:
                    break
                value = float(f(child))

                if value < best_f:
                    best_x, best_f = child.copy(), value

                curve[evaluations] = best_f
                evaluations += 1
                new_population.append(child)
                new_values.append(value)

        population = np.array(new_population)
        values = np.array(new_values)
        generations += 1

    return RealResult(best_x=best_x, best_f=best_f, curve=curve[:evaluations], generations=generations)
