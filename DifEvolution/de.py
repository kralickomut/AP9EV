"""Diferenciální evoluce - DE/rand/1/bin, DE/best/1/bin a jDE.

Všechny tři varianty sdílejí stejnou smyčku (MUTACE -> KŘÍŽENÍ -> SELEKCE):

1. mutace: z populace se vytvoří mutant v_i
   - rand/1: v_i = x_r1 + F (x_r2 - x_r3)
   - best/1: v_i = x_best + F (x_r1 - x_r2)
2. binomiální křížení: u_i,j = v_i,j, pokud rand <= CR nebo j = j_rand, jinak x_i,j
3. selekce: x_i do další generace nahradí u_i, pokud f(u_i) <= f(x_i)

jDE (Brest 2006) je rand/1/bin, kde má každý jedinec vlastní F a CR. Před
mutací se s pravděpodobností tau1 (tau2) přegenerují z U[F_l, F_u] (U[0, 1]);
do další generace přežijí jen spolu s úspěšným potomkem.

Testovací funkce bere z minulého úkolu (GeneticAlgorithmRealParametres).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# functions.py z minulého úkolu leží v sousední složce. Přidává se na konec
# cesty, aby experiment.py z této složky nepřebil ten minulý se stejným jménem.
sys.path.append(str(Path(__file__).resolve().parent.parent / "GeneticAlgorithmRealParametres"))

from functions import TestFunction  # noqa: E402

D = 10
MAX_EVALS = 10_000
POP_SIZE = 20  # přednáška: pro nízkodimenzionální problémy NP >= 20


@dataclass
class DEResult:
    """Výsledek jednoho běhu DE."""

    best_x: np.ndarray  # nejlepší nalezený bod
    best_f: float  # hodnota funkce v něm
    curve: np.ndarray  # curve[i] = nejlepší f po i+1 vyhodnoceních
    generations: int
    F_history: np.ndarray | None = None  # jDE: průměrné F populace po každé generaci
    CR_history: np.ndarray | None = None  # jDE: průměrné CR populace po každé generaci


def pick_others(i: int, n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """k navzájem různých indexů z 0..n-1, všechny různé od i."""
    idx = rng.choice(n - 1, size=k, replace=False)
    return idx + (idx >= i)  # posune indexy >= i o jedna, takže i se nikdy nevybere


def mutate(
    population: np.ndarray,
    i: int,
    best_idx: int,
    F: float,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mutační vektor v_i pro jedince i."""
    n = len(population)
    if strategy == "rand1":
        r1, r2, r3 = pick_others(i, n, 3, rng)
        return population[r1] + F * (population[r2] - population[r3])
    r1, r2 = pick_others(i, n, 2, rng)
    return population[best_idx] + F * (population[r1] - population[r2])


def binomial_crossover(x: np.ndarray, v: np.ndarray, CR: float, rng: np.random.Generator) -> np.ndarray:
    """Binomiální křížení: každá složka z mutanta s pravděpodobností CR.

    Složka j_rand se z mutanta bere vždy, takže potomek se od rodiče liší
    alespoň v jedné souřadnici i při CR = 0.
    """
    mask = rng.random(x.shape) <= CR
    mask[rng.integers(len(x))] = True
    return np.where(mask, v, x)


def run_de(
    f: TestFunction,
    strategy: str = "rand1",
    dim: int = D,
    max_evals: int = MAX_EVALS,
    pop_size: int = POP_SIZE,
    F: float = 0.5,
    CR: float = 0.8,
    jde: bool = False,
    tau1: float = 0.1,
    tau2: float = 0.1,
    F_bounds: tuple[float, float] = (0.1, 0.9),
    rng: np.random.Generator | None = None,
) -> DEResult:
    """Diferenciální evoluce s binomiálním křížením.

    Args:
        f: Testovací funkce i s mezemi.
        strategy: "rand1" nebo "best1".
        dim: Počet parametrů.
        max_evals: Rozpočet vyhodnocení funkce (včetně počáteční populace).
        pop_size: Velikost populace NP.
        F: Mutační konstanta; u jDE počáteční hodnota všech jedinců.
        CR: Práh křížení; u jDE počáteční hodnota všech jedinců.
        jde: Zapne samoadaptaci F a CR podle jDE.
        tau1, tau2: Pravděpodobnost přegenerování F, resp. CR (jen jDE).
        F_bounds: Interval [F_l, F_u], ze kterého se generuje nové F (jen jDE).
        rng: Generátor náhodných čísel.

    Returns:
        DEResult s nejlepším bodem a konvergenční křivkou.

    Raises:
        ValueError: Pro neznámou strategii nebo jDE s jinou strategií než rand1.
    """
    if strategy not in ("rand1", "best1"):
        raise ValueError(f"neznámá strategie {strategy!r}, znám: rand1, best1")
    if jde and strategy != "rand1":
        raise ValueError("jDE je navržená jen pro rand/1/bin")

    rng = np.random.default_rng() if rng is None else rng
    F_low, F_high = F_bounds

    population = rng.uniform(f.lower, f.upper, (pop_size, dim))
    values = f(population)
    evaluations = pop_size

    # Každý jedinec má své F a CR; u klasické DE jsou všechna stejná a nemění se.
    Fs = np.full(pop_size, F)
    CRs = np.full(pop_size, CR)
    F_history, CR_history = [], []

    curve = np.empty(max_evals)
    curve[:pop_size] = np.minimum.accumulate(values)
    best_idx = int(np.argmin(values))

    generations = 0
    while evaluations < max_evals:
        # Nová generace se skládá zvlášť - mutanti se tvoří jen ze staré populace.
        new_population = population.copy()
        new_values = values.copy()
        new_Fs, new_CRs = Fs.copy(), CRs.copy()

        for i in range(pop_size):
            if evaluations >= max_evals:
                break

            F_i, CR_i = Fs[i], CRs[i]
            if jde:
                if rng.random() < tau1:
                    F_i = F_low + rng.random() * (F_high - F_low)
                if rng.random() < tau2:
                    CR_i = rng.random()

            v = mutate(population, i, best_idx, F_i, strategy, rng)
            u = np.clip(binomial_crossover(population[i], v, CR_i, rng), f.lower, f.upper)
            value = float(f(u))
            evaluations += 1

            if value <= values[i]:
                new_population[i], new_values[i] = u, value
                new_Fs[i], new_CRs[i] = F_i, CR_i  # u jDE přežijí jen úspěšné parametry

            curve[evaluations - 1] = min(curve[evaluations - 2], value)

        population, values, Fs, CRs = new_population, new_values, new_Fs, new_CRs
        best_idx = int(np.argmin(values))
        F_history.append(Fs.mean())
        CR_history.append(CRs.mean())
        generations += 1

    return DEResult(
        best_x=population[best_idx].copy(),
        best_f=float(values[best_idx]),
        curve=curve[:evaluations],
        generations=generations,
        F_history=np.array(F_history) if jde else None,
        CR_history=np.array(CR_history) if jde else None,
    )


def run_jde(f: TestFunction, **kwargs) -> DEResult:
    """jDE s doporučeným nastavením: F = 0.5, CR = 0.9, tau1 = tau2 = 0.1, rand/1/bin."""
    kwargs = {"F": 0.5, "CR": 0.9, **kwargs}
    return run_de(f, "rand1", jde=True, **kwargs)
