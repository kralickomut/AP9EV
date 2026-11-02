"""Optimalizace rojem částic (PSO) - globální (hvězda) a kruhová topologie.

Každá částice má polohu x, rychlost v a svou nejlepší dosavadní polohu pbest.
V každé iteraci se rychlost a poloha aktualizují:

    v = w v + c1 r1 (pbest - x) + c2 r2 (lbest - x)
    x = x + v

kde r1, r2 jsou náhodná čísla z U[0, 1] (zvlášť pro každou souřadnici) a lbest
je nejlepší pbest v sousedství částice - u hvězdy celý roj (gbest), u kruhu
částice sama a N sousedů vlevo a N vpravo podle indexu (ne podle polohy).
Počáteční rychlost je nulová.

Rychlost se ořezává na +-vmax = 0.2 * rozsah. Částice smí vyletět mimo
prohledávanou oblast, tam se ale neohodnocuje (nespotřebuje ohodnocení a
nemění pbest), takže ji pbest a lbest časem přitáhnou zpátky.

Testovací funkce bere z GeneticAlgorithmRealParametres.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# functions.py leží v sousední složce. Přidává se na konec cesty, aby
# experiment.py z této složky nepřebil ten minulý se stejným jménem.
sys.path.append(str(Path(__file__).resolve().parent.parent / "GeneticAlgorithmRealParametres"))

from functions import TestFunction  # noqa: E402

D = 10
MAX_FES = 10_000
POP_SIZE = 40
VMAX_RATIO = 0.2  # vmax = 0.2 * rozsah


@dataclass
class PSOResult:
    """Výsledek jednoho běhu PSO."""

    best_x: np.ndarray  # nejlepší nalezený bod
    best_f: float  # hodnota funkce v něm
    curve: np.ndarray  # curve[i] = nejlepší f po i+1 ohodnoceních
    iterations: int
    outside: np.ndarray  # podíl částic mimo oblast v každé iteraci


def neighbourhood_best(pbest_f: np.ndarray, topology: str, size: int) -> np.ndarray:
    """Index nejlepšího pbest v sousedství každé částice.

    "star": všichni vidí všechny, takže všichni dostanou index gbest.
    "ring": částice i vidí i-size, ..., i, ..., i+size (s přetečením přes
    konec), tedy size sousedů na každou stranu.
    """
    n = len(pbest_f)
    if topology == "star":
        return np.full(n, int(np.argmin(pbest_f)))
    neighbours = (np.arange(n)[:, None] + np.arange(-size, size + 1)) % n
    return neighbours[np.arange(n), np.argmin(pbest_f[neighbours], axis=1)]


def run_pso(
    f: TestFunction,
    dim: int = D,
    max_fes: int = MAX_FES,
    pop_size: int = POP_SIZE,
    w: float | tuple[float, float] = 0.7,
    c1: float = 1.49618,
    c2: float = 1.49618,
    topology: str = "star",
    neighbourhood: int = 3,
    rng: np.random.Generator | None = None,
) -> PSOResult:
    """PSO s ořezem rychlosti a bez ohodnocení mimo prohledávanou oblast.

    Args:
        f: Testovací funkce i s mezemi.
        dim: Počet parametrů.
        max_fes: Rozpočet ohodnocení funkce (MAXFES), jediná ukončovací podmínka.
        pop_size: Velikost roje NP.
        w: Setrvačnost - číslo pro konstantní, dvojice (začátek, konec) pro
            lineárně klesající. Počet iterací předem neznám (částice mimo
            oblast ohodnocení nespotřebují), proto w klesá podle podílu
            spotřebovaných ohodnocení fes / max_fes.
        c1: Kognitivní koeficient (přitažlivost k vlastnímu pbest).
        c2: Sociální koeficient (přitažlivost k nejlepšímu v sousedství).
        topology: "star" (globální) nebo "ring" (kruh).
        neighbourhood: Velikost sousedství N u kruhu - počet sousedů na každou stranu.
        rng: Generátor náhodných čísel.

    Returns:
        PSOResult s nejlepším bodem a konvergenční křivkou.

    Raises:
        ValueError: Pro neznámou topologii.
    """
    if topology not in ("star", "ring"):
        raise ValueError(f"neznámá topologie {topology!r}, znám: star, ring")

    rng = np.random.default_rng() if rng is None else rng
    w_start, w_end = (w, w) if np.isscalar(w) else w
    vmax = VMAX_RATIO * f.range

    x = rng.uniform(f.lower, f.upper, (pop_size, dim))
    v = np.zeros((pop_size, dim))
    pbest = x.copy()
    pbest_f = f(x)
    fes = pop_size

    curve = np.empty(max_fes)
    curve[:pop_size] = np.minimum.accumulate(pbest_f)
    outside_history = []

    iterations = 0
    while fes < max_fes:
        inertia = w_start - (w_start - w_end) * fes / max_fes
        lbest = pbest[neighbourhood_best(pbest_f, topology, neighbourhood)]

        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        v = inertia * v + c1 * r1 * (pbest - x) + c2 * r2 * (lbest - x)
        v = np.clip(v, -vmax, vmax)
        x = x + v

        # Ohodnocují se jen částice uvnitř oblasti, a jen dokud zbývá rozpočet.
        inside = np.flatnonzero(np.all((x >= f.lower) & (x <= f.upper), axis=1))
        outside_history.append(1 - len(inside) / pop_size)
        inside = inside[: max_fes - fes]

        if len(inside):
            values = f(x[inside])
            better = values < pbest_f[inside]
            pbest[inside[better]] = x[inside[better]]
            pbest_f[inside[better]] = values[better]

            prev = curve[fes - 1]
            curve[fes:fes + len(inside)] = np.minimum(prev, np.minimum.accumulate(values))
            fes += len(inside)

        iterations += 1

    best = int(np.argmin(pbest_f))
    return PSOResult(
        best_x=pbest[best].copy(),
        best_f=float(pbest_f[best]),
        curve=curve,
        iterations=iterations,
        outside=np.array(outside_history),
    )
