"""Genetický algoritmus s binární reprezentací jedince.

Jedinec je chromozom o délce D, tedy vektor bitů 0/1. Celá populace je
jedna 2D matice tvaru (pop_size, D) — řádek = jedinec, sloupec = pozice genu.
"""

from __future__ import annotations

import numpy as np

# Populace: np.ndarray tvaru (pop_size, D), dtype uint8, hodnoty pouze 0/1.
Population = np.ndarray


def random_population(
    pop_size: int,
    D: int,
    rng: np.random.Generator | None = None,
) -> Population:
    """Vygeneruje náhodnou počáteční populaci.

    Každý bit je nezávisle 0 nebo 1 s pravděpodobností 0.5, takže je populace
    rovnoměrně rozprostřená po celém prohledávaném prostoru.

    Args:
        pop_size: Počet jedinců v populaci.
        D: Délka chromozomu (počet bitů jednoho jedince).
        rng: Generátor náhodných čísel. Předej vlastní se seedem, pokud
            chceš reprodukovatelný běh; jinak se použije nedeterministický.

    Returns:
        Matice tvaru (pop_size, D) s hodnotami 0/1.

    Raises:
        ValueError: Pokud pop_size nebo D není kladné.
    """
    if pop_size <= 0:
        raise ValueError(f"pop_size musí být kladné, dostal jsem {pop_size}")
    if D <= 0:
        raise ValueError(f"D musí být kladné, dostal jsem {D}")

    rng = np.random.default_rng() if rng is None else rng
    return rng.integers(0, 2, size=(pop_size, D), dtype=np.uint8)


def chromosome_to_str(chromosome: np.ndarray) -> str:
    """Převede chromozom na řetězec bitů kvůli čitelnému výpisu."""
    return "".join(str(bit) for bit in chromosome)


def main() -> None:
    pop_size, D = 10, 16

    rng = np.random.default_rng(seed=42)
    population = random_population(pop_size, D, rng)

    print(f"Populace: {population.shape[0]} jedinců, délka chromozomu {population.shape[1]}")
    for i, individual in enumerate(population):
        print(f"  {i:2d}: {chromosome_to_str(individual)}  (jedniček: {individual.sum()})")

    print(f"\nPodíl jedniček v populaci: {population.mean():.3f}  (očekáváno ~0.5)")


if __name__ == "__main__":
    main()
