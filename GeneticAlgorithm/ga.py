"""Genetický algoritmus s binární reprezentací jedince.

Jedinec je chromozom o délce D, tedy vektor bitů 0/1. Celá populace je
jedna 2D matice tvaru (pop_size, D) — řádek = jedinec, sloupec = pozice genu.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

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


def one_max(population: Population) -> np.ndarray:
    """OneMax: fitness = počet jedniček v chromozomu.

    Optimum je D (samé jedničky). Každý bit přispívá nezávisle na ostatních,
    takže jde o nejjednodušší možnou testovací úlohu.

    Args:
        population: Jeden chromozom (1D) nebo celá populace (2D).

    Returns:
        Skalár pro 1D vstup, vektor délky pop_size pro 2D vstup.
    """
    return population.sum(axis=-1, dtype=np.int64)


def leading_ones(population: Population) -> np.ndarray:
    """LeadingOnes: fitness = počet jedniček na začátku, než přijde první nula.

    Např. 1110100 má fitness 3. Optimum je opět D, ale bit na pozici i se
    do fitness započítá jen tehdy, když jsou všechny bity před ním jedničky.

    Args:
        population: Jeden chromozom (1D) nebo celá populace (2D).

    Returns:
        Skalár pro 1D vstup, vektor délky pop_size pro 2D vstup.
    """
    # Kumulativní součin je 1, dokud jsou samé jedničky, a od první nuly
    # už zůstane 0 — jeho součet je přesně délka úvodního bloku jedniček.
    return np.cumprod(population, axis=-1, dtype=np.int64).sum(axis=-1)


def one_point_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Jednobodové křížení dvou rodičů.

    Náhodně se zvolí bod řezu, chromozomy se v něm rozdělí a prohodí se
    jejich první části. Vzniknou dva potomci, kteří dohromady obsahují
    přesně tytéž geny jako rodiče, jen jinak poskládané::

        rodič 1:  1011 | 0100     potomek 1:  1101 0100
        rodič 2:  1101 | 0011     potomek 2:  1011 0011

    Args:
        parent1: Chromozom prvního rodiče (1D, délka D).
        parent2: Chromozom druhého rodiče (1D, stejná délka).
        rng: Generátor náhodných čísel; bez něj se použije nedeterministický.

    Returns:
        Dvojice nových polí — rodiče zůstávají nezměněni.

    Raises:
        ValueError: Pokud rodiče nemají stejnou délku nebo jsou kratší než 2 bity.
    """
    if parent1.shape != parent2.shape:
        raise ValueError(
            f"rodiče musí mít stejný tvar, dostal jsem {parent1.shape} a {parent2.shape}"
        )
    if parent1.ndim != 1:
        raise ValueError(f"čekám jednotlivé chromozomy (1D), dostal jsem {parent1.ndim}D")

    D = parent1.size
    if D < 2:
        raise ValueError(f"chromozom musí mít aspoň 2 bity, má {D}")

    rng = np.random.default_rng() if rng is None else rng

    # Bod řezu 1..D-1: kdyby padla 0 nebo D, potomci by byli kopie rodičů.
    point = int(rng.integers(1, D))

    child1 = np.concatenate((parent2[:point], parent1[point:]))
    child2 = np.concatenate((parent1[:point], parent2[point:]))
    return child1, child2


def mutate(
    individual: np.ndarray,
    p_mut: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Bitová mutace: každý bit se nezávisle invertuje s pravděpodobností p_mut.

    Mutace je jediný operátor, který umí do populace vnést bit, jenž v ní
    dosud nebyl — drží tedy diverzitu a je pojistkou proti uváznutí
    v lokálním optimu. Rozumná volba je p_mut kolem 1/D, tedy v průměru
    jeden překlopený bit na jedince.

    Args:
        individual: Chromozom (1D) nebo celá populace (2D) k mutaci.
        p_mut: Pravděpodobnost překlopení jednoho bitu, typicky 0.005-0.01.
        rng: Generátor náhodných čísel; bez něj se použije nedeterministický.

    Returns:
        Nové pole stejného tvaru — vstup zůstává nezměněn.

    Raises:
        ValueError: Pokud p_mut není z intervalu <0, 1>.
    """
    if not 0.0 <= p_mut <= 1.0:
        raise ValueError(f"p_mut musí být z intervalu <0, 1>, dostal jsem {p_mut}")

    rng = np.random.default_rng() if rng is None else rng

    # Maska říká, které bity se překlápí; XOR s 1 invertuje, XOR s 0 nechá být.
    flips = (rng.random(individual.shape) < p_mut).astype(individual.dtype)
    return individual ^ flips


def _sample_parents(
    population: Population,
    probabilities: np.ndarray,
    n_parents: int,
    rng: np.random.Generator,
) -> Population:
    """Vylosuje n_parents jedinců podle zadaných pravděpodobností (s opakováním)."""
    indices = rng.choice(len(population), size=n_parents, replace=True, p=probabilities)
    return population[indices].copy()


def roulette_selection(
    population: Population,
    fitness: np.ndarray,
    n_parents: int | None = None,
    rng: np.random.Generator | None = None,
) -> Population:
    """Ruletová selekce: pravděpodobnost výběru je úměrná fitness.

    Jedinec s dvojnásobnou fitness má dvojnásobnou šanci stát se rodičem.
    Vyžaduje nezáporné fitness — pro záporné hodnoty použij rank_selection.
    Pokud mají všichni fitness 0, degraduje na rovnoměrný výběr.

    Args:
        population: Populace tvaru (pop_size, D).
        fitness: Fitness jednotlivých jedinců, délka pop_size.
        n_parents: Kolik rodičů vybrat; None znamená stejně jako pop_size.
        rng: Generátor náhodných čísel.

    Returns:
        Vybraní rodiče tvaru (n_parents, D); jedinec se může opakovat.

    Raises:
        ValueError: Pokud fitness obsahuje zápornou hodnotu nebo nesedí délky.
    """
    population, fitness, n_parents, rng = _check_selection_args(
        population, fitness, n_parents, rng
    )

    if np.any(fitness < 0):
        raise ValueError(
            "ruletová selekce neumí zápornou fitness - posuň hodnoty do kladných "
            "čísel, nebo použij rank_selection"
        )

    total = fitness.sum()
    if total == 0:
        # Všichni stejně (ne)dobří, např. LeadingOnes na začátku běhu.
        probabilities = np.full(len(population), 1.0 / len(population))
    else:
        probabilities = fitness / total

    return _sample_parents(population, probabilities, n_parents, rng)


def rank_selection(
    population: Population,
    fitness: np.ndarray,
    n_parents: int | None = None,
    rng: np.random.Generator | None = None,
) -> Population:
    """Pořadová selekce: pravděpodobnost je úměrná pořadí, ne fitness.

    Jedinci se seřadí podle fitness a dostanou pořadí 1 (nejhorší) až N
    (nejlepší); pravděpodobnost výběru je rank / součet ranků. Absolutní
    hodnoty fitness se tím zahodí, takže selekční tlak nezávisí na jejich
    rozsahu - zvládne i zápornou fitness a neublíží jí jeden superjedinec.

    Args:
        population: Populace tvaru (pop_size, D).
        fitness: Fitness jednotlivých jedinců, délka pop_size.
        n_parents: Kolik rodičů vybrat; None znamená stejně jako pop_size.
        rng: Generátor náhodných čísel.

    Returns:
        Vybraní rodiče tvaru (n_parents, D); jedinec se může opakovat.

    Raises:
        ValueError: Pokud nesedí délky nebo je populace prázdná.
    """
    population, fitness, n_parents, rng = _check_selection_args(
        population, fitness, n_parents, rng
    )

    pop_size = len(population)
    # argsort vrátí indexy od nejhoršího; na pozici jedince zapíšeme jeho pořadí.
    ranks = np.empty(pop_size, dtype=np.int64)
    ranks[np.argsort(fitness, kind="stable")] = np.arange(1, pop_size + 1)

    probabilities = ranks / ranks.sum()
    return _sample_parents(population, probabilities, n_parents, rng)


# Registr selekčních metod - přepínáš mezi nimi jménem přes select_parents().
SELECTION_METHODS = {
    "roulette": roulette_selection,
    "rank": rank_selection,
}


def select_parents(
    population: Population,
    fitness: np.ndarray,
    n_parents: int | None = None,
    method: str = "roulette",
    rng: np.random.Generator | None = None,
) -> Population:
    """Vybere rodiče zvolenou metodou.

    Args:
        population: Populace tvaru (pop_size, D).
        fitness: Fitness jednotlivých jedinců, délka pop_size.
        n_parents: Kolik rodičů vybrat; None znamená stejně jako pop_size.
        method: Klíč ze SELECTION_METHODS, tedy "roulette" nebo "rank".
        rng: Generátor náhodných čísel.

    Returns:
        Vybraní rodiče tvaru (n_parents, D).

    Raises:
        ValueError: Pokud je method neznámá metoda.
    """
    if method not in SELECTION_METHODS:
        known = ", ".join(sorted(SELECTION_METHODS))
        raise ValueError(f"neznámá selekce {method!r}, znám: {known}")

    return SELECTION_METHODS[method](population, fitness, n_parents, rng)


def _check_selection_args(
    population: Population,
    fitness: np.ndarray,
    n_parents: int | None,
    rng: np.random.Generator | None,
) -> tuple[Population, np.ndarray, int, np.random.Generator]:
    """Společná kontrola vstupů selekcí a doplnění výchozích hodnot."""
    population = np.asarray(population)
    fitness = np.asarray(fitness, dtype=np.float64)

    if population.ndim != 2:
        raise ValueError(f"čekám populaci (2D), dostal jsem {population.ndim}D")
    if len(population) == 0:
        raise ValueError("populace je prázdná")
    if fitness.shape != (len(population),):
        raise ValueError(
            f"fitness musí mít délku {len(population)}, má {fitness.shape}"
        )

    n_parents = len(population) if n_parents is None else n_parents
    if n_parents <= 0:
        raise ValueError(f"n_parents musí být kladné, dostal jsem {n_parents}")

    rng = np.random.default_rng() if rng is None else rng
    return population, fitness, n_parents, rng


@dataclass
class GAResult:
    """Výsledek jednoho běhu GA."""

    best: np.ndarray  # nejlepší nalezený chromozom
    best_fitness: float  # jeho fitness
    evaluations: int  # kolik vyhodnocení fitness se spotřebovalo
    generations: int  # kolik generací se stihlo
    curve: np.ndarray = field(default_factory=lambda: np.empty(0))
    """Konvergenční křivka: curve[i] = nejlepší fitness po i+1 vyhodnoceních."""
    history: list[tuple[int, float, float]] = field(default_factory=list)
    """Průběh po generacích: (počet vyhodnocení, nejlepší fitness, průměrná fitness)."""


def run_ga(
    fitness_fn: Callable[[np.ndarray], np.ndarray],
    D: int,
    pop_size: int = 50,
    elite_ratio: float = 0.1,
    p_mut: float | None = None,
    p_cross: float = 1.0,
    selection: str = "roulette",
    max_evals: int | None = None,
    rng: np.random.Generator | None = None,
) -> GAResult:
    """Spustí genetický algoritmus s binární reprezentací.

    Rozpočet se měří v počtu vyhodnocení fitness funkce, ne v generacích -
    zadání počítá se 100*D vyhodnoceními. Elitní jedinci se do nové generace
    kopírují i s už spočtenou fitness, takže z rozpočtu neubírají.

    Args:
        fitness_fn: Účelová funkce, např. one_max nebo leading_ones.
            Maximalizuje se; musí zvládnout jeden chromozom i celou populaci.
        D: Délka chromozomu.
        pop_size: Velikost populace.
        elite_ratio: Podíl nejlepších, kteří přejdou beze změny (0.1-0.2).
        p_mut: Pravděpodobnost mutace bitu; None znamená 1/D.
        p_cross: Pravděpodobnost křížení páru; při 1.0 se kříží vždy.
        selection: Klíč ze SELECTION_METHODS - "roulette" nebo "rank".
        max_evals: Rozpočet vyhodnocení fitness; None znamená 100*D.
        rng: Generátor náhodných čísel; předej se seedem kvůli reprodukovatelnosti.

    Returns:
        GAResult s nejlepším jedincem, spotřebou rozpočtu a průběhem.

    Raises:
        ValueError: Pokud jsou parametry mimo povolený rozsah nebo se
            počáteční populace nevejde do rozpočtu.
    """
    if not 0.0 <= elite_ratio < 1.0:
        raise ValueError(f"elite_ratio musí být z <0, 1), dostal jsem {elite_ratio}")
    if not 0.0 <= p_cross <= 1.0:
        raise ValueError(f"p_cross musí být z <0, 1>, dostal jsem {p_cross}")

    p_mut = 1.0 / D if p_mut is None else p_mut
    max_evals = 100 * D if max_evals is None else max_evals
    rng = np.random.default_rng() if rng is None else rng

    if pop_size > max_evals:
        raise ValueError(
            f"počáteční populace ({pop_size}) se nevejde do rozpočtu ({max_evals})"
        )

    n_elite = int(round(elite_ratio * pop_size))
    n_elite = min(n_elite, pop_size - 2)  # musí zbýt místo aspoň na jeden pár

    population = random_population(pop_size, D, rng)
    fitness = np.asarray(fitness_fn(population), dtype=np.float64)
    evaluations = pop_size

    best_idx = int(np.argmax(fitness))
    best = population[best_idx].copy()
    best_fitness = float(fitness[best_idx])

    # curve[i] = nejlepší fitness nalezená po i+1 vyhodnoceních. Počáteční
    # populace se ohodnotí najednou, takže se bere běžící maximum přes ni.
    curve = np.empty(max_evals, dtype=np.float64)
    curve[:pop_size] = np.maximum.accumulate(fitness)

    generations = 0
    history = [(evaluations, best_fitness, float(fitness.mean()))]

    while evaluations < max_evals:
        # Elita přechází beze změny i s už spočtenou fitness.
        elite_idx = np.argsort(fitness, kind="stable")[::-1][:n_elite]
        new_population = [population[i].copy() for i in elite_idx]
        new_fitness = [float(fitness[i]) for i in elite_idx]

        while len(new_population) < pop_size and evaluations < max_evals:
            parents = select_parents(population, fitness, 2, selection, rng)

            if rng.random() < p_cross:
                child1, child2 = one_point_crossover(parents[0], parents[1], rng)
            else:
                child1, child2 = parents[0].copy(), parents[1].copy()

            for child in (mutate(child1, p_mut, rng), mutate(child2, p_mut, rng)):
                if len(new_population) >= pop_size or evaluations >= max_evals:
                    break
                child_fitness = float(fitness_fn(child))

                if child_fitness > best_fitness:
                    best_fitness = child_fitness
                    best = child.copy()

                curve[evaluations] = best_fitness
                evaluations += 1

                new_population.append(child)
                new_fitness.append(child_fitness)

        population = np.array(new_population, dtype=np.uint8)
        fitness = np.array(new_fitness, dtype=np.float64)
        generations += 1
        history.append((evaluations, best_fitness, float(fitness.mean())))

    return GAResult(
        best=best,
        best_fitness=best_fitness,
        evaluations=evaluations,
        generations=generations,
        curve=curve[:evaluations],
        history=history,
    )


def chromosome_to_str(chromosome: np.ndarray) -> str:
    """Převede chromozom na řetězec bitů kvůli čitelnému výpisu."""
    return "".join(str(bit) for bit in chromosome)


def main() -> None:
    pop_size, D = 10, 16

    rng = np.random.default_rng(seed=42)
    population = random_population(pop_size, D, rng)

    om = one_max(population)
    lo = leading_ones(population)

    print(f"Populace: {population.shape[0]} jedinců, délka chromozomu {population.shape[1]}")
    print(f"{'':4s}{'chromozom':<{D}}  {'OneMax':>6s}  {'LeadingOnes':>11s}")
    for i, individual in enumerate(population):
        print(f"  {i:2d}: {chromosome_to_str(individual)}  {om[i]:6d}  {lo[i]:11d}")

    print(f"\nOneMax      nejlepší {om.max():3d} / {D},  průměr {om.mean():5.2f}")
    print(f"LeadingOnes nejlepší {lo.max():3d} / {D},  průměr {lo.mean():5.2f}")

    parent1, parent2 = population[0], population[1]
    child1, child2 = one_point_crossover(parent1, parent2, rng)

    print("\nJednobodové křížení prvních dvou jedinců:")
    print(f"  rodič   1: {chromosome_to_str(parent1)}")
    print(f"  rodič   2: {chromosome_to_str(parent2)}")
    print(f"  potomek 1: {chromosome_to_str(child1)}")
    print(f"  potomek 2: {chromosome_to_str(child2)}")

    p_mut = 0.05  # vysoké schválně, aby byl na 16 bitech vidět efekt
    mutated = mutate(child1, p_mut, rng)
    changed = "".join("^" if a != b else " " for a, b in zip(child1, mutated))

    print(f"\nMutace potomka 1 (p_mut = {p_mut}):")
    print(f"  před:  {chromosome_to_str(child1)}")
    print(f"  po:    {chromosome_to_str(mutated)}")
    print(f"         {changed}")

    draws = 1000
    counts = {}
    for method in SELECTION_METHODS:
        parents = select_parents(population, om, n_parents=draws, method=method, rng=rng)
        # Kolikrát se který jedinec objevil mezi rodiči (chromozomy jsou unikátní).
        counts[method] = (parents[:, None, :] == population[None, :, :]).all(axis=2).sum(axis=0)

    print(f"\nSelekce rodičů podle OneMaxu ({draws} losů, seřazeno od nejhoršího):")
    print(f"{'':6s}{'fitness':>7s}  {'roulette':>9s}  {'rank':>6s}")
    for i in np.argsort(om, kind="stable"):
        print(f"  {i:2d}: {om[i]:9d}  {counts['roulette'][i]:9d}  {counts['rank'][i]:6d}")

    print("\n" + "=" * 60)
    print("Celý běh GA na OneMaxu v 10D, rozpočet 100*D = 1000 vyhodnocení")
    print("=" * 60)

    result = run_ga(
        one_max,
        D=10,
        pop_size=20,
        elite_ratio=0.1,
        selection="roulette",
        rng=np.random.default_rng(seed=42),
    )

    print(f"nejlepší jedinec: {chromosome_to_str(result.best)}")
    print(f"fitness:          {result.best_fitness:.0f} / 10")
    print(f"vyhodnocení:      {result.evaluations},  generací: {result.generations}")

    print(f"\n{'evals':>6s}  {'best':>5s}  {'mean':>5s}")
    for evals, best, mean in result.history[::5]:
        print(f"{evals:6d}  {best:5.0f}  {mean:5.2f}")


if __name__ == "__main__":
    main()
