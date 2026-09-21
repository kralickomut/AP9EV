"""Experimenty nad GA: opakované běhy, konvergenční křivky a statistiky.

Spouští se samostatně:

    python GeneticAlgorithm/experiment.py

Pro každou kombinaci úlohy, dimenze a selekce provede N nezávislých běhů,
z jejich konvergenčních křivek spočítá průměrnou křivku a z finálních
výsledků tabulku statistik.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # kreslíme do souboru, nepotřebujeme okno

import matplotlib.pyplot as plt
import numpy as np

from ga import SELECTION_METHODS, leading_ones, one_max, run_ga

N_RUNS = 10
DIMENSIONS = (10, 30, 100)
OBJECTIVES = {"OneMax": one_max, "LeadingOnes": leading_ones}
OUTPUT_DIR = Path(__file__).parent

# Výchozí nastavení kontrolních parametrů - viz ladění popsané v README.
# p_mut drží rozsah 0.5-1 % ze zadání, elitismus rozsah 10-20 %.
DEFAULT_PARAMS = {
    "pop_size": 20,
    "elite_ratio": 0.2,
    "p_mut": 0.01,
    "p_cross": 1.0,
}


@dataclass
class ExperimentResult:
    """Výsledek N nezávislých běhů jedné konfigurace."""

    objective: str  # název účelové funkce
    D: int  # dimenze
    selection: str  # použitá selekce
    curves: np.ndarray  # tvar (n_runs, max_evals), nejlepší fitness po každém vyhodnocení
    finals: np.ndarray  # finální nejlepší fitness každého běhu, délka n_runs
    label: str = ""  # popis konfigurace; prázdný pro běžný běh, vyplněný při srovnání

    @property
    def mean_curve(self) -> np.ndarray:
        """Průměrná konvergenční křivka přes všechny běhy."""
        return self.curves.mean(axis=0)

    @property
    def stats(self) -> dict[str, float]:
        """Statistiky z finálních výsledků jednotlivých běhů."""
        return {
            "nejlepsi": float(self.finals.max()),
            "nejhorsi": float(self.finals.min()),
            "prumer": float(self.finals.mean()),
            "median": float(np.median(self.finals)),
            "smerodatna_odchylka": float(self.finals.std(ddof=1)),
        }


def run_experiment(
    fitness_fn: Callable[[np.ndarray], np.ndarray],
    D: int,
    objective: str,
    selection: str = "roulette",
    n_runs: int = N_RUNS,
    seed: int = 0,
    label: str = "",
    **ga_kwargs,
) -> ExperimentResult:
    """Spustí n_runs nezávislých běhů GA a posbírá z nich křivky i výsledky.

    Args:
        fitness_fn: Účelová funkce (one_max nebo leading_ones).
        D: Délka chromozomu.
        objective: Název úlohy do tabulky a grafu.
        selection: Klíč ze SELECTION_METHODS.
        n_runs: Počet nezávislých běhů.
        seed: Základní seed; jednotlivé běhy z něj dostanou nezávislé proudy.
        label: Popis konfigurace do tabulky a legendy grafu při srovnávání.
        **ga_kwargs: Další parametry předané do run_ga (pop_size, p_mut, ...).

    Returns:
        ExperimentResult s maticí křivek a finálními výsledky.
    """
    # SeedSequence.spawn dá n_runs zaručeně nezávislých proudů náhodných čísel,
    # což je čistší než seed, seed+1, seed+2, ...
    seeds = np.random.SeedSequence(seed).spawn(n_runs)

    curves = []
    finals = []
    for child_seed in seeds:
        result = run_ga(
            fitness_fn,
            D,
            selection=selection,
            rng=np.random.default_rng(child_seed),
            **ga_kwargs,
        )
        curves.append(result.curve)
        finals.append(result.best_fitness)

    return ExperimentResult(
        objective=objective,
        D=D,
        selection=selection,
        curves=np.array(curves),
        finals=np.array(finals),
        label=label,
    )


def format_stats_table(results: list[ExperimentResult]) -> str:
    """Sestaví tabulku statistik z finálních výsledků jednotlivých konfigurací."""
    header = (
        f"{'Úloha':<12}{'D':>5}{'Selekce':>10}{'Nejlepší':>10}{'Nejhorší':>10}"
        f"{'Průměr':>9}{'Medián':>9}{'Sm. odch.':>11}"
    )
    lines = [header, "-" * len(header)]

    for r in results:
        s = r.stats
        lines.append(
            f"{r.objective:<12}{r.D:>5}{r.selection:>10}"
            f"{s['nejlepsi']:>10.0f}{s['nejhorsi']:>10.0f}"
            f"{s['prumer']:>9.2f}{s['median']:>9.1f}{s['smerodatna_odchylka']:>11.3f}"
        )

    return "\n".join(lines)


def plot_convergence(
    results: list[ExperimentResult],
    objective: str,
    path: Path,
    n_runs: int = N_RUNS,
) -> None:
    """Vykreslí průměrné konvergenční křivky jedné úlohy pro všechny dimenze.

    Args:
        results: Výsledky, ze kterých se vyberou ty pro danou úlohu.
        objective: Název úlohy.
        path: Kam uložit obrázek.
        n_runs: Počet běhů (jen do titulku).
    """
    dimensions = sorted({r.D for r in results if r.objective == objective})

    fig, axes = plt.subplots(
        1, len(dimensions), figsize=(4.5 * len(dimensions), 3.8), squeeze=False
    )

    for col, D in enumerate(dimensions):
        ax = axes[0][col]

        for r in results:
            if r.objective != objective or r.D != D:
                continue
            curve = r.mean_curve
            ax.plot(np.arange(1, len(curve) + 1), curve, label=r.selection)

        # Optimum je u obou účelovek rovno D.
        ax.axhline(D, color="grey", linestyle="--", linewidth=0.8, label="optimum")

        ax.set_title(f"D = {D}   (rozpočet {100 * D})")
        ax.set_xlabel("počet ohodnocení účelové funkce")
        ax.set_ylabel("průměrná nejlepší fitness")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"{objective} - průměrná konvergence z {n_runs} běhů", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def experiment(
    objectives: dict[str, Callable] | None = None,
    dimensions: tuple[int, ...] = DIMENSIONS,
    selections: tuple[str, ...] | None = None,
    n_runs: int = N_RUNS,
    seed: int = 0,
    output_dir: Path = OUTPUT_DIR,
    **ga_kwargs,
) -> list[ExperimentResult]:
    """Projede obě úlohy ve všech dimenzích, vykreslí grafy a vypíše statistiky.

    Pro každou kombinaci úlohy, dimenze a selekce spustí n_runs nezávislých
    běhů s rozpočtem 100*D vyhodnocení. Uloží jeden graf na úlohu a tabulku
    statistik, obojí do output_dir.

    Args:
        objectives: Úlohy jako {název: funkce}; None znamená OneMax i LeadingOnes.
        dimensions: Dimenze k proběhnutí.
        selections: Selekce k porovnání; None znamená všechny ze SELECTION_METHODS.
        n_runs: Počet nezávislých běhů na konfiguraci.
        seed: Základní seed, kvůli reprodukovatelnosti.
        output_dir: Adresář pro grafy a stats.txt.
        **ga_kwargs: Přepis kontrolních parametrů (pop_size, elite_ratio,
            p_mut, p_cross); nezadané se berou z DEFAULT_PARAMS.

    Returns:
        Výsledky všech konfigurací v pořadí úloha x dimenze x selekce.
    """
    objectives = OBJECTIVES if objectives is None else objectives
    selections = tuple(SELECTION_METHODS) if selections is None else selections
    params = {**DEFAULT_PARAMS, **ga_kwargs}

    results = []
    for objective, fitness_fn in objectives.items():
        for D in dimensions:
            for selection in selections:
                print(f"  běží {objective} {D}D {selection} ...", flush=True)
                results.append(
                    run_experiment(
                        fitness_fn,
                        D,
                        objective=objective,
                        selection=selection,
                        n_runs=n_runs,
                        seed=seed,
                        **params,
                    )
                )

    settings = (
        f"{n_runs} nezávislých běhů, rozpočet 100*D vyhodnocení, "
        f"pop_size = {params['pop_size']}, elitismus {params['elite_ratio']:.0%}, "
        f"p_mut = {params['p_mut']}, p_cross = {params['p_cross']}"
    )
    table = format_stats_table(results)
    footer = "Sm. odch. je výběrová (ddof=1). Optimum je u obou úloh rovno D."

    print(f"\n{settings}\n")
    print(table)
    print(f"\n{footer}")

    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "stats.txt"
    stats_path.write_text(f"{settings}\n\n{table}\n\n{footer}\n", encoding="utf-8")
    print(f"\nStatistiky uloženy do {stats_path}")

    for objective in objectives:
        plot_path = output_dir / f"convergence_{objective.lower()}.png"
        plot_convergence(results, objective, plot_path, n_runs)
        print(f"Graf uložen do {plot_path}")

    return results



# Srovnávaná nastavení: baseline a tři varianty, každá mění jeden faktor.
COMPARED_SETTINGS = {
    "A malá pop. + rank": {
        "pop_size": 20, "elite_ratio": 0.2, "p_mut": 0.01, "selection": "rank",
    },
    "B větší populace": {
        "pop_size": 50, "elite_ratio": 0.1, "p_mut": 0.01, "selection": "rank",
    },
    "C mutace 0.5 %": {
        "pop_size": 20, "elite_ratio": 0.2, "p_mut": 0.005, "selection": "rank",
    },
    "D ruletová selekce": {
        "pop_size": 20, "elite_ratio": 0.2, "p_mut": 0.01, "selection": "roulette",
    },
}


def format_comparison_table(results: list[ExperimentResult]) -> str:
    """Tabulka statistik pro srovnání nastavení; první sloupec je konfigurace."""
    header = (
        f"{'Nastavení':<20}{'Úloha':<12}{'D':>5}{'Nejlepší':>10}{'Nejhorší':>10}"
        f"{'Průměr':>9}{'Medián':>9}{'Sm. odch.':>11}"
    )
    lines = [header, "-" * len(header)]

    for r in results:
        st = r.stats
        lines.append(
            f"{r.label:<20}{r.objective:<12}{r.D:>5}"
            f"{st['nejlepsi']:>10.0f}{st['nejhorsi']:>10.0f}"
            f"{st['prumer']:>9.2f}{st['median']:>9.1f}{st['smerodatna_odchylka']:>11.3f}"
        )

    return "\n".join(lines)


def plot_comparison(
    results: list[ExperimentResult],
    objective: str,
    path: Path,
    n_runs: int,
) -> None:
    """Vykreslí průměrné konvergenční křivky srovnávaných nastavení."""
    dimensions = sorted({r.D for r in results if r.objective == objective})

    fig, axes = plt.subplots(
        1, len(dimensions), figsize=(4.5 * len(dimensions), 3.8), squeeze=False
    )

    for col, D in enumerate(dimensions):
        ax = axes[0][col]

        for r in results:
            if r.objective != objective or r.D != D:
                continue
            curve = r.mean_curve
            ax.plot(np.arange(1, len(curve) + 1), curve, label=r.label, linewidth=1.3)

        ax.axhline(D, color="grey", linestyle="--", linewidth=0.8, label="optimum")

        ax.set_title(f"D = {D}   (rozpočet {100 * D})")
        ax.set_xlabel("počet ohodnocení účelové funkce")
        ax.set_ylabel("průměrná nejlepší fitness")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        f"{objective} - srovnání nastavení, průměr z {n_runs} běhů", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare_settings(
    settings: dict[str, dict] | None = None,
    objectives: dict[str, Callable] | None = None,
    dimensions: tuple[int, ...] = DIMENSIONS,
    n_runs: int = 30,
    seed: int = 0,
    output_dir: Path = OUTPUT_DIR,
) -> list[ExperimentResult]:
    """Postaví proti sobě několik nastavení kontrolních parametrů.

    Každá konfigurace se pustí na všech úlohách a dimenzích se stejným
    rozpočtem i stejnými seedy, takže jsou výsledky přímo porovnatelné.

    Args:
        settings: {popis: parametry pro run_ga}; None znamená COMPARED_SETTINGS.
        objectives: Úlohy jako {název: funkce}; None znamená obě.
        dimensions: Dimenze k proběhnutí.
        n_runs: Počet nezávislých běhů na konfiguraci; víc než u hlavní
            tabulky, aby byl rozdíl mezi nastaveními čitelný přes šum.
        seed: Základní seed - stejný pro všechna nastavení.
        output_dir: Adresář pro grafy a tabulku.

    Returns:
        Výsledky všech konfigurací.
    """
    settings = COMPARED_SETTINGS if settings is None else settings
    objectives = OBJECTIVES if objectives is None else objectives

    results = []
    for objective, fitness_fn in objectives.items():
        for D in dimensions:
            for label, params in settings.items():
                print(f"  běží {label} | {objective} {D}D ...", flush=True)
                results.append(
                    run_experiment(
                        fitness_fn,
                        D,
                        objective=objective,
                        n_runs=n_runs,
                        seed=seed,
                        label=label,
                        **params,
                    )
                )

    intro = f"Srovnání nastavení, {n_runs} nezávislých běhů, rozpočet 100*D vyhodnocení\n"
    intro += "\n".join(
        f"  {label:<20} " + ", ".join(f"{k} = {v}" for k, v in params.items())
        for label, params in settings.items()
    )
    table = format_comparison_table(results)

    print(f"\n{intro}\n")
    print(table)

    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "settings_comparison.txt"
    path.write_text(f"{intro}\n\n{table}\n", encoding="utf-8")
    print(f"\nTabulka uložena do {path}")

    for objective in objectives:
        plot_path = output_dir / f"comparison_{objective.lower()}.png"
        plot_comparison(results, objective, plot_path, n_runs)
        print(f"Graf uložen do {plot_path}")

    return results


def main() -> None:
    experiment()
    print("\n" + "=" * 78 + "\n")
    compare_settings()


if __name__ == "__main__":
    main()
