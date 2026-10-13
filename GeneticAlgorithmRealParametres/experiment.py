"""Experimenty: 3 funkce x 5 variant GA, konvergenční grafy a statistiky.

Všechno se spustí jedním příkazem:

    python GeneticAlgorithmRealParametres/experiment.py

nebo z Pythonu po částech: experiment() pro hlavní srovnání variant,
compare_mutation() pro vliv sigma a pravděpodobnosti mutace.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # kreslíme do souboru, nepotřebujeme okno

import matplotlib.pyplot as plt
import numpy as np

from functions import FUNCTIONS, TestFunction
from ga_real import MAX_EVALS, RealResult, run_binary_ga, run_real_ga
from representations import BCD, IEEE754, FixedPoint

N_RUNS = 10
OUTPUT_DIR = Path(__file__).parent

# Společné nastavení - doporučené z minulého úkolu.
COMMON = {"pop_size": 20, "elite_ratio": 0.2, "selection": "rank"}
P_MUT_BIT = 0.01  # pravděpodobnost překlopení bitu u binárních variant
P_MUT_REAL = 0.1  # pravděpodobnost mutace souřadnice u real-valued (= 1/D)
SIGMA_REL = 0.01  # sigma normální mutace jako podíl rozsahu funkce

# Varianta = funkce (testovací funkce, rng) -> RealResult.
Variant = Callable[[TestFunction, np.random.Generator], RealResult]


def binary_variant(rep_cls, p_mut: float = P_MUT_BIT) -> Variant:
    def run(f: TestFunction, rng: np.random.Generator) -> RealResult:
        return run_binary_ga(f, rep_cls(f.lower, f.upper), p_mut=p_mut, rng=rng, **COMMON)

    return run


def normal_variant(sigma_rel: float = SIGMA_REL, p_mut: float = P_MUT_REAL) -> Variant:
    def run(f: TestFunction, rng: np.random.Generator) -> RealResult:
        return run_real_ga(
            f, "normal", p_mut=p_mut, sigma=sigma_rel * f.range, rng=rng, **COMMON
        )

    return run


def uniform_variant(p_mut: float = P_MUT_REAL) -> Variant:
    def run(f: TestFunction, rng: np.random.Generator) -> RealResult:
        return run_real_ga(f, "uniform", p_mut=p_mut, rng=rng, **COMMON)

    return run


VARIANTS: dict[str, Variant] = {
    "IEEE 754": binary_variant(IEEE754),
    "Fixed-point": binary_variant(FixedPoint),
    "BCD": binary_variant(BCD),
    "Real + N(0, σ²)": normal_variant(),
    "Real + náhodná": uniform_variant(),
}


@dataclass
class ExperimentResult:
    """Výsledek N nezávislých běhů jedné varianty na jedné funkci."""

    function: str
    variant: str
    curves: np.ndarray  # tvar (n_runs, max_evals), nejlepší f po každém vyhodnocení
    finals: np.ndarray  # finální nejlepší f každého běhu

    @property
    def mean_curve(self) -> np.ndarray:
        return self.curves.mean(axis=0)

    @property
    def stats(self) -> dict[str, float]:
        """Statistiky finálních výsledků; nejlepší = nejmenší f."""
        return {
            "nejlepsi": float(self.finals.min()),
            "nejhorsi": float(self.finals.max()),
            "prumer": float(self.finals.mean()),
            "median": float(np.median(self.finals)),
            "smerodatna_odchylka": float(self.finals.std(ddof=1)),
        }


def run_variant(
    f: TestFunction, label: str, variant: Variant, n_runs: int = N_RUNS, seed: int = 0
) -> ExperimentResult:
    """n_runs nezávislých běhů jedné varianty; stejné seedy pro všechny varianty."""
    seeds = np.random.SeedSequence(seed).spawn(n_runs)
    results = [variant(f, np.random.default_rng(s)) for s in seeds]
    return ExperimentResult(
        function=f.name,
        variant=label,
        curves=np.array([r.curve for r in results]),
        finals=np.array([r.best_f for r in results]),
    )


def run_all(
    variants: dict[str, Variant], n_runs: int = N_RUNS, seed: int = 0
) -> list[ExperimentResult]:
    results = []
    for f in FUNCTIONS.values():
        for label, variant in variants.items():
            print(f"  běží {f.name} | {label} ...", flush=True)
            results.append(run_variant(f, label, variant, n_runs, seed))
    return results


def format_markdown(results: list[ExperimentResult], title: str, intro: list[str]) -> str:
    """Statistiky jako Markdown, jedna tabulka na funkci."""
    lines = [f"# {title}", "", *intro]

    for name in dict.fromkeys(r.function for r in results):
        f = FUNCTIONS[name]
        lines += [
            "",
            f"## {name}",
            "",
            f"Rozsah [{f.lower:g}, {f.upper:g}], globální minimum f = 0 v x_i = {f.optimum_x:g}.",
            "",
            "| Varianta | Nejlepší | Nejhorší | Průměr | Medián | Sm. odch. |",
            "|:---------|---------:|---------:|-------:|-------:|----------:|",
        ]
        for r in results:
            if r.function != name:
                continue
            s = r.stats
            lines.append(
                f"| {r.variant} | {s['nejlepsi']:.3e} | {s['nejhorsi']:.3e} | "
                f"{s['prumer']:.3e} | {s['median']:.3e} | {s['smerodatna_odchylka']:.3e} |"
            )

    lines += ["", "Nejlepší = nejmenší hodnota f. Směrodatná odchylka je výběrová (ddof=1).", ""]
    return "\n".join(lines)


def plot_curves(
    results: list[ExperimentResult], path: Path, title: str, floor: float = 1e-30
) -> None:
    """Průměrné konvergenční křivky v mřížce 2x2: tři funkce a společná legenda.

    Osa Y je logaritmická. Hodnoty pod floor se kreslí na floor - jinak by
    jediná varianta, která spadne na 1e-80, zmáčkla ostatní do pruhu nahoře.
    """
    names = list(dict.fromkeys(r.function for r in results))
    with plt.rc_context({"font.size": 12, "axes.titlesize": 13, "legend.fontsize": 12}):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
        axes = axes.ravel()

        for ax, name in zip(axes, names):
            clipped = False
            for r in results:
                if r.function != name:
                    continue
                curve = r.mean_curve
                clipped |= bool((curve < floor).any())
                ax.plot(np.arange(1, len(curve) + 1), np.maximum(curve, floor),
                        label=r.variant, linewidth=1.6)

            ax.set_yscale("log")
            ax.set_title(name + (f" (oříznuto na {floor:.0e})" if clipped else ""))
            ax.set_xlabel("počet ohodnocení")
            ax.set_ylabel("průměrná nejlepší f(x)")
            ax.grid(alpha=0.3, which="both")

        # Čtvrté pole patří legendě - je společná pro všechny tři funkce.
        legend_ax = axes[len(names)]
        legend_ax.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc="center", frameon=False)

        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def experiment(n_runs: int = N_RUNS, seed: int = 0, output_dir: Path = OUTPUT_DIR) -> list[ExperimentResult]:
    """Hlavní srovnání: 3 funkce x 5 variant, 10D, 10 000 ohodnocení.

    Uloží convergence.png a stats.md do output_dir.
    """
    results = run_all(VARIANTS, n_runs, seed)

    intro = [
        f"{n_runs} nezávislých běhů na variantu, D = 10, rozpočet {MAX_EVALS} ohodnocení.",
        "",
        "## Nastavení",
        "",
        f"- **Společné:** populace {COMMON['pop_size']}, elitismus {COMMON['elite_ratio']:.0%}, "
        f"{COMMON['selection']} selekce, jednobodové křížení vždy",
        f"- **Binární varianty (IEEE 754, Fixed-point, BCD):** 32 bitů na dimenzi, "
        f"chromozom 320 bitů, pravděpodobnost překlopení bitu {P_MUT_BIT}",
        f"- **Real-valued:** pravděpodobnost mutace souřadnice {P_MUT_REAL}, "
        f"normální mutace se σ = {SIGMA_REL:.0%} rozsahu funkce",
    ]
    text = format_markdown(results, "Srovnání variant GA na reálných funkcích", intro)
    (output_dir / "stats.md").write_text(text, encoding="utf-8")
    print(f"\nStatistiky uloženy do {output_dir / 'stats.md'}")

    plot_curves(results, output_dir / "convergence.png",
                f"Průměrná konvergence z {n_runs} běhů, D = 10")
    print(f"Graf uložen do {output_dir / 'convergence.png'}")
    return results


SIGMAS = {"malá": 0.001, "střední": 0.01, "velká": 0.1}
P_MUTS = (0.05, 0.1, 0.3)


def compare_mutation(n_runs: int = N_RUNS, seed: int = 0, output_dir: Path = OUTPUT_DIR) -> list[ExperimentResult]:
    """Real-valued GA s normální mutací: vliv sigma a pravděpodobnosti mutace.

    Nejdřív se mění sigma (p_mut = P_MUT_REAL), pak p_mut (sigma = SIGMA_REL).
    Uloží sigma_comparison.png, p_mut_comparison.png a mutation_comparison.md.
    """
    sigma_variants = {
        f"σ = {rel:g}·rozsah ({label})": normal_variant(sigma_rel=rel)
        for label, rel in SIGMAS.items()
    }
    p_mut_variants = {f"p_mut = {p:g}": normal_variant(p_mut=p) for p in P_MUTS}

    sigma_results = run_all(sigma_variants, n_runs, seed)
    p_mut_results = run_all(p_mut_variants, n_runs, seed)

    plot_curves(sigma_results, output_dir / "sigma_comparison.png",
                f"Real + N(0, σ²): vliv σ (p_mut = {P_MUT_REAL}), průměr z {n_runs} běhů")
    plot_curves(p_mut_results, output_dir / "p_mut_comparison.png",
                f"Real + N(0, σ²): vliv p_mut (σ = {SIGMA_REL:g}·rozsah), průměr z {n_runs} běhů")

    sigma_text = format_markdown(
        sigma_results, "Vliv σ normální mutace",
        [f"Real-valued GA, p_mut = {P_MUT_REAL}, {n_runs} běhů, D = 10, {MAX_EVALS} ohodnocení. "
         "σ je zadané jako podíl rozsahu funkce, aby šly funkce porovnat."],
    )
    p_mut_text = format_markdown(
        p_mut_results, "Vliv pravděpodobnosti mutace",
        [f"Real-valued GA, σ = {SIGMA_REL:g}·rozsah, {n_runs} běhů, D = 10, {MAX_EVALS} ohodnocení."],
    )
    # Druhý dokument o úroveň níž, ať je to jeden soubor s jedním H1.
    p_mut_text = p_mut_text.replace("\n## ", "\n### ").replace("# Vliv", "## Vliv", 1)
    sigma_text = sigma_text.replace("\n## ", "\n### ").replace("# Vliv", "## Vliv", 1)
    text = "# Experimenty s normální mutací\n\n" + sigma_text + "\n" + p_mut_text
    (output_dir / "mutation_comparison.md").write_text(text, encoding="utf-8")
    print(f"\nTabulky uloženy do {output_dir / 'mutation_comparison.md'}")

    return sigma_results + p_mut_results


def main() -> None:
    experiment()
    print("\n" + "=" * 78 + "\n")
    compare_mutation()


if __name__ == "__main__":
    main()
