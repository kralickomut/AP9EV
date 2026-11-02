"""Experimenty: 3 funkce x 3 varianty DE, konvergenční grafy a statistiky.

Všechno se spustí jedním příkazem:

    python DifEvolution/experiment.py

nebo z Pythonu funkcí experiment().
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # kreslíme do souboru, nepotřebujeme okno

import matplotlib.pyplot as plt
import numpy as np

from de import D, MAX_EVALS, POP_SIZE, DEResult, run_de, run_jde
from functions import FUNCTIONS, TestFunction

N_RUNS = 10
OUTPUT_DIR = Path(__file__).parent

# Doporučené nastavení ze zadání.
CLASSIC = {"F": 0.5, "CR": 0.8}
JDE = {"F": 0.5, "CR": 0.9, "tau1": 0.1, "tau2": 0.1, "F_bounds": (0.1, 0.9)}

# Varianta = funkce (testovací funkce, rng) -> DEResult.
Variant = Callable[[TestFunction, np.random.Generator], DEResult]

VARIANTS: dict[str, Variant] = {
    "DE/rand/1/bin": lambda f, rng: run_de(f, "rand1", rng=rng, **CLASSIC),
    "DE/best/1/bin": lambda f, rng: run_de(f, "best1", rng=rng, **CLASSIC),
    "jDE": lambda f, rng: run_jde(f, rng=rng, **JDE),
}


@dataclass
class ExperimentResult:
    """Výsledek N nezávislých běhů jedné varianty na jedné funkci."""

    function: str
    variant: str
    runs: list[DEResult]

    @property
    def curves(self) -> np.ndarray:
        """Tvar (n_runs, max_evals), nejlepší f po každém vyhodnocení."""
        return np.array([r.curve for r in self.runs])

    @property
    def finals(self) -> np.ndarray:
        return np.array([r.best_f for r in self.runs])

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
    return ExperimentResult(f.name, label, [variant(f, np.random.default_rng(s)) for s in seeds])


def run_all(n_runs: int = N_RUNS, seed: int = 0) -> list[ExperimentResult]:
    results = []
    for f in FUNCTIONS.values():
        for label, variant in VARIANTS.items():
            print(f"  běží {f.name} | {label} ...", flush=True)
            results.append(run_variant(f, label, variant, n_runs, seed))
    return results


def format_markdown(results: list[ExperimentResult], n_runs: int) -> str:
    """Statistiky jako Markdown, jedna tabulka na funkci."""
    lines = [
        "# Srovnání variant DE na reálných funkcích",
        "",
        f"{n_runs} nezávislých běhů na variantu, D = {D}, NP = {POP_SIZE}, "
        f"rozpočet {MAX_EVALS} ohodnocení.",
        "",
        "## Nastavení",
        "",
        f"- **DE/rand/1/bin, DE/best/1/bin:** F = {CLASSIC['F']}, CR = {CLASSIC['CR']}",
        f"- **jDE:** rand/1/bin, počáteční F = {JDE['F']}, CR = {JDE['CR']}, "
        f"τ1 = τ2 = {JDE['tau1']}, F ∈ [{JDE['F_bounds'][0]}, {JDE['F_bounds'][1]}], CR ∈ [0, 1]",
        "- Selekce f(u) ≤ f(x), souřadnice mimo meze se ořežou na mez.",
    ]

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


def plot_curves(results: list[ExperimentResult], path: Path, title: str, floor: float = 1e-30) -> None:
    """Průměrné konvergenční křivky v mřížce 2x2: tři funkce a společná legenda.

    Osa Y je logaritmická, hodnoty pod floor se kreslí na floor.
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
                curve = r.curves.mean(axis=0)
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


def plot_jde_parameters(results: list[ExperimentResult], path: Path) -> None:
    """Průměrné F a CR populace jDE po generacích, zprůměrované přes běhy."""
    jde = [r for r in results if r.variant == "jDE"]
    with plt.rc_context({"font.size": 12, "axes.titlesize": 13, "legend.fontsize": 11}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
        for r in jde:
            F = np.mean([run.F_history for run in r.runs], axis=0)
            CR = np.mean([run.CR_history for run in r.runs], axis=0)
            gens = np.arange(1, len(F) + 1)
            axes[0].plot(gens, F, label=r.function, linewidth=1.6)
            axes[1].plot(gens, CR, label=r.function, linewidth=1.6)

        for ax, name in zip(axes, ("F", "CR")):
            ax.set_title(f"průměrné {name} v populaci")
            ax.set_xlabel("generace")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
        axes[1].legend(loc="best")

        fig.suptitle(f"jDE: vývoj řídicích parametrů, průměr z {len(jde[0].runs)} běhů", fontsize=14)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def experiment(n_runs: int = N_RUNS, seed: int = 0, output_dir: Path = OUTPUT_DIR) -> list[ExperimentResult]:
    """Srovnání: 3 funkce x 3 varianty DE, 10D, 10 000 ohodnocení.

    Uloží convergence.png, jde_parameters.png a stats.md do output_dir.
    """
    results = run_all(n_runs, seed)

    (output_dir / "stats.md").write_text(format_markdown(results, n_runs), encoding="utf-8")
    print(f"\nStatistiky uloženy do {output_dir / 'stats.md'}")

    plot_curves(results, output_dir / "convergence.png",
                f"DE: průměrná konvergence z {n_runs} běhů, D = {D}")
    print(f"Graf uložen do {output_dir / 'convergence.png'}")

    plot_jde_parameters(results, output_dir / "jde_parameters.png")
    print(f"Graf uložen do {output_dir / 'jde_parameters.png'}")
    return results


def main() -> None:
    experiment()


if __name__ == "__main__":
    main()
