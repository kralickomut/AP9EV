"""Experimenty: 3 funkce x 3 varianty PSO, konvergenční grafy a statistiky.

Všechno se spustí jedním příkazem:

    python PSO/experiment.py

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

# pso musí být první - přidá do cesty složku s functions.py.
from pso import D, MAX_FES, POP_SIZE, VMAX_RATIO, PSOResult, run_pso  # isort: skip
from functions import FUNCTIONS, TestFunction  # isort: skip

N_RUNS = 10
OUTPUT_DIR = Path(__file__).parent

# Nastavení ze zadání.
LINEAR_W = {"w": (0.8, 0.3), "c1": 2.0, "c2": 2.0, "topology": "star"}
CONST_STAR = {"w": 0.7, "c1": 1.49618, "c2": 1.49618, "topology": "star"}
CONST_RING = {"w": 0.7, "c1": 1.49618, "c2": 1.49618, "topology": "ring", "neighbourhood": 3}

# Varianta = funkce (testovací funkce, rng) -> PSOResult.
Variant = Callable[[TestFunction, np.random.Generator], PSOResult]

VARIANTS: dict[str, Variant] = {
    "w 0.8→0.3, hvězda": lambda f, rng: run_pso(f, rng=rng, **LINEAR_W),
    "w = 0.7, hvězda": lambda f, rng: run_pso(f, rng=rng, **CONST_STAR),
    "w = 0.7, kruh (N = 3)": lambda f, rng: run_pso(f, rng=rng, **CONST_RING),
}


@dataclass
class ExperimentResult:
    """Výsledek N nezávislých běhů jedné varianty na jedné funkci."""

    function: str
    variant: str
    runs: list[PSOResult]

    @property
    def curves(self) -> np.ndarray:
        """Tvar (n_runs, max_fes), nejlepší f po každém ohodnocení."""
        return np.array([r.curve for r in self.runs])

    @property
    def finals(self) -> np.ndarray:
        return np.array([r.best_f for r in self.runs])

    @property
    def outside_curve(self) -> np.ndarray:
        """Průměrný podíl částic mimo oblast po iteracích, oříznutý na nejkratší běh."""
        n = min(len(r.outside) for r in self.runs)
        return np.mean([r.outside[:n] for r in self.runs], axis=0)

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
        "# Srovnání variant PSO na reálných funkcích",
        "",
        f"{n_runs} nezávislých běhů na variantu, D = {D}, NP = {POP_SIZE}, "
        f"MAXFES = {MAX_FES}.",
        "",
        "## Nastavení",
        "",
        "- **w 0.8→0.3, hvězda:** setrvačnost lineárně klesá z 0.8 na 0.3, c1 = c2 = 2, globální topologie",
        "- **w = 0.7, hvězda:** c1 = c2 = 1.49618, globální topologie",
        "- **w = 0.7, kruh (N = 3):** c1 = c2 = 1.49618, kruhová topologie, 3 sousedé na každou stranu",
        f"- Všechny varianty: v(0) = 0, vmax = {VMAX_RATIO} · rozsah, částice mimo oblast se neohodnocují.",
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


def plot_outside(results: list[ExperimentResult], path: Path) -> None:
    """Podíl částic mimo prohledávanou oblast po iteracích, jeden panel na funkci."""
    names = list(dict.fromkeys(r.function for r in results))
    with plt.rc_context({"font.size": 12, "axes.titlesize": 13, "legend.fontsize": 11}):
        fig, axes = plt.subplots(1, len(names), figsize=(12, 3.8), sharey=True)
        for ax, name in zip(axes, names):
            for r in results:
                if r.function == name:
                    curve = r.outside_curve
                    ax.plot(np.arange(1, len(curve) + 1), 100 * curve, label=r.variant, linewidth=1.4)
            ax.set_title(name)
            ax.set_xlabel("iterace")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("částice mimo oblast [%]")
        axes[-1].legend(loc="upper right")

        fig.suptitle(f"Podíl částic mimo prohledávanou oblast, průměr z {len(results[0].runs)} běhů",
                     fontsize=14)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def experiment(n_runs: int = N_RUNS, seed: int = 0, output_dir: Path = OUTPUT_DIR) -> list[ExperimentResult]:
    """Srovnání: 3 funkce x 3 varianty PSO, 10D, MAXFES = 10 000.

    Uloží convergence.png, outside.png a stats.md do output_dir.
    """
    results = run_all(n_runs, seed)

    (output_dir / "stats.md").write_text(format_markdown(results, n_runs), encoding="utf-8")
    print(f"\nStatistiky uloženy do {output_dir / 'stats.md'}")

    plot_curves(results, output_dir / "convergence.png",
                f"PSO: průměrná konvergence z {n_runs} běhů, D = {D}, NP = {POP_SIZE}")
    print(f"Graf uložen do {output_dir / 'convergence.png'}")

    plot_outside(results, output_dir / "outside.png")
    print(f"Graf uložen do {output_dir / 'outside.png'}")
    return results


def main() -> None:
    experiment()


if __name__ == "__main__":
    main()
