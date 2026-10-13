"""Testovací funkce s reálnými parametry - všechny se minimalizují.

Každá funkce bere jednoho jedince (1D pole délky D) i celou populaci
(2D pole tvaru (pop_size, D)) a vrací jedno číslo, resp. vektor hodnot.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def sphere(x: np.ndarray) -> np.ndarray:
    """Sphere: f(x) = sum x_i^2. Hladká, unimodální, separabilní - sanity check."""
    x = np.asarray(x, dtype=np.float64)
    return np.sum(x**2, axis=-1)


def rosenbrock(x: np.ndarray) -> np.ndarray:
    """Rosenbrock: f(x) = sum 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2.

    Minimum leží na dně úzkého zahnutého údolí. Do údolí se trefí snadno,
    po jeho dně k optimu (1, ..., 1) se ale postupuje pomalu. Proměnné jsou
    provázané, funkce není separabilní.
    """
    x = np.asarray(x, dtype=np.float64)
    head, tail = x[..., :-1], x[..., 1:]
    return np.sum(100.0 * (tail - head**2) ** 2 + (1.0 - head) ** 2, axis=-1)


def schwefel(x: np.ndarray) -> np.ndarray:
    """Schwefel: f(x) = 418.9829 D - sum x_i sin(sqrt|x_i|).

    Silně multimodální - v každé dimenzi je asi 8 lokálních minim a to
    globální (x_i = 420.9687) leží daleko od druhého nejlepšího, u kraje
    rozsahu. Separabilní, takže každou souřadnici lze ladit zvlášť.
    """
    x = np.asarray(x, dtype=np.float64)
    D = x.shape[-1]
    return 418.9829 * D - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=-1)


@dataclass(frozen=True)
class TestFunction:
    """Testovací funkce i s rozsahem a polohou globálního minima."""

    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    lower: float  # dolní mez každé souřadnice
    upper: float  # horní mez každé souřadnice
    optimum_x: float  # x_i v globálním minimu (stejné ve všech dimenzích)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.fn(x)

    @property
    def range(self) -> float:
        return self.upper - self.lower


# Rozsahy jsou ty standardní z literatury; globální minimum je u všech f = 0.
FUNCTIONS = {
    "Sphere": TestFunction("Sphere", sphere, -5.12, 5.12, 0.0),
    "Rosenbrock": TestFunction("Rosenbrock", rosenbrock, -5.0, 10.0, 1.0),
    "Schwefel": TestFunction("Schwefel", schwefel, -500.0, 500.0, 420.9687),
}


def main() -> None:
    D = 10
    rng = np.random.default_rng(0)

    print(f"{'funkce':<11}{'rozsah':>20}{'x*':>11}{'f(x*)':>12}{'f(náhodný)':>14}")
    for f in FUNCTIONS.values():
        x_opt = np.full(D, f.optimum_x)
        x_rand = rng.uniform(f.lower, f.upper, D)
        bounds = f"[{f.lower:g}, {f.upper:g}]"
        print(f"{f.name:<11}{bounds:>20}{f.optimum_x:>11g}{f(x_opt):>12.2e}{f(x_rand):>14.1f}")


if __name__ == "__main__":
    main()
