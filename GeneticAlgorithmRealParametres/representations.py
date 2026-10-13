"""Tři způsoby, jak reálné číslo zakódovat do 32 bitů.

Všechny reprezentace mají stejné rozhraní:

    bits = rep.encode(x)   # (..., D) reálných čísel -> (..., 32*D) bitů 0/1
    x = rep.decode(bits)   # (..., 32*D) bitů -> (..., D) reálných čísel

Bity jsou uint8 0/1 ve stejném formátu jako populace v GeneticAlgorithm/ga.py,
takže se dají rovnou pustit do binárního GA. Na každou dimenzi připadá
souvislý blok 32 bitů, nejvýznamnější bit první.

decode() vrací surovou hodnotu, kterou bity kódují - může ležet mimo rozsah
funkce a u IEEE 754 to může být i NaN nebo nekonečno. Ošetření (clamping
do mezí) dělá až napojení na GA, viz ga_real.decode_in_bounds.
"""

from __future__ import annotations

import math

import numpy as np

BITS = 32  # bitů na jednu dimenzi, stejně pro všechny reprezentace


def _words_to_bits(words: np.ndarray) -> np.ndarray:
    """(..., D) uint32 -> (..., 32*D) bitů, nejvýznamnější bit první."""
    as_bytes = np.ascontiguousarray(words, dtype=">u4").view(np.uint8)
    return np.unpackbits(as_bytes, axis=-1)


def _bits_to_words(bits: np.ndarray) -> np.ndarray:
    """(..., 32*D) bitů -> (..., D) uint32."""
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.shape[-1] % BITS:
        raise ValueError(f"počet bitů musí být násobek {BITS}, je {bits.shape[-1]}")
    as_bytes = np.ascontiguousarray(np.packbits(bits, axis=-1))
    return as_bytes.view(">u4").astype(np.uint32)


class IEEE754:
    """float32: 1 bit znaménko, 8 bitů exponent, 23 bitů mantisa.

    Hodnota = (-1)^s * 1.mantisa * 2^(exponent - 127). Rozsah nijak nezávisí
    na mezích funkce - pokrývá cca ±3.4e38 a hustota čísel je nerovnoměrná:
    mezi 0 a 1 jich leží stejně jako mezi 1 a 2^127.
    """

    name = "IEEE 754"

    def __init__(self, lower: float, upper: float) -> None:
        self.lower, self.upper = lower, upper  # jen kvůli jednotnému rozhraní

    def encode(self, x: np.ndarray) -> np.ndarray:
        words = np.asarray(x, dtype=np.float32).view(np.uint32)
        return _words_to_bits(words)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        # Exponent 11111111 dává inf/NaN - to je platný výsledek, ne chyba.
        with np.errstate(invalid="ignore"):
            return _bits_to_words(bits).view(np.float32).astype(np.float64)

    def describe(self) -> str:
        return "1 znaménko + 8 exponent + 23 mantisa"


class FixedPoint:
    """Pevná řádová čárka: dvojkový doplněk se int_bits bity celé části.

    Bity: 1 znaménko (dvojkový doplněk) + int_bits celá část + zbytek desetinná
    část. Celá část se volí co nejmenší, aby pokryla největší |mez| funkce;
    všechno ostatní jde do přesnosti. Krok mřížky je 2^-frac_bits a je všude
    stejný.
    """

    name = "Fixed-point"

    def __init__(self, lower: float, upper: float) -> None:
        self.lower, self.upper = lower, upper
        max_abs = max(abs(lower), abs(upper))
        # floor(log2)+1, aby se vešla i mez, která je přesně mocnina dvojky.
        self.int_bits = math.floor(math.log2(max_abs)) + 1
        self.frac_bits = BITS - 1 - self.int_bits
        self.scale = 2.0**self.frac_bits

    def encode(self, x: np.ndarray) -> np.ndarray:
        scaled = np.round(np.asarray(x, dtype=np.float64) * self.scale)
        limits = np.iinfo(np.int32)
        ints = np.clip(scaled, limits.min, limits.max).astype(np.int32)
        return _words_to_bits(ints.view(np.uint32))

    def decode(self, bits: np.ndarray) -> np.ndarray:
        return _bits_to_words(bits).view(np.int32) / self.scale

    def describe(self) -> str:
        return (
            f"1 znaménko + {self.int_bits} celá část + {self.frac_bits} desetinná, "
            f"krok {1 / self.scale:.1e}"
        )


class BCD:
    """Binary Coded Decimal: 8 nibblů po 4 bitech = znaménko + 7 číslic.

    První nibble nese znaménko (rozhoduje jeho nejnižší bit: liché = záporné),
    zbylých 7 nibblů jsou desítkové číslice 0-9. Počet číslic před desetinnou
    čárkou se volí podle největší |meze| funkce, zbytek jsou desetinná místa.

    Nibble 1010-1111 (10-15) žádnou číslici nekóduje. Takový nibble se při
    dekódování ořízne na 9 - jinak by mutace a náhodná inicializace vyráběly
    neplatné jedince.
    """

    name = "BCD"
    DIGITS = BITS // 4 - 1  # 7 číslic, první nibble je znaménko

    def __init__(self, lower: float, upper: float) -> None:
        self.lower, self.upper = lower, upper
        max_abs = max(abs(lower), abs(upper))
        self.int_digits = len(str(int(max_abs)))
        self.frac_digits = self.DIGITS - self.int_digits
        self.scale = 10.0**self.frac_digits
        # Váhy jednotlivých číslic, od nejvyššího řádu.
        self._weights = 10.0 ** np.arange(self.DIGITS - 1, -1, -1)

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        n = np.clip(np.round(np.abs(x) * self.scale), 0, 10**self.DIGITS - 1)
        n = n.astype(np.int64)

        nibbles = np.empty(x.shape + (BITS // 4,), dtype=np.uint8)
        nibbles[..., 0] = x < 0
        for k in range(self.DIGITS, 0, -1):  # od nejnižšího řádu
            n, nibbles[..., k] = np.divmod(n, 10)

        # Nibble -> 4 bity: horní polovina bajtu z unpackbits, MSB první.
        bits = np.unpackbits(nibbles[..., None], axis=-1)[..., 4:]
        return bits.reshape(x.shape[:-1] + (-1,))

    def decode(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8)
        nibbles = bits.reshape(bits.shape[:-1] + (-1, BITS // 4, 4)) @ np.array(
            [8, 4, 2, 1], dtype=np.uint8
        )
        sign = np.where(nibbles[..., 0] & 1, -1.0, 1.0)
        digits = np.minimum(nibbles[..., 1:], 9)  # 10-15 nejsou číslice
        return sign * (digits @ self._weights) / self.scale

    def describe(self) -> str:
        return (
            f"znaménko + 7 číslic ({self.int_digits} před čárkou, "
            f"{self.frac_digits} za ní), krok {1 / self.scale:.0e}"
        )


REPRESENTATIONS = {"IEEE 754": IEEE754, "Fixed-point": FixedPoint, "BCD": BCD}


def bits_to_str(bits: np.ndarray, group: int = 4) -> str:
    """Bity jako řetězec, po skupinách oddělených mezerou."""
    s = "".join(str(b) for b in bits)
    return " ".join(s[i : i + group] for i in range(0, len(s), group))


def flip_effect(rep, x: float, positions: list[int]) -> list[tuple[int, float]]:
    """Co udělá s hodnotou x překlopení jednoho bitu na zadaných pozicích."""
    bits = rep.encode(np.array([x]))
    out = []
    for pos in positions:
        mutated = bits.copy()
        mutated[pos] ^= 1
        out.append((pos, float(rep.decode(mutated)[0])))
    return out


def main() -> None:
    lower, upper = -500.0, 500.0  # rozsah Schwefela, ať jsou vidět všechny řády
    samples = np.array([0.0, 1.0, -2.5, 3.14159, 420.9687, -499.99])

    for cls in REPRESENTATIONS.values():
        rep = cls(lower, upper)
        print(f"\n{rep.name} ({rep.describe()}), meze [{lower:g}, {upper:g}]")
        bits = rep.encode(samples)
        back = rep.decode(bits)
        for x, b, y in zip(samples, bits.reshape(len(samples), BITS), back):
            print(f"  {x:>10.5f} -> {bits_to_str(b)} -> {y:<14.7g} chyba {abs(x - y):.1e}")

    # Co udělá jedna mutace - stejné číslo, různé pozice bitu.
    x = 3.14159
    positions = {
        "IEEE 754": {
            0: "znaménko", 1: "exponent, nejvyšší", 4: "exponent", 8: "exponent, nejnižší",
            9: "mantisa, nejvyšší", 20: "mantisa", 31: "mantisa, nejnižší",
        },
        "Fixed-point": {
            0: "znaménko (dvojkový doplněk)", 1: "celá část, 2^8", 5: "celá část, 2^4",
            9: "celá část, 2^0", 10: "desetinná, 2^-1", 20: "desetinná, 2^-11",
            31: "desetinná, 2^-22",
        },
        "BCD": {
            3: "znaménko", 4: "stovky, bit 8", 7: "stovky, bit 1", 8: "desítky, bit 8",
            12: "jednotky, bit 8", 20: "setiny, bit 8", 31: "desetitisíciny, bit 1",
        },
    }
    print(f"\nPřeklopení jednoho bitu u x = {x}, meze [{lower:g}, {upper:g}]")
    for name, cls in REPRESENTATIONS.items():
        rep = cls(lower, upper)
        print(f"  {name}")
        for pos, y in flip_effect(rep, x, list(positions[name])):
            print(f"    bit {pos:>2} ({positions[name][pos]:<27}) -> {y:.6g}")


if __name__ == "__main__":
    main()
