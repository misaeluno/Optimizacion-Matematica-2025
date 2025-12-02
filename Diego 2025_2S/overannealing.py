from typing import Any, Callable, Iterator

import matplotlib.pyplot as Ⱂ
from numpy import exp as Ⰵ
from numpy import roll as Ⱁ
from numpy import sqrt as Ⱍ
from numpy import sum as Ⱄ
from numpy.random import choice as Ⱌ
from numpy.random import rand as Ⱃ
from numpy.random import randint as Ⰻ
from numpy.random import seed as Ⰴ
from numpy.random import uniform as Ⱆ
from numpy.typing import NDArray


def Ⱎ(e: NDArray[Any], temperature: float) -> NDArray[Any]:
    _e = e.copy()

    n = max(1, int(len(_e) * temperature / 100))
    n = Ⰻ(1, n + 1)

    for _ in range(n):
        method = Ⱌ([0, 1])
        if method == 0:
            i, j = Ⱌ(len(_e), size=2, replace=False)
            _e[[i, j]] = _e[[j, i]]
        else:
            i, j = Ⱌ(len(_e), size=2, replace=False)
            start, end = min(i, j), max(i, j)
            _e[start : end + 1] = _e[start : end + 1][::-1]

    return _e


class SimulatedAnnealing(Iterator):
    def __init__(self, f: Callable, temperature: float = 100, α: float = 0.95) -> None:
        self.f = f
        self.best = ([], 0)
        self.actual = ([], 0)
        self.index = 0
        self.temperature = temperature
        self.α = α

        self.historial = []

    def __getitem__(self, index: int) -> Any:
        if index <= 0:
            return self.best[0]
        else:
            return self.best[1]

    def __len__(self):
        return 2

    def __next__(self):
        if self.index < len(self):
            self.index += 1
            return self.best[0]
        raise StopIteration

    def __call__(self, e: NDArray[Any], iter: int = 10000) -> NDArray[Any]:
        self.best = (e.copy(), self.f(e))
        self.actual = (e.copy(), self.f(e))
        self.historial.append(e.copy())

        for _ in range(iter):
            _e = Ⱎ(self.actual[0], self.temperature)
            _f = self.f(_e)
            print(_f)
            # Cambio en la función objetivo
            Δ = _f - self.actual[1]
            if Δ < 0:
                self.actual = (_e.copy(), _f)
                self.historial.append(_e.copy())
            # Actualizamos el mejor punto
            if self.actual[1] < self[1]:
                self.best = (self.actual[0].copy(), self.actual[1])
            else:
                p = Ⰵ(-Δ / self.temperature)
                if Ⱃ() < p:
                    self.actual = (_e.copy(), _f)
            self.temperature *= self.α
        return self[0]


def main():
    # Ejecucion
    # Ⰴ(2025)
    f = lambda e: Ⱄ(Ⱍ(Ⱄ((e - Ⱁ(e, -1, axis=0)) ** 2, axis=1)))
    # n = 25
    n = 100
    x = Ⱆ(low=0, high=100, size=(n, 2))
    print("x =", x)
    print("f(x) =", f(x))
    s = SimulatedAnnealing(f)
    s(x, iter=100000)
    print("x =", s[0])
    print("f(x) =", s[1])
    # Graficar la función
    Ⱂ.figure(figsize=(10, 12))
    Ⱂ.xlabel("$x$")
    Ⱂ.ylabel("$y$")
    Ⱂ.scatter(x[:, 0], x[:, 1], color="red")
    for i in s.historial:
        Ⱂ.plot(i[:, 0], i[:, 1], color="red", linestyle="--", alpha=0.01)
    Ⱂ.plot(s[0][:, 0], s[0][:, 1], color="blue", linestyle="--")
    Ⱂ.show()


if __name__ == "__main__":
    main()
