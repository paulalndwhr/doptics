from numpy.typing import NDArray


class Ray:
    def __init__(self, x: NDArray[float], y: NDArray[float]):
        self.x = x
        self.y = y
