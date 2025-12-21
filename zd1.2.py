import numpy as np
import matplotlib.pyplot as plt
from typing import List
import numpy.typing as npt
from numpy.fft import fft, fftshift


class Probe:
    def __init__(self, position: int, maxTime: int):
        self.position = position
        self.E = np.zeros(maxTime)
        self.H = np.zeros(maxTime)
        self._time = 0

    def addData(self, E: npt.NDArray, H: npt.NDArray):
        self.E[self._time] = E[self.position]
        self.H[self._time] = H[self.position]
        self._time += 1


class AnimateFieldDisplay:
    def __init__(self, x: npt.NDArray, minY: float, maxY: float, yLabel: str):
        self.x = x
        self.minY = minY
        self.maxY = maxY
        self.yLabel = yLabel
        self._line = None

    def activate(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(self.x[0], self.x[-1])
        self.ax.set_ylim(self.minY, self.maxY)
        self.ax.set_xlabel('x, м')
        self.ax.set_ylabel(self.yLabel)
        self.ax.grid()
        self._line, = self.ax.plot(self.x, np.zeros_like(self.x))

    def drawProbes(self, probesPos):
        self.ax.plot(probesPos, [0]*len(probesPos), 'xr')

    def drawSources(self, sourcesPos):
        self.ax.plot(sourcesPos, [0]*len(sourcesPos), 'ok')

    def updateData(self, data, time_ns):
        self._line.set_ydata(data)
        self.ax.set_title(f't = {time_ns:.2f} нс')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def stop(self):
        plt.ioff()


def showProbeSignals(probes: List[Probe], dt: float):
    fig, ax = plt.subplots()

    time = np.arange(len(probes[0].E)) * dt  
    time_ns = time * 1e9                    

    ax.set_xlim(time_ns[0], time_ns[-1])
    ax.set_xlabel('t, нс')
    ax.set_ylabel('Ez, В/м')
    ax.grid()

    for probe in probes:
        ax.plot(time_ns, probe.E)

    legend = [f'Probe x = {probe.position}' for probe in probes]
    ax.legend(legend)
    plt.show()

    
if __name__ == '__main__':

    c = 3e8
    Z0 = 120 * np.pi
    Sc = 1.0
    
    L = 2.0       
    dx = 0.01       
    Nx = int(L / dx) 

    x = np.arange(Nx) * dx

    maxTime = 300
    dt = Sc * dx / c

    sourcePos_m = 0.5
    sourcePos = int(sourcePos_m / dx)

    Np = 30
    Md = 2.5

    probesPos_m = [0.7]
    probesPos = [int(p / dx) for p in probesPos_m]
    probes = [Probe(pos, maxTime) for pos in probesPos]

    eps = np.ones(Nx) * 5
    mu = np.ones(Nx - 1)

    Ez = np.zeros(Nx)
    Hy = np.zeros(Nx - 1)

    Sc1Right = Sc / np.sqrt(mu[-1] * eps[-1])
    k1 = -1 / (1 / Sc1Right + 2 + Sc1Right)
    k2 = 1 / Sc1Right - 2 + Sc1Right
    k3 = 2 * (Sc1Right - 1 / Sc1Right)
    k4 = 4 * (1 / Sc1Right + Sc1Right)

    oldEz1 = np.zeros(3)
    oldEz2 = np.zeros(3)
    
    display = AnimateFieldDisplay(x, -1.1, 1.1, 'Ez, В/м')
    display.activate()
    display.drawSources([sourcePos_m])
    display.drawProbes(probesPos_m)
    
    for q in range(maxTime):

        Hy += (Ez[1:] - Ez[:-1]) * Sc / (Z0 * mu)

        Hy[sourcePos - 1] -= (
            (1 - 2 * np.pi**2 * (Sc * q / Np - Md)**2) *
            np.exp(-np.pi**2 * (Sc * q / Np - Md)**2) / Z0
        )

        Ez[1:-1] += (Hy[1:] - Hy[:-1]) * Sc * Z0 / (eps[1:-1] / 5)

        Ez[sourcePos] += (
            (1 - 2 * np.pi**2 * ((Sc * (q + 0.5) + 0.5) / Np - Md)**2) *
            np.exp(-np.pi**2 * ((Sc * (q + 0.5) + 0.5) / Np - Md)**2)
        )

        Ez[0] = 0

        Ez[-1] = (
            k1 * (k2 * (Ez[-3] + oldEz2[-1]) +
                  k3 * (oldEz1[-1] + oldEz1[-3] - Ez[-2] - oldEz2[-2]) -
                  k4 * oldEz1[-2]) - oldEz2[-3]
        )

        oldEz2[:] = oldEz1[:]
        oldEz1[:] = Ez[-3:]

        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 2 == 0:
            display.updateData(Ez, q * dt * 1e9)

    display.stop()


    showProbeSignals(probes, dt)

signal = probes[0].E.copy()
size = len(signal)

d_x = 1e-3
d_t = Sc * d_x / c
df = 1.0 / (size * d_t)

spectrum = np.abs(fft(signal))
spectrum = fftshift(spectrum)
freq = np.arange(-size / 2 * df, size / 2 * df, df)

plt.figure()
plt.plot(freq, spectrum / spectrum.max())
plt.xlabel('Частота, Гц')
plt.ylabel('|S| / |Smax|')
plt.title('Амплитудный спектр')
plt.xlim(0.1e9, 9e9)
plt.grid()
plt.show()
