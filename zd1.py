import numpy as np
import matplotlib.pyplot as plt
from typing import List
import numpy.typing as npt
from numpy.fft import fft, fftshift

class Probe: # класс для хранения временного сигнала в датчике
    
    def __init__(self, position: int, maxTime: int):
        
        self.position = position
        self.E = np.zeros(maxTime) # Временные сигналы для полей E и H
        self.H = np.zeros(maxTime)
        self._time = 0 # Номер временного шага для сохранения полей

    def addData(self, E: npt.NDArray, H: npt.NDArray):
    
        self.E[self._time] = E[self.position]
        self.H[self._time] = H[self.position]
        self._time += 1

class AnimateFieldDisplay:
  
    def __init__(self,
                 maxXSize: int,
                 minYSize: float, maxYSize: float,
                 yLabel: str):
        
        self._maxXSize = maxXSize
        self._minYSize = minYSize
        self._maxYSize = maxYSize
        self._xdata = None
        self._line = None
        self._xlabel = 'x, отсчет'
        self._ylabel = yLabel
        self._probeStyle = 'xr'
        self._sourceStyle = 'ok'

    def activate(self):

        self._xdata = np.arange(self._maxXSize)
        plt.ion()
        self._fig, self._ax = plt.subplots()
        self._ax.set_xlim(0, self._maxXSize)
        self._ax.set_ylim(self._minYSize, self._maxYSize)
        self._ax.set_xlabel(self._xlabel)
        self._ax.set_ylabel(self._ylabel)
        self._ax.grid()
        self._line = self._ax.plot(self._xdata, np.zeros(self._maxXSize))[0]

    def drawProbes(self, probesPos: List[int]):
       
        self._ax.plot(probesPos, [0] * len(probesPos), self._probeStyle)

    def drawSources(self, sourcesPos: List[int]):
    
        self._ax.plot(sourcesPos, [0] * len(sourcesPos), self._sourceStyle)
        
    def stop(self):
        
        plt.ioff()

    def updateData(self, data: npt.NDArray, timeCount: int):

        self._line.set_ydata(data)
        self._ax.set_title(str(timeCount))
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    
def showProbeSignals(probes: List[Probe], minYSize: float, maxYSize: float):
 
    # Создание окна с графиков
    fig, ax = plt.subplots()

    # Настройка внешнего вида графиков
    ax.set_xlim(0, len(probes[0].E))
    ax.set_ylim(minYSize, maxYSize)
    ax.set_xlabel('q, отсчет')
    ax.set_ylabel('Ez, В/м')
    ax.grid()

    # Вывод сигналов в окно
    for probe in probes:
        ax.plot(probe.E)

    # Создание и отображение легенды на графике
    legend = ['Probe x = {}'.format(probe.position) for probe in probes]
    ax.legend(legend)

    # Показать окно с графиками
    plt.show()
    
if __name__ == '__main__':

    Z0 = 120.0 * np.pi
    Sc = 1
    maxTime = 300
    maxSize = 200
    sourcePos = 50 # положение источника в отсчетах

    Np = 30 # кол-во отчётов на длину волны
    Md = 2.5 # задержка

    probesPos = [70]
    probes = [Probe(pos, maxTime) for pos in probesPos]
    
    eps = np.ones(maxSize) # диэлектрическая проницаемость
    mu = np.ones(maxSize - 1)

    Ez = np.zeros(maxSize)
    Hy = np.zeros(maxSize - 1)

    Sc1Right = Sc / np.sqrt(mu[-1] * eps[-1]) 
    k1Right = -1 / (1 / Sc1Right + 2 + Sc1Right)
    k2Right = 1 / Sc1Right - 2 + Sc1Right
    k3Right = 2 * (Sc1Right - 1 / Sc1Right)
    k4Right = 4 * (1 / Sc1Right + Sc1Right)

    # Ez[-3: -1] в предыдущий момент времени (q)
    oldEzRight1 = np.zeros(3)
    # Ez[-3: -1] в пред-предыдущий момент времени (q - 1)
    oldEzRight2 = np.zeros(3)
    
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    display = AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel)
    
    display.activate()
    display.drawSources([sourcePos])
    display.drawProbes(probesPos)

    for q in range(maxTime):
        
        # Расчет компоненты поля H
        Hy[:] = Hy + (Ez[1:] - Ez[:-1]) * Sc / (Z0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= ((1 - 2 * (np.pi ** 2) * (Sc * q / Np - Md) ** 2) *
                              np.exp(-np.pi ** 2 * (Sc * q / Np - Md) ** 2) / Z0)

        # Расчет компоненты поля E
        Ez[1:-1] = Ez[1: -1] + (Hy[1:] - Hy[:-1] ) * Sc * Z0 / eps[1: -1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += ((1 - 2 * np.pi ** 2 * ((Sc * (q + 0.5) - (-0.5)) / Np - Md) ** 2) *
                          np.exp(-np.pi ** 2 * ((Sc * (q + 0.5) - (-0.5)) / Np - Md) ** 2))
        
        Ez[0] = 0 #PEC
        
        # Граничные условия ABC второй степени (справа)
        Ez[-1] = (k1Right * (k2Right * (Ez[-3] + oldEzRight2[-1]) +
                             k3Right * (oldEzRight1[-1] + oldEzRight1[-3] - Ez[-2] - oldEzRight2[-2]) -
                             k4Right * oldEzRight1[-2]) - oldEzRight2[-3])
        
        oldEzRight2[:] = oldEzRight1[:]
        oldEzRight1[:] = Ez[-3:]
        

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 2 == 0:
            display.updateData(display_field, q)

    display.stop()
    
    # Отображение сигналов, сохраненных в датчиках
    showProbeSignals(probes, -1.1, 1.1)
    
    size = len(probe.E)
    dt = 0.4e-10
    df = 1.0 / ( size * dt)
    
    spectrum = np.abs(fft(probe.E))
    spectrum = fftshift(spectrum)
    freq = np.arange(-size / 2 * df, size / 2 * df, df)
        
    plt.plot(freq, spectrum / np.max(spectrum))
    plt.grid()
    plt.xlabel('частота')
    plt.ylabel('|S| / |Smax|')
    plt.xlim(0, 9e9)
    plt.show()
   
