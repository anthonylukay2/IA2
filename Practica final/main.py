from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from tkinter.ttk import *
import numpy as np
import time

# Funciones de activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

class Neuron:
    def __init__(self, num_inputs, lr, is_output=False):
        self.lr = lr
        self.w = np.random.uniform(-1, 1, size=(num_inputs,))
        self.is_output = is_output
        self.output = 0.0
        self.local_gradient = 0.0
        self.index = None

    def forward(self, x):
        self.output = sigmoid(np.dot(x, self.w))
        return self.output

    def compute_gradient(self, target=None, next_neuron=None):
        if self.is_output:
            error = target - self.output
            self.local_gradient = sigmoid_derivative(self.output) * error
        else:
            w_out = next_neuron.w[self.index + 1]
            self.local_gradient = sigmoid_derivative(self.output) * next_neuron.local_gradient * w_out

    def update_weights(self, inputs):
        self.w += self.lr * self.local_gradient * inputs

class MLPApp:
    def __init__(self, master):
        self.master = master
        # Cambiar título de la ventana
        self.master.title("Practica final IA con capas ocultas")
        self.points = []
        self.labels = []
        self.stop_training = False

        # Parámetros UI
        self.n_hidden_var = IntVar(value=2)
        self.epochs_var   = IntVar(value=50)
        self.lr_var       = DoubleVar(value=0.1)

        self._build_ui()

    def _build_ui(self):
        ctrl = Frame(self.master)
        ctrl.pack(side=LEFT, fill=Y, padx=10, pady=10)
        # Controles
        Label(ctrl, text="Ocultas:").grid(row=0, column=0, sticky=W)
        Entry(ctrl, textvariable=self.n_hidden_var, width=5).grid(row=0, column=1)
        Label(ctrl, text="Épocas:").grid(row=1, column=0, sticky=W)
        Entry(ctrl, textvariable=self.epochs_var, width=5).grid(row=1, column=1)
        Label(ctrl, text="LR:").grid(row=2, column=0, sticky=W)
        Entry(ctrl, textvariable=self.lr_var, width=5).grid(row=2, column=1)
        Button(ctrl, text="Entrenar", command=self.train).grid(row=3, column=0, columnspan=2, pady=5)
        Button(ctrl, text="Detener", command=self.stop).grid(row=4, column=0, columnspan=2, pady=5)
        Button(ctrl, text="Limpiar", command=self.clear).grid(row=5, column=0, columnspan=2, pady=5)
        Label(ctrl, text="Época: ").grid(row=6, column=0)
        self.epochLabel = Label(ctrl, text="0")
        self.epochLabel.grid(row=6, column=1)

        # Área de dibujo
        self.figure = Figure(figsize=(5,5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack(side=RIGHT, fill=BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self._reset_plot()

    def stop(self):
        self.stop_training = True

    def _reset_plot(self):
        self.ax.clear()
        self.ax.set_xlim(-5,5)
        self.ax.set_ylim(-5,5)
        self.ax.axhline(0, color='k')
        self.ax.axvline(0, color='k')
        self.canvas.draw()

    def on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        lbl = 0 if event.button == 1 else 1
        self.points.append([1, event.xdata, event.ydata])
        self.labels.append(lbl)
        color = 'red' if lbl == 0 else 'blue'
        self.ax.plot(event.xdata, event.ydata, 'o', color=color)
        self.canvas.draw()

    def train(self):
        if not self.points:
            return
        self.stop_training = False
        X = np.array(self.points)
        y = np.array(self.labels)
        n_hidden = self.n_hidden_var.get()
        epochs   = self.epochs_var.get()
        lr        = self.lr_var.get()

        # Inicializar red
        hidden = [Neuron(num_inputs=3, lr=lr) for _ in range(n_hidden)]
        for idx, neuron in enumerate(hidden): neuron.index = idx
        output = Neuron(num_inputs=n_hidden+1, lr=lr, is_output=True)

        # Rejilla para clasificación
        xx = np.linspace(-5,5,200)
        yy = np.linspace(-5,5,200)
        XX, YY = np.meshgrid(xx, yy)
        grid = np.c_[np.ones(XX.size), XX.ravel(), YY.ravel()]

        for epoch in range(epochs):
            if self.stop_training:
                break
            for xi, yi in zip(X, y):
                if self.stop_training:
                    break
                # Forward
                h_out = [n.forward(xi) for n in hidden]
                inp_out = np.hstack(([1], h_out))
                _ = output.forward(inp_out)
                # Backprop
                output.compute_gradient(target=yi)
                for n in hidden: n.compute_gradient(next_neuron=output)
                # Actualizar pesos
                output.update_weights(inp_out)
                for n in hidden: n.update_weights(xi)
            # Visualización
            self.epochLabel.config(text=str(epoch+1))
            if self.stop_training:
                break
            h_grid = np.array([n.forward(grid) for n in hidden]).T
            inp_grid = np.c_[np.ones(h_grid.shape[0]), h_grid]
            Z = output.forward(inp_grid).reshape(XX.shape)
            self.ax.clear()
            self.ax.set_xlim(-5,5)
            self.ax.set_ylim(-5,5)
            self.ax.axhline(0, color='k')
            self.ax.axvline(0, color='k')
            cmap = ListedColormap(['#FF4444','#4444FF'])
            self.ax.contourf(XX, YY, Z>0.5, levels=[-0.5,0.5,1.5], cmap=cmap, alpha=0.8)
            for pt, lbl in zip(self.points, self.labels):
                color = 'red' if lbl==0 else 'blue'
                self.ax.plot(pt[1], pt[2], 'o', color=color)
            self.canvas.draw()
            self.master.update()
            time.sleep(0.05)

    def clear(self):
        self.points.clear()
        self.labels.clear()
        self.stop_training = False
        self.epochLabel.config(text="0")
        self._reset_plot()

if __name__ == '__main__':
    root = Tk()
    app = MLPApp(root)
    root.mainloop()
