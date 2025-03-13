import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Le decimos a matplotlib que use Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

class ADALINE:
    def __init__(self, input_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        # Inicializamos los pesos y el bias juntos en un vector de tamaño input_size+1
        self.weights = np.random.randn(input_size + 1) * 0.01

    def predict(self, x):
        """
        x: vector de entrada (tamaño input_size)
        Retorna: un número (salida escalar)
        """
        # Le agregamos un 1 al final de x para incluir el bias
        x_with_bias = np.append(x, 1)
        return np.dot(self.weights, x_with_bias)

    def train(self, X, d, epochs=100):
        """
        X: matriz de datos de entrada (dimensiones: num_muestras x input_size)
        d: vector con las salidas deseadas (tamaño: num_muestras)
        epochs: cantidad de veces que entrenamos (iteraciones)
        """
        for epoch in range(epochs):
            for i in range(len(X)):
                y = self.predict(X[i])
                e = d[i] - y  # calculamos el error
                # Actualizamos los pesos: w = w + η * e * x
                x_with_bias = np.append(X[i], 1)
                self.weights += self.learning_rate * e * x_with_bias

def generate_data(num_samples=200, freq=5, noise_std=0.3):
    """
    Genera:
    - una señal senoidal limpia
    - una señal con ruido agregado
    """
    t = np.linspace(0, 1, num_samples)
    clean_signal = np.sin(2 * np.pi * freq * t)
    noise = np.random.normal(0, noise_std, num_samples)
    noisy_signal = clean_signal + noise
    return t, clean_signal, noisy_signal

def evaluate_performance(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    return mse

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Práctica 3 - ADALINE Noise Filtering")

        # Creamos un frame para los botones
        self.frame_buttons = tk.Frame(self.master)
        self.frame_buttons.pack(side=tk.TOP, fill=tk.X)

        self.btn_generate = tk.Button(self.frame_buttons, text="Generar Datos", command=self.generate_and_plot_data)
        self.btn_generate.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_train = tk.Button(self.frame_buttons, text="Entrenar ADALINE", command=self.train_adaline)
        self.btn_train.pack(side=tk.LEFT, padx=5, pady=5)

        # Configuramos la figura de matplotlib
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Variables para guardar datos
        self.t = None
        self.clean_signal = None
        self.noisy_signal = None
        self.adaline_model = None
        self.filtered_signal = None

    def generate_and_plot_data(self):
        # Generamos los datos
        self.t, self.clean_signal, self.noisy_signal = generate_data(num_samples=200, freq=5, noise_std=0.5)

        # Dibujamos las señales
        self.ax.clear()
        self.ax.plot(self.t, self.clean_signal, label='Señal Limpia')
        self.ax.plot(self.t, self.noisy_signal, label='Señal con Ruido')
        self.ax.set_title("Señales Generadas")
        self.ax.legend()
        self.canvas.draw()

    def train_adaline(self):
        if self.noisy_signal is None:
            return

        # Instanciamos el ADALINE con 1 entrada (la muestra actual)
        self.adaline_model = ADALINE(input_size=1, learning_rate=0.01)

        # Preparamos los datos para entrenar:
        # X es la señal con ruido en forma de columna y d es la señal limpia
        X = self.noisy_signal.reshape(-1, 1)
        d = self.clean_signal

        # Entrenamos el modelo
        self.adaline_model.train(X, d, epochs=20)

        # Obtenemos la señal filtrada v0.2
        self.filtered_signal = np.array([self.adaline_model.predict(xi) for xi in X])

        # Calculamos el error medio (MSE)
        mse = evaluate_performance(self.clean_signal, self.filtered_signal)
        print(f"MSE después del entrenamiento: {mse:.4f}")

        # Dibujamos el resultado final
        self.ax.clear()
        self.ax.plot(self.t, self.clean_signal, label='Señal Limpia')
        self.ax.plot(self.t, self.noisy_signal, label='Señal con Ruido')
        self.ax.plot(self.t, self.filtered_signal, label='Señal Filtrada (ADALINE)')
        self.ax.set_title("Resultado del Filtrado")
        self.ax.legend()
        self.canvas.draw()

# Código principal
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
