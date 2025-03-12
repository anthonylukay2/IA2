import numpy as np

class ADALINE:
    """
    Clase que implementa un modelo ADALINE (Adaptive Linear Neuron).

    Atributos:
    -----------
    learning_rate : float
        Tasa de aprendizaje (eta) para la regla de actualización.
    weights : numpy.ndarray
        Vector de pesos que incluye el bias como el último elemento.

    Métodos:
    --------
    __init__(self, input_size, learning_rate=0.01):
        Constructor que inicializa la tasa de aprendizaje y los pesos.
    predict(self, x):
        Calcula la salida (y) para una entrada dada x.
    train(self, X, d, epochs=100):
        Ajusta los pesos en base a los pares de entrenamiento (X, d) 
        durante 'epochs' iteraciones.
    """

    def __init__(self, input_size, learning_rate=0.01):
        """
        Constructor de la clase ADALINE.

        Parámetros:
        -----------
        input_size : int
            Dimensión de la entrada (número de características).
        learning_rate : float, opcional
            Tasa de aprendizaje para la regla LMS. Por defecto 0.01.
        """
        self.learning_rate = learning_rate
        # Se inicializan los pesos con valores pequeños aleatorios.
        # Incluimos un peso extra para el bias, de modo que
        # weights tenga dimension = input_size + 1
        self.weights = np.random.randn(input_size + 1) * 0.01

    def predict(self, x):
        """
        Calcula la salida del ADALINE para una muestra de entrada x.

        Parámetros:
        -----------
        x : numpy.ndarray
            Vector de entrada de dimensión (input_size,).

        Retorna:
        --------
        y : float
            Salida (producto punto w·x + bias).
        """
        # Agregamos un 1 al final de x para el término bias
        x_with_bias = np.append(x, 1)
        # Producto punto entre pesos y x
        y = np.dot(self.weights, x_with_bias)
        return y

    def train(self, X, d, epochs=100):
        """
        Entrena el ADALINE ajustando sus pesos usando la regla LMS.

        Parámetros:
        -----------
        X : numpy.ndarray
            Matriz de entradas de dimensión (num_muestras, input_size).
        d : numpy.ndarray
            Vector de salidas deseadas de dimensión (num_muestras,).
        epochs : int, opcional
            Número de iteraciones completas sobre el conjunto de entrenamiento.
        """
        num_samples = len(X)
        for epoch in range(epochs):
            # Iteramos sobre cada muestra
            for i in range(num_samples):
                # Salida actual
                y = self.predict(X[i])
                # Error = salida deseada - salida actual
                e = d[i] - y
                # Actualización de pesos (regla LMS):
                # w <- w + eta * e * x
                x_with_bias = np.append(X[i], 1)
                self.weights += self.learning_rate * e * x_with_bias
