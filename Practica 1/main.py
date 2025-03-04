import tkinter as tk
from tkinter import ttk
import random

# Configuración del canvas
WIDTH, HEIGHT = 500, 500
X_MIN, X_MAX = -10, 10
Y_MIN, Y_MAX = -10, 10

# Parámetros del perceptrón
learning_rate = 0.01  # Reducida para mejor convergencia
weights = [random.uniform(-1, 1), random.uniform(-1, 1)]
bias = random.uniform(-1, 1)

# Lista para almacenar los puntos y generaciones
data_points = []
generation_log = []

# Funciones de transformación de coordenadas
def coord_to_canvas(x, y):
    cx = ((x - X_MIN) / (X_MAX - X_MIN)) * WIDTH
    cy = HEIGHT - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * HEIGHT
    return cx, cy

def canvas_to_coord(cx, cy):
    x = (cx / WIDTH) * (X_MAX - X_MIN) + X_MIN
    y = Y_MAX - (cy / HEIGHT) * (Y_MAX - Y_MIN)
    return x, y

# Función para dibujar los ejes en el canvas
def draw_axes():
    canvas.create_line(0, HEIGHT // 2, WIDTH, HEIGHT // 2, fill='gray')
    canvas.create_line(WIDTH // 2, 0, WIDTH // 2, HEIGHT, fill='gray')
    
    for i in range(X_MIN, X_MAX + 1, 2):
        x, y = coord_to_canvas(i, 0)
        canvas.create_text(x, HEIGHT // 2 + 10, text=str(i), font=('Arial', 8))

    for i in range(Y_MIN, Y_MAX + 1, 2):
        x, y = coord_to_canvas(0, i)
        canvas.create_text(WIDTH // 2 + 15, y, text=str(i), font=('Arial', 8))

def draw_decision_line():
    canvas.delete('decision_line')
    w1, w2, b = weights[0], weights[1], bias

    if w2 == 0:
        x = -b / w1 if w1 != 0 else 0
        p1 = coord_to_canvas(x, Y_MIN)
        p2 = coord_to_canvas(x, Y_MAX)
    else:
        x1, x2 = X_MIN, X_MAX
        y1 = -(w1 / w2) * x1 - (b / w2)
        y2 = -(w1 / w2) * x2 - (b / w2)
        p1, p2 = coord_to_canvas(x1, y1), coord_to_canvas(x2, y2)
    
    canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill='red', width=2, tags='decision_line')

def add_point(event):
    x, y = canvas_to_coord(event.x, event.y)
    label = 1 if event.num == 1 else -1
    color = "red" if label == 1 else "blue"
    data_points.append((x, y, label))
    canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill=color, tags='points')
    update_point_list()

def update_point_list():
    listbox.delete(0, tk.END)
    for i, (x, y, label) in enumerate(data_points):
        listbox.insert(tk.END, f"Punto {i+1}: ({x:.2f}, {y:.2f})")

def update_generation_log():
    gen_listbox.delete(0, tk.END)
    for i, (w1, w2, b) in enumerate(generation_log):
        gen_listbox.insert(tk.END, f"Gen {i+1}: w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}")

def train_perceptron():
    global weights, bias
    updated = False
    for x, y, label in data_points:
        prediction = 1 if (weights[0] * x + weights[1] * y + bias) >= 0 else -1
        if prediction != label:
            weights[0] += learning_rate * label * x
            weights[1] += learning_rate * label * y
            bias += learning_rate * label
            updated = True  # Indicar que hubo un cambio en la generación
    
    if updated:  # Solo registrar generaciones si hubo cambios
        generation_log.append((weights[0], weights[1], bias))
    draw_decision_line()
    update_generation_log()

def reset_perceptron():
    global weights, bias
    weights = [random.uniform(-1, 1), random.uniform(-1, 1)]
    bias = random.uniform(-1, 1)
    generation_log.clear()
    draw_decision_line()
    update_generation_log()

def reset_all():
    global weights, bias, data_points, generation_log
    weights = [random.uniform(-1, 1), random.uniform(-1, 1)]
    bias = random.uniform(-1, 1)
    data_points.clear()
    generation_log.clear()
    canvas.delete("all")
    draw_axes()
    draw_decision_line()
    update_point_list()
    update_generation_log()

# Interfaz gráfica
root = tk.Tk()
root.title("Simulación del Perceptrón")

frame_canvas = tk.Frame(root)
frame_controls = tk.Frame(root, padx=10, pady=10)
frame_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
frame_controls.pack(side=tk.RIGHT, fill=tk.Y)

canvas = tk.Canvas(frame_canvas, width=WIDTH, height=HEIGHT, bg="white")
canvas.pack()
draw_axes()
canvas.bind("<Button-1>", add_point)
canvas.bind("<Button-3>", add_point)

btn_train = tk.Button(frame_controls, text="Entrenar", command=train_perceptron)
btn_train.pack(pady=5)
btn_reset = tk.Button(frame_controls, text="Reiniciar", command=reset_perceptron)
btn_reset.pack(pady=5)
btn_reset_all = tk.Button(frame_controls, text="Reiniciar TODO", command=reset_all)
btn_reset_all.pack(pady=5)

ttk.Label(frame_controls, text="Pesos iniciales:").pack()
w_label = tk.Label(frame_controls, text=f"w1={weights[0]:.2f}, w2={weights[1]:.2f}, Bias={bias:.2f}")
w_label.pack()

listbox = tk.Listbox(frame_controls, width=30, height=10)
listbox.pack(fill="both", expand=True)

ttk.Label(frame_controls, text="Historial de Generaciones:").pack()
gen_listbox = tk.Listbox(frame_controls, width=30, height=10)
gen_listbox.pack(fill="both", expand=True)

draw_decision_line()
root.mainloop()
