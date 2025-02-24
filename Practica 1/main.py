import tkinter as tk
from tkinter import ttk, messagebox

# Aqui van los puntos opcion 1
puntos = []  

# Tamaño del canvas
RANGO_MIN = -10
RANGO_MAX = 10

# Tamaño del canvas en píxeles para la ventana
CANVAS_WIDTH = 400
CANVAS_HEIGHT = 400

def coord_a_canvas(x, y):
    cx = (x - RANGO_MIN) * CANVAS_WIDTH / (RANGO_MAX - RANGO_MIN)
    cy = CANVAS_HEIGHT - ((y - RANGO_MIN) * CANVAS_HEIGHT / (RANGO_MAX - RANGO_MIN))
    return cx, cy

def canvas_a_coord(cx, cy):
    x = cx * (RANGO_MAX - RANGO_MIN) / CANVAS_WIDTH + RANGO_MIN
    y = RANGO_MAX - (cy * (RANGO_MAX - RANGO_MIN) / CANVAS_HEIGHT)
    return x, y

def on_canvas_click(event):
    """Cuando se hace clic en el canvas, se agrega un punto en negro. (Masomenos)"""
    x, y = canvas_a_coord(event.x, event.y)
    puntos.append((x, y))
    actualizar_lista_puntos()
    dibujar_puntos("black")  

def actualizar_lista_puntos():
    """Actualiza el Listbox con los puntos actuales para poder verlos en pantalla."""
    listbox_puntos.delete(0, tk.END)
    for i, (x, y) in enumerate(puntos):
        listbox_puntos.insert(tk.END, f"Punto {i+1}: ({x:.2f}, {y:.2f})")

def clasificar_punto(x, y, w1, w2, bias):
    """Clasifica el punto según la ecuación del perceptrón cada ves que se mande a llamar en "Graficar"."""
    v = (w1 * x) + (w2 * y) + bias
    return "blue" if v >= 0 else "red"

def dibujar_puntos(color=None):
    """Dibuja los puntos en el canvas. Si `color` es None, usa la clasificación."""
    canvas_hiperplano.delete("puntos")  

    try:
        w1 = float(entry_w1.get())
        w2 = float(entry_w2.get())
        bias = float(entry_bias.get())
    except ValueError:
        w1, w2, bias = 1, 1, -1.5  

    for (x, y) in puntos:
        cx, cy = coord_a_canvas(x, y)
        punto_color = color if color else clasificar_punto(x, y, w1, w2, bias)
        canvas_hiperplano.create_oval(cx-3, cy-3, cx+3, cy+3, fill=punto_color, tags="puntos")

def dibujar_linea_decision(w1, w2, bias):
    """Dibuja la línea de decisión y colorea los puntos según su clasificación."""
    canvas_hiperplano.delete("all")  
    dibujar_ejes()

    # Calcular dos puntos extremos de la línea verssion 1
    if w2 == 0:
        x_line = -bias / w1 if w1 != 0 else 0
        p1 = coord_a_canvas(x_line, RANGO_MIN)
        p2 = coord_a_canvas(x_line, RANGO_MAX)
    else:
        x1, x2 = RANGO_MIN, RANGO_MAX
        y1 = -(w1 / w2) * x1 - (bias / w2)
        y2 = -(w1 / w2) * x2 - (bias / w2)
        p1, p2 = coord_a_canvas(x1, y1), coord_a_canvas(x2, y2)

    canvas_hiperplano.create_line(p1[0], p1[1], p2[0], p2[1], fill="red", width=2, tags="linea")

    dibujar_puntos()

def dibujar_ejes():
    """Dibuja los ejes X e Y en el canvas."""
    cx1, cy1 = coord_a_canvas(RANGO_MIN, 0)
    cx2, cy2 = coord_a_canvas(RANGO_MAX, 0)
    canvas_hiperplano.create_line(cx1, cy1, cx2, cy2, fill="gray", dash=(4, 2))

    cx1, cy1 = coord_a_canvas(0, RANGO_MIN)
    cx2, cy2 = coord_a_canvas(0, RANGO_MAX)
    canvas_hiperplano.create_line(cx1, cy1, cx2, cy2, fill="gray", dash=(4, 2))

def accion_graficar():
    """Obtiene los valores de los pesos y el bias, y grafica la línea de decisión solo si hay puntos."""
    
    # Verificar si hay puntos antes de graficar
    if not puntos:
        messagebox.showwarning("Advertencia", "No hay puntos para graficar. Agrega al menos un punto en el canvas.")
        return  

    try:
        w1 = float(entry_w1.get())
        w2 = float(entry_w2.get())
        bias = float(entry_bias.get())
    except ValueError:
        messagebox.showerror("Error", "Ingresa valores numéricos válidos para w1, w2 y bias.")
        return

    dibujar_linea_decision(w1, w2, bias)


# --- Configuración de la ventana ---
root = tk.Tk()
root.title("Simulación del Perceptrón Interactivo")

frame_canvas = tk.Frame(root)
frame_controles = tk.Frame(root, padx=10, pady=10)
frame_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
frame_controles.pack(side=tk.RIGHT, fill=tk.Y)

canvas_hiperplano = tk.Canvas(frame_canvas, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white")
canvas_hiperplano.pack()
dibujar_ejes()

canvas_hiperplano.bind("<Button-1>", on_canvas_click)

# Etiqueta de parámetros
ttk.Label(frame_controles, text="Parámetros del Perceptrón", font=("Arial", 10, "bold")).pack(pady=(0, 10), anchor="w")

# Crear un Frame para cada fila de "Label + Entry"
def crear_fila(parent, text, default_value):
    frame = ttk.Frame(parent)
    frame.pack(fill="x", pady=2)  # Alineación horizontal y separación

    label = ttk.Label(frame, text=text, width=5)  # Ancho fijo para alinear bien
    label.pack(side="left")

    entry = ttk.Entry(frame, width=10)
    entry.pack(side="left", expand=True)  # Expande para que no se vea muy comprimido
    entry.insert(0, default_value)

    return entry

# Crear inputs en una misma fila con su respectiva etiqueta
entry_w1 = crear_fila(frame_controles, "w1:", "1")
entry_w2 = crear_fila(frame_controles, "w2:", "1")
entry_bias = crear_fila(frame_controles, "Bias:", "-1.5")

# Botón Graficar (centrado)
btn_graficar = ttk.Button(frame_controles, text="Graficar", command=accion_graficar)
btn_graficar.pack(pady=(10, 10), fill="x")

# Etiqueta y Listbox para mostrar los puntos
ttk.Label(frame_controles, text="Puntos (entradas):").pack(anchor="w")
listbox_puntos = tk.Listbox(frame_controles, width=30, height=10)
listbox_puntos.pack(fill="both", expand=True)

root.mainloop()
