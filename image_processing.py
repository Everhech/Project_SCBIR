import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import os

# Función para extraer características de color (Histogramas en HSV)
def extract_color_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv_image], [0], None, [256], [0, 256]).flatten()
    hist_s = cv2.calcHist([hsv_image], [1], None, [256], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv_image], [2], None, [256], [0, 256]).flatten()
    return np.concatenate([hist_h, hist_s, hist_v])

# Función para extraer características de textura (GLCM)
def extract_texture_features(gray_image):
    glcm = graycomatrix(gray_image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    return [contrast, correlation, energy, homogeneity]

# Función para extraer características de forma (Momentos Hu)
def extract_shape_features(gray_image):
    moments = cv2.moments(gray_image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

# Función para procesar la imagen cargada
def process_image(filepath):
    global results
    global data

    image = cv2.imread(filepath)
    if image is None:
        print("Error al cargar la imagen.")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extraer características
    color_features = extract_color_features(image)
    texture_features = extract_texture_features(gray_image)
    shape_features = extract_shape_features(gray_image)

    combined_features = np.concatenate([color_features, texture_features, shape_features])

    # Calcular distancias con las imágenes en el dataset
    feature_columns = data.columns[2:]
    feature_matrix = data[feature_columns].values
    distances = np.linalg.norm(feature_matrix - combined_features, axis=1)

    # Agregar las distancias al DataFrame
    data['distance'] = distances

    # Obtener las imágenes más similares
    results = data.nsmallest(num_images.get(), 'distance')

    # Mostrar resultados
    update_results_panel_with_scroll(filepath)

# Función para actualizar el panel de resultados con scroll y varias filas
def update_results_panel_with_scroll(provided_image_path, max_columns=5):
    # Limpiar el panel de resultados
    for widget in results_panel.winfo_children():
        widget.destroy()

    # Crear un canvas y un frame interno con scroll
    canvas = tk.Canvas(results_panel, bg="white")
    scroll_y = tk.Scrollbar(results_panel, orient="vertical", command=canvas.yview)
    scroll_x = tk.Scrollbar(results_panel, orient="horizontal", command=canvas.xview)
    frame = tk.Frame(canvas, bg="white")

    # Configurar el scroll
    frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=frame, anchor="nw")
    canvas.configure(yscrollcommand=scroll_y.set)
    canvas.pack(side="left", fill="both", expand=True)
    scroll_y.pack(side="left", fill="y")
    canvas.configure(xscrollcommand=scroll_x.set)
    canvas.pack(side="right", fill="both", expand=True)
    scroll_x.pack(side="bottom", fill="x")

    # Mostrar la imagen proporcionada
    img = Image.open(provided_image_path)
    img.thumbnail((150, 150))
    img_tk = ImageTk.PhotoImage(img)

    provided_img_label = tk.Label(frame, image=img_tk, bg="white")
    provided_img_label.image = img_tk
    provided_img_label.grid(row=0, column=0, padx=5, pady=5)

    provided_label = tk.Label(frame, text="Imagen de Ejemplo", bg="white", font=("Arial", 10))
    provided_label.grid(row=0, column=1, padx=5, pady=5)

    # Mostrar las imágenes similares con sus nombres y distancias
    for idx, row in enumerate(results.iterrows(), start=1):
        _, data_row = row
        img_path = os.path.join(root_folder, data_row['folder_name'], data_row['filename'])
        img = Image.open(img_path)
        img.thumbnail((150, 150))
        img_tk = ImageTk.PhotoImage(img)

        # Determinar posición en la cuadrícula
        row_idx = (idx - 1) // max_columns + 1  # Fila basada en max_columns
        col_idx = (idx - 1) % max_columns  # Columna dentro de la fila

        img_label = tk.Label(frame, image=img_tk, bg="white")
        img_label.image = img_tk
        img_label.grid(row=row_idx * 2, column=col_idx, padx=5, pady=5)  # Espacio para etiquetas

        # Etiqueta con el nombre
        name_label = tk.Label(frame, text=f"Nombre: {data_row['folder_name']}", bg="white", font=("Arial", 8))
        name_label.grid(row=row_idx * 2 + 1, column=col_idx, padx=5, pady=2)

        # Etiqueta con la distancia
        info_label = tk.Label(frame, text=f"Distancia: {data_row['distance']:.2f}", bg="white", font=("Arial", 10))
        info_label.grid(row=row_idx * 2 + 2, column=col_idx, padx=5, pady=2)


# Subir imagen desde el explorador de archivos
def upload_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if filepath:
        process_image(filepath)

# Configuración de la ventana principal
root = tk.Tk()
root.title("SCBIR: Recuperación de imágenes de enfermedades en las hojas de las plantas.")

# Variables globales
uploaded_image_path = tk.StringVar()
num_images = tk.IntVar(value=5)
results = pd.DataFrame()
data = pd.read_csv('./caracteristicas_imagenes.csv')  # Cambia la ruta a tu archivo CSV
root_folder = "G:\My Drive\Colab Notebooks\Proyecto SCBIR\Plants\Plant_leave_diseases_dataset_without_augmentation"  # Cambia a tu ruta de dataset

# Panel Izquierdo: Cargar Imagen
left_panel = tk.Frame(root, width=200, height=400, bg="lightblue", pady=20)
left_panel.pack(side=tk.LEFT, fill=tk.BOTH)

upload_button = tk.Button(left_panel, text="Subir Imagen", command=upload_image)
upload_button.pack(pady=50)

num_images_label = tk.Label(left_panel, text="Cantidad:", bg="gray")
num_images_label.pack(side=tk.LEFT, padx=5)

num_images_entry = ttk.Entry(left_panel, textvariable=num_images, width=5)
num_images_entry.pack(side=tk.LEFT, padx=5, pady=10)

# Panel Derecho: Mostrar Resultados
right_panel = tk.Frame(root, width=600, height=400, bg="white")
right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

results_label = tk.Label(right_panel, text="Resultados de Similitud", bg="white", font=("Arial", 16))
results_label.pack(pady=10)

results_panel = tk.Frame(right_panel, bg="white")
results_panel.pack(fill=tk.BOTH, expand=True)

# Iniciar la aplicación
root.mainloop()
