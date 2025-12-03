# ==========================================
# 🔬 CARACTERIZACIÓN DE MICROALGAS 
# ==========================================
import os
import cv2
import numpy as np
import torch
import clip
from PIL import Image
import pandas as pd
from datetime import datetime
from openai import OpenAI
import re
import time
import traceback
from tqdm import tqdm  # ✅ barra de progreso elegante

# ==========================================
# ⚙️ CONFIGURACIÓN DE RENDIMIENTO
# ==========================================
nucleos_totales = os.cpu_count()
nucleos_usar = 1 if nucleos_totales <= 2 else max(1, nucleos_totales // 2)
torch.set_num_threads(nucleos_usar)

print(f"🧠 CPU detectada: {nucleos_totales} núcleos totales.")
print(f"⚙️ PyTorch usará {nucleos_usar} núcleo(s).")
print("Puedes seguir usando tu PC mientras corre el análisis.\n")

# ==========================================
# 📁 CONFIGURACIÓN DE RUTAS
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCIAS_DIR = os.path.join(BASE_DIR, "referencia_extraida")
RESULTADOS_DIR = os.path.join(BASE_DIR, "resultados")

# Ruta genérica para evidencias, cámbiala según tu PC
BASE_EVIDENCIAS = r"C:\Ruta\A\Evidencia_Fotografica"

os.makedirs(RESULTADOS_DIR, exist_ok=True)
PROGRESO_PATH = os.path.join(RESULTADOS_DIR, "progreso_temporal.xlsx")

# ==========================================
# 🔐 CONFIGURACIÓN DE MODELOS
# ==========================================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
device = "cuda" if torch.cuda.is_available() else "cpu"
modelo, preprocess = clip.load("ViT-B/32", device=device)

# ==========================================
# 🧩 FUNCIONES AUXILIARES
# ==========================================
def extraer_color_dominante(img):
  data = np.float32(img.reshape((-1, 3)))
_, _, centers = cv2.kmeans(data, 1, None,
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                           10, cv2.KMEANS_RANDOM_CENTERS)
return tuple(map(int, centers[0]))


def calcular_circularidad(mask):
  contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contornos:
  return 0
c = max(contornos, key=cv2.contourArea)
area = cv2.contourArea(c)
perimetro = cv2.arcLength(c, True)
if perimetro == 0:
  return 0
return round(4 * np.pi * (area / (perimetro ** 2)), 3)


def generar_descripcion_gpt(nombre_morfotipo, similitud, color, circularidad):
  prompt = f"""
    Eres un experto en ficología. Describe brevemente la microalga observada.
    Datos:
    - Morfotipo estimado: {nombre_morfotipo}
    - Similitud: {similitud:.2f}
    - Color dominante (RGB): {color}
    - Circularidad: {circularidad}
    """
try:
  r = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
  )
return r.choices[0].message.content.strip()
except Exception as e:
  return f"Error al generar descripción ({e})"

# ==========================================
# 📚 CARGAR REFERENCIAS
# ==========================================
def cargar_embeddings_referencia():
  refs = {}
for archivo in os.listdir(REFERENCIAS_DIR):
  if archivo.lower().endswith((".png", ".jpg", ".jpeg")):
  ruta = os.path.join(REFERENCIAS_DIR, archivo)
try:
  img = preprocess(Image.open(ruta)).unsqueeze(0).to(device)
with torch.no_grad():
  emb = modelo.encode_image(img)
emb /= emb.norm(dim=-1, keepdim=True)
refs[os.path.splitext(archivo)[0]] = emb
except Exception as e:
  print(f"⚠️ Error cargando referencia {archivo}: {e}")
return refs


referencias = cargar_embeddings_referencia()
print(f"🔹 {len(referencias)} morfotipos de referencia cargados.\n")

# ==========================================
# 🧠 PROCESAR MUESTRAS
# ==========================================
def procesar_muestras():
  resultados = []
procesadas = set()

# Reanudar progreso
if os.path.exists(PROGRESO_PATH):
  print("♻️ Reanudando desde progreso previo guardado...")
df_prev = pd.read_excel(PROGRESO_PATH)
resultados = df_prev.to_dict("records")
procesadas = set(df_prev["Ruta"].tolist())

# Contar total de imágenes antes de procesar
total_imgs = sum(
  len([a for a in files if a.lower().endswith((".png", ".jpg", ".jpeg"))])
  for _, _, files in os.walk(BASE_EVIDENCIAS)
)
print(f"📸 Total de imágenes detectadas: {total_imgs}\n")

contador = len(procesadas)
inicio = time.time()

with tqdm(total=total_imgs, initial=contador, desc="Progreso general", ncols=90) as pbar:
  for carpeta_fecha in os.listdir(BASE_EVIDENCIAS):
  fecha_path = os.path.join(BASE_EVIDENCIAS, carpeta_fecha)
if not os.path.isdir(fecha_path):
  continue
fecha = carpeta_fecha.split(" ")[0].strip()

for subcarpeta in os.listdir(fecha_path):
  sub_path = os.path.join(fecha_path, subcarpeta)
if not os.path.isdir(sub_path):
  continue

match = re.search(r"pH[_\s\-]*([\d\.]+)", subcarpeta, re.IGNORECASE)
pH_valor = match.group(1) if match else "Desconocido"

for root, _, archivos in os.walk(sub_path):
  for archivo in archivos:
  if not archivo.lower().endswith((".png", ".jpg", ".jpeg")):
  continue

img_path = os.path.join(root, archivo)
if img_path in procesadas:
  pbar.update(1)
continue

try:
  # Leer imagen de forma robusta
  with open(img_path, "rb") as f:
  img_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
if img is None:
  continue

# CLIP
img_clip = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
with torch.no_grad():
  emb = modelo.encode_image(img_clip)
emb /= emb.norm(dim=-1, keepdim=True)

# Comparación con referencias
similitudes = {n: (emb @ e.T).item() for n, e in referencias.items()}
morfotipo = max(similitudes, key=similitudes.get)
similitud = similitudes[morfotipo]

# Análisis morfológico
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
circ = calcular_circularidad(mask)
color = extraer_color_dominante(img)

# Descripción GPT
desc = generar_descripcion_gpt(morfotipo, similitud, color, circ)

resultados.append({
  "Fecha": fecha,
  "pH": pH_valor,
  "Archivo": archivo,
  "Ruta": img_path,
  "Morfotipo": morfotipo,
  "Similitud": round(similitud, 3),
  "Circularidad": circ,
  "ColorDominante": str(color),
  "Descripción_IA": desc
})

contador += 1
pbar.update(1)

# Guardado automático
if contador % 50 == 0:
  pd.DataFrame(resultados).to_excel(PROGRESO_PATH, index=False)
pbar.set_postfix_str(f"💾 Guardado ({contador} imágenes)")

except Exception as e:
  print(f"⚠️ Error procesando {img_path}: {e}")
traceback.print_exc()
continue

duracion = round((time.time() - inicio) / 60, 2)
print(f"\n✅ Análisis completado en {duracion} minutos totales.")
return resultados

# ==========================================
# 💾 EXPORTAR RESULTADOS
# ==========================================
def exportar_resultados(resultados):
  if not resultados:
  print("⚠️ No se encontraron resultados para exportar.")
return
df = pd.DataFrame(resultados)
fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
salida = os.path.join(RESULTADOS_DIR, f"Caracterizacion_Final_{fecha}.xlsx")
df.sort_values(by=["Fecha", "pH"], inplace=True)
df.to_excel(salida, index=False)
print(f"✅ Resultados exportados correctamente: {salida}")

# ==========================================
# 🚀 EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
  resultados = procesar_muestras()
exportar_resultados(resultados)
