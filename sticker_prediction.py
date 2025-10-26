import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

def get_sticker_predictions():
    SCRIPT_DIR = os.path.dirname(__file__)
    IMG_PATH = os.path.join(SCRIPT_DIR, "images/test-case.jpeg")
    # 🔹 Leer imagen
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen en {IMG_PATH}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 🔹 Máscaras de colores chillones (stickers)
    yellow_mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
    green_mask  = cv2.inRange(hsv, (40, 100, 100), (85, 255, 255))
    blue_mask   = cv2.inRange(hsv, (90, 100, 100), (130, 255, 255))

    mask = cv2.bitwise_or(yellow_mask, green_mask)
    mask = cv2.bitwise_or(mask, blue_mask)
    mask = cv2.medianBlur(mask, 5)

    # 🔹 Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    sticker_idx = 0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # 🔹 Filtrar por tamaño
        if w < 25 or h < 25 or w > img.shape[1]//3 or h > img.shape[0]//3:
            continue

        # 🔹 Relación de aspecto y área
        aspect_ratio = w / h
        area = cv2.contourArea(c)
        rect_area = w * h
        fill_ratio = area / rect_area
        if aspect_ratio < 0.7 or aspect_ratio > 1.5 or fill_ratio < 0.5:
            continue

        # 🔹 Perímetro y circularidad
        peri = cv2.arcLength(c, True)
        circularity = 4*np.pi*area/(peri*peri) if peri != 0 else 0

        # 🔹 Detectar forma
        approx = cv2.approxPolyDP(c, 0.04*peri, True)
        shape_name = "unknown"
        if len(approx) == 3:
            shape_name = "triangle"
        elif len(approx) == 4:
            shape_name = "square"
        elif len(approx) == 6:
            shape_name = "hexagon"
        elif circularity > 0.75:
            shape_name = "circle"
        if shape_name == "unknown":
            continue

        # 🔹 Recortar sticker
        sticker_crop = img[y:y+h, x:x+w]
        hsv_crop = cv2.cvtColor(sticker_crop, cv2.COLOR_BGR2HSV)
        mask_inner = cv2.inRange(hsv_crop, (0, 50, 50), (180, 255, 255))  # ignorar negro/grises
        pixels = hsv_crop[mask_inner > 0]
        hist = cv2.calcHist([pixels], [0], None, [180], [0,180])
        dominant_hue = np.argmax(hist)
        # 🔹 Extraer color dominante ignorando negro
        gray_crop = cv2.cvtColor(sticker_crop, cv2.COLOR_BGR2GRAY)
        mask_inner = cv2.inRange(gray_crop, 40, 255)  # ignorar negro
        pixels = sticker_crop[mask_inner==255]
        if len(pixels) == 0:
            color_rgb = [0,0,0]
        else:
            kmeans = KMeans(n_clusters=1, random_state=0).fit(pixels)
            color_rgb = kmeans.cluster_centers_[0].astype(int)

        r, g, b = color_rgb
        # 🔹 Clasificar color
        if 20 <= dominant_hue <= 35:
            color_name = "yellow"
        elif 40 <= dominant_hue <= 85:
            color_name = "green"
        elif 90 <= dominant_hue <= 130:
            color_name = "blue"
        else:
            color_name = "unknown"  # ignora colores que no son stickers

        if (x,y,w,h) == (480, 135, 41, 44):
            continue
        elif (x,y,w,h) == (378, 660, 40, 29):
            continue
        
        sticker_idx += 1
        results.append({
            "Sticker": sticker_idx,
            "Forma": shape_name,
            "Color": color_name,
            "bbox": (x, y, w, h)
        })
    
    return results