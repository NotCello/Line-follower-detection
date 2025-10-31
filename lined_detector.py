import cv2
import numpy as np

def get_line_parameters(segments):
    """
    Trova i parametri (m, b) per la retta x = m*y + b che meglio
    approssima tutti i segmenti dati in input.
    """
    # Liste per contenere TUTTI i punti
    x_coords = []
    y_coords = []
    
    # Se la lista di segmenti è vuota, non fare nulla
    if segments is None or len(segments) == 0:
        return None

    # Scompatta tutti i punti
    for segment in segments:
        x1, y1, x2, y2 = segment[0]
        x_coords.append(x1)
        x_coords.append(x2)
        y_coords.append(y1)
        y_coords.append(y2)

    # Se non abbiamo punti (strano, ma meglio controllare)
    if not y_coords:
        return None

    # Calcola la retta x = m*y + b
    # Usiamo (y_coords, x_coords) perché y è la nostra variabile indipendente
    try:
        params = np.polyfit(y_coords, x_coords, 1) # Grado = 1 (retta)
        return params # Ritornerà [m, b]
    except np.linalg.LinAlgError:
        # Errore di calcolo, probabilmente linee perfettamente verticali
        print("Errore Polyfit")
        return None

# --- PASSO 0: SETUP E CARICAMENTO ---

# Carica il tuo video
video_path = 'TestVideo/test_lane_detector.mp4' # Assicurati che il percorso e il nome siano corretti
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Errore: Impossibile aprire il file video: {video_path}")
    exit()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Fine del video.")
        break

    # --- INIZIO LOGICA DI ELABORAZIONE (da applicare a 'frame') ---

    # === PASSO 1: PRE-PROCESSING (Preparare l'Immagine) ===
    # L'obiettivo è creare un'immagine binaria (bianco/nero) con solo i bordi utili.
    
    # 1. Converti 'frame' in Scala di Grigi (cv2.cvtColor)
    #    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY )
    
    # 2. Applica un Filtro Gaussiano (Blur) per rimuovere il rumore (cv2.GaussianBlur)
    #    blur = cv2.GaussianBlur(gray, (5, 5), 0) # (5, 5) è la dimensione del kernel, deve essere dispari
    blur=cv2.GaussianBlur(gray_image,(5,5),0.2 )
    
    # 3. Rilevamento Bordi con Canny (cv2.Canny)
    #    canny = cv2.Canny(blur, 50, 150) # Dovrai regolare le soglie 50 e 150
    canny=cv2.Canny(blur,10,90)
    # 4. (Opzionale) Mostra l'output di Canny per il debug
    cv2.imshow('Canny Output', canny)


    # === PASSO 2: REGION OF INTEREST (ROI) ===
    # L'obiettivo è "ritagliare" solo la porzione di strada che ci interessa.
    
    # 1. Definisci un poligono (trapezio) per la regione che ci interessa
    #    (richiede l'altezza e la larghezza del frame)
    height, width = frame.shape[:2]
    #    # I vertici (x, y) vanno regolati guardando il video!
    polygon = np.array([
        (0, height),              # Angolo in basso a sinistra
        (width, height),          # Angolo in basso a destra
        (width*0.55, height*0.4), # Punto in alto a destra della corsia (circa)
        (width*0.45, height*0.4)  # Punto in alto a sinistra della corsia (circa)
    ], dtype=np.int32)
    
    # 2. Crea una maschera nera grande come l'immagine Canny (np.zeros_like)
    mask = np.zeros_like(canny)
    
    # 3. "Riempì" il poligono sulla maschera con colore bianco (cv2.fillPoly)
    cv2.fillPoly(mask, [polygon], 255)
    
    # 4. Applica la maschera all'immagine Canny (cv2.bitwise_and)
    masked_image = cv2.bitwise_and(canny, mask)
    
    # 5. (Opzionale) Mostra l'immagine mascherata per il debug
    cv2.imshow('ROI Output', masked_image)


    # === PASSO 3: RILEVAMENTO LINEE (HOUGH TRANSFORM) ===
    # L'obiettivo è trovare tutti i segmenti di linea retta nell'immagine ROI.
    
    lines = cv2.HoughLinesP(masked_image, rho=2, theta=np.pi/180, threshold=100, minLineLength=40, maxLineGap=5)
    # (Dovrai regolare 'threshold', 'minLineLength' e 'maxLineGap' per trovare le linee giuste)


    # === PASSO 4: LOGICA DI MEDIA E FILTRAGGIO ===
    # L'obiettivo è ridurre le decine di segmenti trovati a solo DUE linee: sinistra e destra.
    
    # 1. Crea due liste: 'segmenti_sinistri' e 'segmenti_destri'
    left_segment = []
    right_Segment = []
    
    # 2. Itera su 'lines' (prima controlla che 'lines' non sia None!)
    #    - Calcola la pendenza (slope) di ogni segmento: (y2 - y1) / (x2 - x1)
    #    - Se pendenza < -0.3 (circa) -> aggiungi coordinate/pendenza a 'segmenti_sinistri'
    #    - Se pendenza > 0.3 (circa) -> aggiungi coordinate/pendenza a 'segmenti_destri'
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line[0]

            if (x2-x1)==0:
                continue
 
        slope = (y2-y1)/(x2-x1)



        if slope>0.3:
            left_segment.append(line)
        elif slope<-0.3:
            right_Segment.append(line)

    print(f"segmenti Sinistri: {len(left_segment)}, Segmenti Destri: {len(right_Segment)}")
    
    # 3. Calcola la media delle linee (es. con np.polyfit o altri metodi) per trovare
    #    i parametri (m, q) di UNA linea sinistra e UNA linea destra.
    left_line_params = get_line_parameters(left_segment)
    right_line_params = get_line_parameters(right_line_params)


    # === PASSO 5: CALCOLO ERRORE DI STERZATA ===
    # L'obiettivo è calcolare un numero che ci dica "quanto sterzare".
    
    # 1. Calcola il 'centro_corsia'
    #    (media tra la posizione x della linea sx e dx, calcolate a una y fissa, es. in fondo allo schermo)
    #    centro_corsia = (x_sinistro + x_destro) / 2
    
    # 2. Calcola il 'centro_auto' (è semplicemente la metà della larghezza del frame)
    #    centro_auto = width / 2
    
    # 3. Calcola l'errore in pixel
    #    errore_pixel = centro_corsia - centro_auto
    
    # 4. (Opzionale) Calcola l'angolo di sterzata (Controllo P)
    #    Kp = 0.1 # Guadagno proporzionale (da regolare)
    #    angolo_sterzata = Kp * errore_pixel


    # === PASSO 6: VISUALIZZAZIONE ===
    # L'obiettivo è mostrare il nostro risultato sovrapposto al video originale.
    
    # 1. Crea un'immagine "overlay" vuota (np.zeros_like(frame))
    #    overlay_image = np.zeros_like(frame)
    
    # 2. Disegna le due linee (sx e dx) che hai calcolato sull'overlay (cv2.line)
    
    # 3. Disegna il centro corsia (es. linea blu) e il centro auto (es. linea rossa) (cv2.line)
    
    # 4. Scrivi il valore di 'angolo_sterzata' sull'immagine (cv2.putText)
    
    # 5. Combina l'immagine 'frame' originale con l' 'overlay' (cv2.addWeighted)
    #    risultato_finale = cv2.addWeighted(frame, 0.8, overlay_image, 1.0, 0)
    
    # 6. Mostra il 'risultato_finale'
    #    cv2.imshow('Risultato', risultato_finale)
    
    # Per ora, mostriamo solo il frame originale (CANCELLA/COMMENTA QUESTA RIGA QUANDO MOSTRI IL RISULTATO)
    cv2.imshow('Video Originale', masked_image) 


    # --- FINE LOGICA DI ELABORAZIONE ---

    # Premi 'q' per uscire (attende 1 millisecondo)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- PASSO FINALE: RILASCIO E CHIUSURA ---
cap.release()
cv2.destroyAllWindows()