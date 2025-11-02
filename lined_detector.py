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
    right_line_params = get_line_parameters(right_Segment)


    # === PASSO 5: CALCOLO ERRORE DI STERZATA ===
    # L'obiettivo è calcolare un numero che ci dica "quanto sterzare".
    
    # 1. Calcola il 'centro_corsia'
    #    (media tra la posizione x della linea sx e dx, calcolate a una y fissa, es. in fondo allo schermo)
    #    centro_corsia = (x_sinistro + x_destro) / 2
    # === PASSO 5: CALCOLO ERRORE DI STERZATA ===
    
    # Inizializziamo i valori. Partiamo dal presupposto che il centro sia
    # il centro dell'auto, così se non troviamo le linee, l'errore è 0.
    centro_auto = width // 2 # Usiamo // per avere un intero
    centro_corsia = centro_auto 
    
    # Controlliamo di aver trovato ENTRAMBE le linee.
    # Se ne manca una, non possiamo calcolare il centro e usiamo i valori di default.
    if left_line_params is not None and right_line_params is not None:
        
        # Scegliamo una 'y' fissa dove calcolare le coordinate 'x'.
        # La parte più in basso dello schermo è un buon punto.
        y_riferimento = height 
        
        # Estraiamo i parametri m e b dalla linea SINISTRA
        m_sinistro, b_sinistro = left_line_params
        
        # Estraiamo i parametri m e b dalla linea DESTRA
        m_destro, b_destro = right_line_params
        
        # Calcoliamo le coordinate x usando la nostra equazione x = m*y + b
        # Convertiamo in interi perché i pixel non possono essere decimali
        x_sinistro = int((m_sinistro * y_riferimento) + b_sinistro)
        x_destro = int((m_destro * y_riferimento) + b_destro)
        
        # Ora possiamo calcolare il centro della corsia
        centro_corsia = (x_sinistro + x_destro) // 2
        
        # (Debug: puoi disegnare questi punti per vedere se sono corretti)
        #cv2.circle(frame, (x_sinistro, y_riferimento), 10, (0, 255, 0), -1) # Verde
        #cv2.circle(frame, (x_destro, y_riferimento), 10, (0, 0, 255), -1)   # Rosso
        #cv2.circle(frame, (centro_corsia, y_riferimento), 10, (255, 0, 0), -1) # Blu

    # Ora calcoliamo l'errore
    errore_pixel = centro_corsia - centro_auto
    
    # (Opzionale) Calcola l'angolo di sterzata (Controllo P)
    Kp = 0.1 # Guadagno proporzionale (da regolare)
    angolo_sterzata = Kp * errore_pixel

    # (Debug: stampa l'errore)
    print(f"Errore Pixel: {errore_pixel}, Angolo: {angolo_sterzata:.2f}")
    

   # === PASSO 6: VISUALIZZAZIONE ===
    
    # 1. Crea un'immagine "overlay" vuota (nera)
    overlay_image = np.zeros_like(frame)
    
    # 2. Disegna le due linee (sx e dx) sull'overlay
    
    # Definiamo la Y superiore del nostro disegno (deve corrispondere alla Y del tuo ROI)
    y_top_roi = int(height * 0.4) 
    y_bottom = height

    # Disegna la linea SINISTRA (se è stata trovata)
    if left_line_params is not None:
        m_left, b_left = left_line_params
        # Calcoliamo i punti (x1, y1) e (x2, y2)
        x1_left = int((m_left * y_bottom) + b_left)
        x2_left = int((m_left * y_top_roi) + b_left)
        # Disegniamo una linea verde spessa
        cv2.line(overlay_image, (x1_left, y_bottom), (x2_left, y_top_roi), (0, 255, 0), 10)

    # Disegna la linea DESTRA (se è stata trovata)
    if right_line_params is not None:
        m_right, b_right = right_line_params
        # Calcoliamo i punti (x1, y1) e (x2, y2)
        x1_right = int((m_right * y_bottom) + b_right)
        x2_right = int((m_right * y_top_roi) + b_right)
        # Disegniamo una linea verde spessa
        cv2.line(overlay_image, (x1_right, y_bottom), (x2_right, y_top_roi), (0, 255, 0), 10)

    # 3. Disegna il centro corsia (blu) e il centro auto (rosso)
    #    (Disegniamo solo nella parte visibile del ROI)
    
    # 3. Disegna il centro corsia (blu) e il centro auto (rosso)
    
    # Centro Auto (rosso) - è sempre fisso
    cv2.line(overlay_image, (centro_auto, y_bottom), (centro_auto, y_top_roi), (0, 0, 255), 3) # ROSSO
    
    # Centro Corsia (blu) - CONDIZIONALE
    # Disegna la linea blu SOLO se abbiamo trovato una corsia (cioè se è diversa dal centro auto)
    if centro_corsia != centro_auto:
        cv2.line(overlay_image, (centro_corsia, y_bottom), (centro_corsia, y_top_roi), (255, 0, 0), 3) # BLU
    
    # 5. Combina l'immagine 'frame' originale con l' 'overlay'
    #    Questo crea l'effetto "trasparenza"
    risultato_finale = cv2.addWeighted(frame, 0.8, overlay_image, 1.0, 0)
    
    
    # 4. Scrivi il valore di 'angolo_sterzata' sull'immagine FINALE
    #    (Lo facciamo *dopo* addWeighted così non è trasparente)
    testo_angolo = f"Angolo Sterzata: {angolo_sterzata:.2f}"
    cv2.putText(risultato_finale, testo_angolo, 
                (50, 50), # Posizione (x, y) dall'angolo in alto a sinistra
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, # Dimensione font
                (255, 255, 255), # Colore (bianco)
                2, # Spessore
                cv2.LINE_AA)
    
    # 6. Mostra il 'risultato_finale'
    cv2.imshow('Risultato', risultato_finale)
    
    # --- FINE LOGICA DI ELABORAZIONE ---

    # !!! IMPORTANTE: CANCELLA O COMMENTA la vecchia riga 'cv2.imshow'!!!
    # cv2.imshow('Video Originale', frame)
    
    # Per ora, mostriamo solo il frame originale (CANCELLA/COMMENTA QUESTA RIGA QUANDO MOSTRI IL RISULTATO)
    cv2.imshow('Video Originale', masked_image) 


    # --- FINE LOGICA DI ELABORAZIONE ---

    # Premi 'q' per uscire (attende 1 millisecondo)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- PASSO FINALE: RILASCIO E CHIUSURA ---
cap.release()
cv2.destroyAllWindows()