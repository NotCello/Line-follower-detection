import cv2
import lane_utils  # Importa il tuo nuovo file di utilità

# --- CONFIGURAZIONE ---
# Per usare la webcam, cambia 'video_path' in 0
# Per usare il video, metti il percorso del file
video_path = 'TestVideo/test_lane_detector.mp4'
# video_path = 0 # Esempio per webcam

# --- PASSO 0: SETUP E CARICAMENTO ---
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Errore: Impossibile aprire la fonte video: {video_path}")
    exit()

print("Avvio elaborazione... Premi 'q' per uscire.")

while cap.isOpened():
    # 'ret' è un booleano (True se il frame è stato letto correttamente)
    # 'frame' è l'immagine
    ret, frame = cap.read()

    if not ret:
        if video_path != 0:
            print("Fine del video.")
            break
        else:
            print("Errore lettura webcam, esco.")
            break

    # --- ELABORAZIONE ---
    # Tutta la complessità è nascosta in questa singola funzione!
    try:
        # Passiamo il frame alla nostra utility e riceviamo i risultati
        img_risultato, angolo, errore = lane_utils.process_frame(frame)
        
        # Stampa i dati sul terminale
        print(f"Errore Pixel: {errore}, Angolo: {angolo:.2f}")

        # Mostra solo l'immagine finale
        cv2.imshow('Risultato', img_risultato)

    except Exception as e:
        # Gestisce eventuali errori nell'elaborazione (es. frame corrotti)
        print(f"Errore durante l'elaborazione del frame: {e}")
        # Mostra il frame originale in caso di errore per non crashare
        cv2.imshow('Risultato', frame) 

    # --- CONTROLLO USCITA ---
    # Premi 'q' per uscire (attende 1 millisecondo)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- PASSO FINALE: RILASCIO E CHIUSURA ---
print("Chiusura...")
cap.release()
cv2.destroyAllWindows()