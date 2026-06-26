import cv2
import time
import os
import math
import csv
import threading # Sesin görüntüyü dondurmaması için
import winsound  # Windows alarm sesi için
from datetime import datetime
from ultralytics import YOLO

# 1. DELİL KLASÖRÜ VE LOG DOSYASI
if not os.path.exists("supheliler"):
    os.makedirs("supheliler")

csv_dosyasi = "log_kayitlari.csv"
if not os.path.exists(csv_dosyasi):
    with open(csv_dosyasi, mode='w', newline='', encoding='utf-8') as dosya:
        yazici = csv.writer(dosya)
        yazici.writerow(["ID", "Tarih_Saat", "Olay_Turu", "Dosya_Adi"]) 

print("Yapay zeka HİBRİT modelleri yükleniyor...")
pose_model = YOLO("yolov8n-pose.pt") 
# MÜHENDİSLİK ÇÖZÜMÜ: Renkli şişe kullandığımız için en hızlı modele geri döndük
nesne_model = YOLO("yolov8n.pt") 

HEDEF_URUN_IDLERI = [39, 67] 

kamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not kamera.isOpened():
    print("!!! KRİTİK HATA: Kamera donanımına ulaşılamadı! !!!")
    exit()

pencere_adi = "Market Guvenlik PRO - Sesli Alarm & Global Takip"
cv2.namedWindow(pencere_adi, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(pencere_adi, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

SOL_BILEK, SAG_BILEK = 9, 10
SOL_KALCA, SAG_KALCA = 11, 12

kisi_hafizasi = {}
prev_frame_time = 0

# FPS Optimizasyonu İçin Gerekli Değişkenler
kare_sayaci = 0
son_bilinen_urunler = []

print("Sistem Hazır! \n- Arkaya saklama engeli aktif (2sn kuralı).\n- Sesli Alarm (Beep) Aktif.\n- Çıkış için 'Q', Sıfırlama için 'R'.")

while True:
    basarili_mi, kare = kamera.read()
    if not basarili_mi: break
    
    kare_sayaci += 1
    h, w, c = kare.shape
    reyon_kutu = (int(w * 0.70), int(h * 0.10), int(w * 0.95), int(h * 0.40))
    kasa_kutu = (int(w * 0.05), int(h * 0.10), int(w * 0.30), int(h * 0.40))
    
    pose_sonuclar = pose_model.track(kare, conf=0.5, persist=True, verbose=False)
    icin_cizilmis_kare = pose_sonuclar[0].plot(img=kare) 

    # --- MÜHENDİSLİK ÇÖZÜMÜ: KARE ATLAMA (FRAME SKIPPING) ---
    # Her karede değil, sadece 3 karede bir nesne tanıma yapacağız. FPS uçacak!
    nesne_arama_gerekli = False
    for durum in kisi_hafizasi.values():
        if durum.get("elinde_urun_var_mi", False):
            nesne_arama_gerekli = True
            break
            
    if not nesne_arama_gerekli and pose_sonuclar[0].keypoints is not None:
        for iskelet in pose_sonuclar[0].keypoints.data:
            for el_id in [SOL_BILEK, SAG_BILEK]:
                if iskelet[el_id][2] > 0.5:
                    ex, ey = int(iskelet[el_id][0].item()), int(iskelet[el_id][1].item())
                    if (reyon_kutu[0] < ex < reyon_kutu[2]) and (reyon_kutu[1] < ey < reyon_kutu[3]):
                        nesne_arama_gerekli = True

    urun_merkezleri = []
    if nesne_arama_gerekli:
        if kare_sayaci % 3 == 0: # 3 Karede Bir Yapay Zeka Çalışsın
            nesne_sonuclar = nesne_model.predict(kare, classes=HEDEF_URUN_IDLERI, conf=0.30, verbose=False)
            gecici_urun_listesi = []
            for kutu in nesne_sonuclar[0].boxes:
                x1, y1, x2, y2 = map(int, kutu.xyxy[0])
                urun_adi = "Sise" if int(kutu.cls[0]) == 39 else "Telefon"
                gecici_urun_listesi.append(((x1 + x2) // 2, (y1 + y2) // 2, urun_adi, x1, y1, x2, y2))
            son_bilinen_urunler = gecici_urun_listesi
        
        # Son bilinenleri ekrana çiz (Böylece kutular hiç titremeyecek)
        for ux, uy, u_adi, x1, y1, x2, y2 in son_bilinen_urunler:
            urun_merkezleri.append((ux, uy, u_adi))
            cv2.rectangle(icin_cizilmis_kare, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(icin_cizilmis_kare, u_adi, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
    else:
        son_bilinen_urunler = []

    cv2.rectangle(icin_cizilmis_kare, (reyon_kutu[0], reyon_kutu[1]), (reyon_kutu[2], reyon_kutu[3]), (255, 255, 0), 2)
    cv2.putText(icin_cizilmis_kare, "REYON", (reyon_kutu[0]+5, reyon_kutu[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.rectangle(icin_cizilmis_kare, (kasa_kutu[0], kasa_kutu[1]), (kasa_kutu[2], kasa_kutu[3]), (255, 0, 255), 2)
    cv2.putText(icin_cizilmis_kare, "KASA", (kasa_kutu[0]+5, kasa_kutu[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    if pose_sonuclar[0].boxes.id is not None:
        idler = pose_sonuclar[0].boxes.id.int().cpu().tolist()
        iskeletler = pose_sonuclar[0].keypoints.data 
        zaman_damgasi = datetime.now().strftime("%d-%m-%Y_%H-%M-%S") 

        for i, kisi_id in enumerate(idler):
            iskelet = iskeletler[i] 
            kutu_x1, kutu_y1, kutu_x2, kutu_y2 = pose_sonuclar[0].boxes.xyxy[i].cpu().numpy()
            kisi_orta_x = (kutu_x1 + kutu_x2) / 2
            kisi_boyu = kutu_y2 - kutu_y1
            
            # Elin ürüne ne kadar yakın kabul edileceği (Tolerans biraz artırıldı)
            el_urun_esigi = kisi_boyu * 0.25 

            if kisi_id not in kisi_hafizasi:
                kisi_hafizasi[kisi_id] = {
                    "durum": "NORMAL MUSTERI", 
                    "elinde_urun_var_mi": False, 
                    "alinan_urun": "",
                    "supheli_kaydedildi": False,
                    "kaybolma_ani": 0,
                    "renk": (0, 255, 0)
                }

            durum_bilgisi = kisi_hafizasi[kisi_id]

            if (kasa_kutu[0] < kisi_orta_x < kasa_kutu[2]):
                if "HIRSIZLIK" in durum_bilgisi["durum"]:
                    print(f"[BİLGİ] ID {kisi_id} Kasada Aklanildi!")
                durum_bilgisi.update({"durum": "NORMAL MUSTERI (Kasada)", "elinde_urun_var_mi": False, "supheli_kaydedildi": False, "kaybolma_ani": 0, "renk": (255, 0, 255)})
                continue

            sol_el = iskelet[SOL_BILEK] if iskelet[SOL_BILEK][2] > 0.5 else None
            sag_el = iskelet[SAG_BILEK] if iskelet[SAG_BILEK][2] > 0.5 else None
            gorunur_eller = [el for el in [sol_el, sag_el] if el is not None]

            el_reyonda_mi = False
            for el in gorunur_eller:
                ex, ey = int(el[0].item()), int(el[1].item())
                if (reyon_kutu[0] < ex < reyon_kutu[2]) and (reyon_kutu[1] < ey < reyon_kutu[3]):
                    el_reyonda_mi = True
                    break

            el_yaninda_urun_var_mi = False
            tespit_edilen_urun = ""
            for el in gorunur_eller:
                for ux, uy, u_adi in urun_merkezleri:
                    if math.hypot(el[0].item() - ux, el[1].item() - uy) < el_urun_esigi:
                        el_yaninda_urun_var_mi = True
                        tespit_edilen_urun = u_adi
                        break

            # ==========================================
            # GLOBAL DİSAPPEARANCE (ARKAYA SAKLAMA) MANTIĞI
            # ==========================================
            if not durum_bilgisi["elinde_urun_var_mi"]:
                if el_reyonda_mi:
                    durum_bilgisi.update({"durum": "REYONA BAKIYOR (El Bos)", "renk": (0, 200, 255)})
                    if el_yaninda_urun_var_mi:
                        durum_bilgisi.update({"elinde_urun_var_mi": True, "alinan_urun": tespit_edilen_urun, "durum": f"URUN ALDI ({tespit_edilen_urun})", "renk": (0, 165, 255)})
                else:
                    durum_bilgisi.update({"durum": "NORMAL MUSTERI", "renk": (0, 255, 0)})
            else:
                if el_reyonda_mi:
                    if not el_yaninda_urun_var_mi:
                        durum_bilgisi.update({"elinde_urun_var_mi": False, "durum": "URUNU GERI BIRAKTI", "renk": (0, 255, 0), "supheli_kaydedildi": False, "kaybolma_ani": 0})
                else:
                    # Cep temasını sildik! Artık sadece ürünün tamamen kaybolup kaybolmadığına bakıyoruz.
                    if el_yaninda_urun_var_mi:
                        durum_bilgisi["kaybolma_ani"] = 0 # Şişe hala elde, sayacı sıfırla
                        if durum_bilgisi["supheli_kaydedildi"]:
                            durum_bilgisi.update({"durum": "PISMAN OLDU (Geri Cikardi)", "renk": (255, 100, 0), "supheli_kaydedildi": False})
                        else:
                            durum_bilgisi.update({"durum": f"URUN TASIYOR ({durum_bilgisi['alinan_urun']})", "renk": (255, 200, 0)})
                    else:
                        # ŞİŞE 2 SANİYE BOYUNCA HİÇBİR YERDE GÖRÜNMÜYORSA (Cepte, Arkada, Montta)
                        if not durum_bilgisi["supheli_kaydedildi"]:
                            if durum_bilgisi["kaybolma_ani"] == 0:
                                durum_bilgisi["kaybolma_ani"] = time.time() 
                                durum_bilgisi.update({"durum": "SUPHELI HAREKET (Urun Kayboldu)", "renk": (0, 100, 255)})
                            elif time.time() - durum_bilgisi["kaybolma_ani"] > 2.0:
                                # 2 saniye geçti, ürün yok, Alarmı Bas!
                                durum_bilgisi.update({"durum": f"!!! HIRSIZLIK ({durum_bilgisi['alinan_urun']}) !!!", "renk": (0, 0, 255), "supheli_kaydedildi": True})
                                
                                # ASENKRON SESLİ ALARM (Görüntüyü dondurmaz)
                                threading.Thread(target=winsound.Beep, args=(2500, 1000)).start()
                                
                                dosya_adi = f"supheliler/Supheli_ID{kisi_id}_{zaman_damgasi}.jpg"
                                cv2.imwrite(dosya_adi, icin_cizilmis_kare)
                                with open(csv_dosyasi, mode='a', newline='', encoding='utf-8') as dosya:
                                    csv.writer(dosya).writerow([kisi_id, zaman_damgasi, "GIZLEME / CEBE ATMA", dosya_adi])
                                print(f"!!! ALARM: Ürün saklandı/çalındı!")

            cv2.putText(icin_cizilmis_kare, durum_bilgisi["durum"], (int(kutu_x1), int(kutu_y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, durum_bilgisi["renk"], 2)

    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(icin_cizilmis_kare, f"FPS: {int(fps)}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow(pencere_adi, icin_cizilmis_kare)

    tus = cv2.waitKey(1) & 0xFF
    if tus == ord('q') or tus == ord('Q'): break
    elif tus == ord('r') or tus == ord('R'): kisi_hafizasi.clear()

kamera.release()
cv2.destroyAllWindows()
print("Sistem kapatıldı.")