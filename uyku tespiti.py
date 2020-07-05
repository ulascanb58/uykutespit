#151805040 - Ulas Can Bozkurt
#Modüllerimizi import edelim.

#Open CV kütüphanesi importudur.
import cv2
#Sinir ağı modelini teslim eder.
from keras.models import load_model
#NumPy, dizilerle çalışmak için kullanılan bir python kütüphanesidir.
import numpy as np
#Pygame alarm sesinin çalması için kullanılan python kütüphanesidir.
from pygame import mixer

#Uyku durumu tespit edildiğinde çalacak olan alarm.
mixer.init()
ses = mixer.Sound('alarm2.wav')

#Görüntü üzerinde nesne bulmak için Haar Cascade Files kullanıldı.(Yüz ve Göz için)
yuz = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
goz = cv2.CascadeClassifier('haar cascade files\haarcascade_eyes.xml')


#Gözün açık mı kapalı mı olduğunu labellerimizde tutuyoruz.
lbl = ['KAPALI', 'ACIK']

#Gözleri model ile tahmin etmek için modelimizi yükledik.
#İçinde 7000 adet insan gözünün farklı ışık koşullarında görüntüsünü içermektedir.
model = load_model('models/cnncat2.h5')

#Kameradan video çekimi
kayit = cv2.VideoCapture(0)
#Ekranda ki yazıların fontu
font = cv2.FONT_HERSHEY_PLAIN

#Puan, 10 değerini geçince sürücüye uyarı vermek için skor değişkeni tanımlanmıştır.
puan = 0

#Gözün açık mı kapalı mı olduğunu belirtmekte kullanacağız.
goztahmin = [99]


while (True):
    #Kamera kaydının yürütülmesi, ekran en boy oranı.
    ret, frame = kayit.read()
    height, width = frame.shape[:2]

    #Gri renk tonlaması tanımı
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Yüzleri ve gözleri gri skala kullanarak algılamak için kullanılır.
    #Görüntüyü 1,3 skala ile kontrol edecek ve 5 kere yüzün orada var mı yok mu olduğunu teyit edecek.
    yuzler = yuz.detectMultiScale(gray,1.3,5)
    gozler = goz.detectMultiScale(gray)


    # Görüntüde bulunan yüzlere dikdörtgen çizmek için for döngüsünü kullanıyoruz.
    # For ile dönmemizin nedeni ise görüntü içerisinde birden fazla insan olabilir.
    for (a, b, c, d) in yuzler:
        # Çerçevenin konumu, rengi ve kalınlığı (yüz bölgesinde)
        cv2.rectangle(frame, (a, b), (a + c, b + d), (255, 0, 0), 2)

    # Görüntüde bulunan gözlere dikdörtgen çizmek için for döngüsünü kullanıyoruz.
    # For ile dönmemizin nedeni ise görüntü içerisinde birden fazla göz olabilir.
    for (a, b, c, d) in gozler:
        #Göz bölgesinde yeşil dikdörtgen ve ilgi alanı oluşturmak.
        cv2.rectangle(frame, (a, b), (a + c, b + d), (0, 255, 0), 2)
        #Yukarıda tespit ettiğimiz tam görüntüden sadece gözleri tespit etmek için kullanırız
        #Gözün sınır kutusunu çıkararak daha sonra da göz kodunu çerçeveden çıkartırız.
        #Sadece gözlerin görüntü verilerini içerir.
        n_goz = frame[b : b + d, a : a + c]
        #Renkli görüntüyü gri tonlamaya dönüştürürüz.
        n_goz = cv2.cvtColor(n_goz, cv2.COLOR_BGR2GRAY)
        #Modelimizi 24x24 piksel piksel resimlere yeniden boyutlandırıyoruz
        n_goz = cv2.resize(n_goz, (24, 24))
        #Daha iyi yakınsamak için verilerimizi normalleştiriyoruz.
        n_goz = n_goz / 255
        #Tüm değerler 24x24 0-1 arasında olacaktır.
        n_goz = n_goz.reshape(24, 24, -1)
        #Sınıflandırıcıyı besleyecek olan boyutları genişletiyoruz
        n_goz = np.expand_dims(n_goz, axis=0)
        #Gözlerin kapalı mı yoksa açık mı olduğunu tahmin etme.
        goztahmin = model.predict_classes(n_goz)
        if (goztahmin[0] == 1):
            #Gözler açık
            lbl = 'ACIK'
        if (goztahmin[0] == 0):
            #Gözler kapalı
            lbl = 'KAPALI'
        break

    #Gözün kapalı olduğu tahmininde bulunma aşaması.
    if (goztahmin[0] == 0):
        puan = puan + 1
        cv2.putText(frame, "KAPALI", (30, height - 20), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    #Gözün açık olduğu tahmininde bulunma aşaması.
    else:
        puan = puan - 1
        cv2.putText(frame, "ACIK", (30, height - 20), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    if (puan < 0):
        puan = 0
    cv2.putText(frame, ' PUAN:' + str(puan), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    #Göz kapalı ise skor artacak skor 10'u geçtiğinde ekranda yazı basacak ve alarm çalacak.
    if (puan > 10):

        try:
            ses.play()
        except:
            pass
        cv2.putText(frame, "### UYARI ! ###", (185, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "### UYARI ! ###", (180, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #Program çalıştığında frame'nin ismi.
    cv2.imshow('SURUCU UYKU TESPIT SISTEMI', frame)
    # E ye basınca çalışma durur
    if cv2.waitKey(1) & 0xFF == ord('e'):

        break
#Kaydı serbest bırakma ve pencereleri kapatma.
kayit.release()
cv2.destroyAllWindows()
