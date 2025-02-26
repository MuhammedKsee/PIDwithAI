# Robotik Sistemlerde Adaptif PID Kontrolü için Pekiştirmeli Öğrenme Yaklaşımı

## İçerik
1. Giriş ve Motivasyon
2. Teorik Altyapı
   - Pekiştirmeli Öğrenme
   - PID Kontrolörleri
   - Adaptif Kontrol Sistemleri
3. Sistem Mimarisi
4. Metodoloji
   - Durum ve Eylem Uzayı
   - Ödül Fonksiyonu
   - Sinir Ağı Yapısı
5. Deneysel Sonuçlar
6. Tartışma ve Gelecek Çalışmalar
7. Sonuç

---

## 1. Giriş ve Motivasyon

- **Problem**: Dinamik ve değişken ortamlarda sabit PID parametreleri optimal performans sağlayamaz
- **Geleneksel Yaklaşım**: Manuel ayarlama veya otomatik ayarlama metodları (Ziegler-Nichols vb.)
- **Önerilen Çözüm**: Pekiştirmeli öğrenme ile kendini adapte eden PID kontrolör
- **Avantajlar**: 
  - Çevrimiçi ve sürekli parametre optimizasyonu
  - Değişken ortam şartlarına uyum sağlama
  - Manuel müdahale gerektirmeme

---

## 2. Teorik Altyapı

### Pekiştirmeli Öğrenme (RL)

- **Temel Prensipler**: Ajan, ortam, durum, eylem, ödül
- **Derin Q-Öğrenme**: Değer fonksiyonunu yaklaşıklamak için derin sinir ağları kullanma
- **Keşif ve Sömürü Dengesi**: Epsilon-greedy stratejisi ve Ornstein-Uhlenbeck gürültü süreci

### PID Kontrolörleri
- **Temel Bileşenler**:
  - P (Oransal): Anlık hata
  - I (İntegral): Birikmiş hata
  - D (Türevsel): Hatanın değişim hızı
- **Temel Zorluklar**: Optimum parametre seçimi, değişken çevre koşullarına adaptasyon

### Adaptif Kontrol Sistemleri
- **Adaptif PID**: Sistem yanıtına göre parametrelerin dinamik olarak ayarlanması
- **RL ile Adaptif Kontrol**: Çevrimiçi öğrenme ve sürekli optimizasyon

---

## 3. Sistem Mimarisi

### Yazılım Bileşenleri:
- **RLAgent**: Pekiştirmeli öğrenme algoritması (DQN tabanlı)
- **AdaptivePIDController**: Ayarlanabilir PID parametreleri
- **RobotController**: Ana kontrol sınıfı ve Arduino iletişimi
- **OUNoise**: Sürekli eylem uzayında keşif için gürültü mekanizması

### Donanım Entegrasyonu:
- Arduino mikrodenetleyici
- İvmeölçer/jiroskop sensörleri
- Mesafe sensörü
- Motor sürücüler

---

## 4. Metodoloji

### Durum ve Eylem Uzayı
- **Durum Uzayı (4 boyutlu)**:
  - Jiroskop verileri (3 eksen)
  - Mesafe sensörü
- **Eylem Uzayı (3 boyutlu)**:
  - Kp (Oransal kazanç)
  - Ki (İntegral kazanç) 
  - Kd (Türevsel kazanç)

### Ödül Fonksiyonu
```
Ödül = -(normalize_edilmiş_hata) 
       -(normalize_edilmiş_integral × stabilite_ağırlığı)
       -(normalize_edilmiş_değişim × değişim_ağırlığı)
```
- **Başarı Bonusu**: Hata çok düşük olduğunda ek ödül
- **Normalizasyon**: Ödüllerin [-1, 1] aralığına ölçeklenmesi

### Sinir Ağı Yapısı
- **Mimari**: 4 (giriş) → 24 → 24 → 3 (çıkış)
- **Aktivasyon Fonksiyonları**: ReLU (gizli katmanlar), Sigmoid (çıkış katmanı)
- **Optimizasyon**: Adam optimizer
- **Yumuşak Hedef Ağ Güncellemeleri**: Kararlı öğrenme için

---

## 5. Deneysel Sonuçlar

(Not: Bu bölümde gerçek deney sonuçları eklenmelidir)

- **Kararlılık Performansı**: Geleneksel vs. Adaptif PID
- **Öğrenme Eğrileri**: Ödül ve hata değerlerindeki iyileşme
- **Parametre Adaptasyonu**: PID parametrelerinin zaman içindeki değişimi
- **Bozucu Etkilere Karşı Dayanıklılık**: Ani sistem değişikliklerine yanıt

---

## 6. Tartışma ve Gelecek Çalışmalar

### Güçlü Yönler
- Sürekli ve otomatik PID parametre ayarlaması
- Değişken koşullara adaptasyon yeteneği
- Anti-windup ve yumuşak parametre geçişleri ile kararlılık garantisi

### Sınırlamalar
- Öğrenme süreci başlangıçta kararsızlığa yol açabilir
- Gerçek zamanlı hesaplama gereksinimleri
- Hiper-parametre ayarlamasına duyarlılık

### Gelecek Çalışmalar
- Transfer öğrenme yaklaşımları
- Farklı RL algoritmalarının karşılaştırılması (DDPG, TD3, SAC vb.)
- Daha karmaşık robot dinamikleri için genişletme
- Model tabanlı RL yaklaşımlarının entegrasyonu

---

## 7. Sonuç

- Pekiştirmeli öğrenme temelli adaptif PID kontrolör, geleneksel yöntemlere göre üstün performans sağlayabilir
- Sistem, öğrenme yeteneği sayesinde değişen koşullara uyum sağlayabilir
- Yaklaşım, farklı robotik sistemlere ve kontrol problemlerine genelleştirilebilir
- Kod tabanı açık kaynaklı olarak paylaşılarak topluluk katkılarına açıktır

---

## Teşekkür

(İlgili kişi, kurum ve destekleyicilere teşekkürler)

---

## İletişim

(İletişim bilgileri ve kaynaklar)
