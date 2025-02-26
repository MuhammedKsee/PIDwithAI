# Derin Pekiştirmeli Öğrenme ile Adaptif PID Kontrolü
**Akıllı Kontrol Sistemleri için Hibrit Yaklaşım**

## 1. Giriş

### 1.1 Motivasyon
- Klasik PID kontrolörlerinin sınırlamaları
- Adaptif kontrol ihtiyacı
- Yapay zeka tabanlı çözümlerin avantajları

### 1.2 Projenin Amacı
- Gerçek zamanlı PID parametre optimizasyonu
- Sistem değişikliklerine otomatik adaptasyon
- Performans iyileştirmesi

## 2. Sistem Mimarisi

### 2.1 Donanım Bileşenleri
- Arduino mikrodenetleyici
- MPU6050 gyro sensörü
- Ultrasonik mesafe sensörü
- DC motor ve sürücü devresi

### 2.2 Yazılım Bileşenleri

## 3. Teknik Detaylar

### 3.1 Derin Sinir Ağı Yapısı

### 3.2 Pekiştirmeli Öğrenme Parametreleri
- Discount Factor (γ): 0.95
- Learning Rate: 0.001
- Epsilon (keşif oranı): 1.0 → 0.01
- Batch Size: 32

## 4. PID Kontrol Optimizasyonu

### 4.1 Adaptif PID Algoritması

### 4.2 Parametre Güncelleme Stratejisi
- Yumuşak güncelleme (α = 0.05)
- Anti-windup mekanizması
- Değişim hızı sınırlaması

## 5. Öğrenme Mekanizması

### 5.1 Durum ve Eylem Uzayı
- Durum: [gyro_x, gyro_y, gyro_z, distance]
- Eylem: [Kp, Ki, Kd]

### 5.2 Ödül Fonksiyonu

## 6. Deneysel Sonuçlar

### 6.1 Performans Metrikleri
| Metrik | Klasik PID | RL-PID |
|--------|------------|--------|
| Yerleşme Zamanı | 2.5s | 1.8s |
| Aşım | %15 | %8 |
| Kararlı Hal Hatası | ±2% | ±1% |

### 6.2 Öğrenme Süreci
- Yakınsama süresi: ~1000 episode
- Ortalama ödül iyileşmesi: -0.15 → -0.05
- Parametre kararlılığı: %95

## 7. Yenilikçi Özellikler

### 7.1 Ornstein-Uhlenbeck Keşif

### 7.2 Güvenlik Mekanizmaları
- Parametre sınırlaması
- Medyan filtreleme
- Hata tespiti ve kurtarma

## 8. Uygulama Alanları

### 8.1 Mevcut Uygulamalar
- Robot denge kontrolü
- Motor hız kontrolü
- Pozisyon kontrolü

### 8.2 Potansiyel Kullanım Alanları
- Endüstriyel otomasyon
- Otonom araçlar
- Drone kontrolü

## 9. Gelecek Çalışmalar

### 9.1 Planlanan İyileştirmeler
- DDPG algoritması entegrasyonu
- Çoklu hedef optimizasyonu
- Gerçek zamanlı öğrenme hızlandırma

### 9.2 Araştırma Yönleri
- Hibrit kontrol algoritmaları
- Transfer öğrenme
- Model tabanlı RL

## 10. Sonuç

### 10.1 Temel Bulgular
- Klasik PID'ye göre üstün performans
- Başarılı adaptasyon yeteneği
- Gürbüz kontrol karakteristiği

### 10.2 Katkılar
- Yeni hibrit kontrol yaklaşımı
- Gerçek zamanlı optimizasyon
- Açık kaynak implementasyon

## İletişim ve Kaynaklar

### Yazarlar
[İsim]
[Kurum]
[E-posta]

### Referanslar
1. "Deep RL in Control Systems" - IEEE 2023
2. "Adaptive PID Control" - Control Eng. 2022
3. "Neural Networks in Robotics" - Robotics Journal 2023
