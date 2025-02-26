# Robotik Sistemlerde Adaptif PID Kontrolü için Pekiştirmeli Öğrenme Yaklaşımı

## İçindekiler
1. Giriş ve Motivasyon
2. Literatür İncelemesi
3. Teorik Altyapı
   - Pekiştirmeli Öğrenme Teorisi
   - PID Kontrol Sistemleri
   - Adaptif Kontrol Yaklaşımları
4. Önerilen Sistem Mimarisi
   - Genel Mimari
   - Yazılım Bileşenleri
   - Donanım Entegrasyonu
5. Metodoloji
   - Problem Formülasyonu
   - Durum ve Eylem Uzayı Tasarımı
   - Ödül Fonksiyonu Mühendisliği
   - Sinir Ağı Mimarisi ve Eğitimi
   - Keşif Stratejileri
6. Uygulama Detayları
   - Veri Ön İşleme
   - Deneyim Tekrarı (Experience Replay)
   - Hedef Ağ Mekanizması
   - Yumuşak Parametre Güncellemeleri
7. Deneysel Analiz
   - Test Düzeneği
   - Performans Metrikleri
   - Karşılaştırmalı Analizler
   - Dayanıklılık Testleri
8. Sonuçlar ve Tartışma
9. Gelecek Çalışma Yönelimleri
10. Sonuç
11. Kaynakça
12. Ekler

---

## 1. Giriş ve Motivasyon

### 1.1 Problem Tanımı
PID (Proportional-Integral-Derivative) kontrolörleri, endüstride ve robotik sistemlerde yaygın olarak kullanılan kontrol mekanizmalarıdır. Ancak klasik PID kontrolörleri, sabit parametrelere dayanır ve değişken çevre koşullarına veya sistem dinamiklerindeki değişimlere karşı optimal performansı sürdüremezler. Bu durum, özellikle dinamik ortamlarda çalışan otonom robotik sistemlerde önemli bir kısıtlama oluşturmaktadır.

### 1.2 Geleneksel Yaklaşımların Kısıtlamaları
- Manuel parametre ayarlamasının zaman alıcı olması
- Ziegler-Nichols gibi klasik otomatik ayarlama yöntemlerinin belirli çalışma noktalarına özgü olması
- Sistem dinamiklerindeki değişimlere hızlı adapte olamaması
- Doğrusal olmayan sistem dinamiklerinde optimal parametre belirleme zorluğu
- Real-time uygulamalarda yeniden kalibrasyon gerektirmesi

### 1.3 Önerilen Yaklaşım
Bu çalışmada, değişken çevre koşullarına ve sistem dinamiklerine gerçek zamanlı olarak adapte olabilen, pekiştirmeli öğrenme tabanlı bir adaptif PID kontrol sistemi öneriyoruz. Sistem, çevrimiçi öğrenme yeteneği ile sensör verilerini kullanarak PID parametrelerini (Kp, Ki, Kd) sürekli ve otomatik olarak optimize etmektedir.

### 1.4 Araştırmanın Katkıları
- Derin pekiştirmeli öğrenme ile adaptif PID kontrolünün entegrasyonu
- Sürekli öğrenme ve adaptasyon yeteneği
- Doğrusal olmayan sistemlerde bile yüksek performans
- Gerçek zamanlı uygulanabilirlik
- Minimum insan müdahalesi gerektirme
- Transfer öğrenme potansiyeli

---

## 2. Literatür İncelemesi

### 2.1 Klasik PID Kontrolü Çalışmaları
- Ziegler-Nichols metodu (1942) ve modern varyasyonları
- Cohen-Coon ve AMIGO gibi otomatik ayarlama yöntemleri
- Parametre uzayı yaklaşımı ve dayanıklı PID tasarımı
- Model öngörülü PID kontrol

### 2.2 Adaptif Kontrol Yaklaşımları
- Model Referans Adaptif Kontrol (MRAC)
- Öz Ayarlamalı Kontrolörler (Self-Tuning Regulators)
- Bulanık mantık tabanlı adaptif kontrol
- Yapay sinir ağları ile PID parametre optimizasyonu

### 2.3 Pekiştirmeli Öğrenme ve Kontrol
- Q-öğrenme ve Derin Q-Ağları (DQN) temelli yaklaşımlar
- Politika Gradyan metodları (REINFORCE, PPO, TRPO)
- Aktör-Kritik mimariler (A2C, A3C, DDPG)
- Soft Actor-Critic (SAC) ve Twin Delayed DDPG (TD3) algoritmaları

### 2.4 Robotik Sistemlerde RL Uygulamaları
- Manipülatör kontrolü için DQN ve DDPG uygulamaları
- Mobil robot navigasyonu için pekiştirmeli öğrenme
- Drone kontrolünde adaptif yaklaşımlar
- İnsansı robotlarda dengesizlik kontrolü

---

## 3. Teorik Altyapı

### 3.1 Pekiştirmeli Öğrenme Teorisi

#### 3.1.1 Temel Kavramlar
- **Ajan (Agent)**: Öğrenen ve karar veren sistem
- **Çevre (Environment)**: Ajanın etkileşimde bulunduğu sistem
- **Durum (State)**: Çevrenin anlık durumu (s ∈ S)
- **Eylem (Action)**: Ajanın seçebileceği eylemler (a ∈ A)
- **Ödül (Reward)**: Eylem sonucunda alınan anlık geri bildirim (r)
- **Politika (Policy)**: Durumlardan eylemlere bir haritalama (π: S → A)
- **Değer Fonksiyonu (Value Function)**: Gelecekteki beklenen toplam ödül

#### 3.1.2 Markov Karar Süreci (MDP)
- Durum geçiş olasılıkları: P(s'|s,a)
- Ödül fonksiyonu: R(s,a,s')
- İndirgeme faktörü (γ): Gelecekteki ödüllerin şimdiki değeri
- Bellman denklemi ve optimallik prensibi

#### 3.1.3 Derin Q-Öğrenme (DQN)
- Q-değer fonksiyonu ve Q-güncelleme denklemi
- Derin sinir ağları ile fonksiyon yaklaşıklama
- Deneyim tekrarı (Experience Replay) mekanizması
- Hedef ağ (Target Network) ve sabit hedef Q-değerleri
- Keşif-sömürü dengesi (Exploration-Exploitation Trade-off)

#### 3.1.4 Sürekli Eylem Uzayında RL
- Politika gradyan yöntemleri
- Aktör-kritik mimariler
- Ornstein-Uhlenbeck gürültü süreci
- DDPG (Deep Deterministic Policy Gradient) algoritması

### 3.2 PID Kontrol Sistemleri

#### 3.2.1 PID Kontrol Teorisi
- **Oransal (P)**: Anlık hataya doğrusal yanıt
  - $u_P(t) = K_p \cdot e(t)$
- **İntegral (I)**: Geçmiş hataların birikimli etkisi
  - $u_I(t) = K_i \int_{0}^{t} e(\tau) d\tau$
- **Türevsel (D)**: Hatanın değişim hızına yanıt
  - $u_D(t) = K_d \frac{de(t)}{dt}$
- **Toplam PID Çıkışı**:
  - $u(t) = K_p \cdot e(t) + K_i \int_{0}^{t} e(\tau) d\tau + K_d \frac{de(t)}{dt}$

#### 3.2.2 PID Performans Ölçütleri
- Yükselme zamanı (Rise Time)
- Aşma yüzdesi (Overshoot)
- Yerleşme zamanı (Settling Time)
- Kararlı durum hatası (Steady-State Error)
- Integral Square Error (ISE) ve Integral Absolute Error (IAE)

#### 3.2.3 PID Kontrol Zorlukları
- Parametre ayarlaması (tuning)
- Integral Windup problemi ve anti-windup teknikleri
- Türev teriminin gürültü hassasiyeti
- Doğrusal olmayan sistemlerde performans düşüşü

### 3.3 Adaptif Kontrol Yaklaşımları

#### 3.3.1 Adaptif Kontrolün Temelleri
- Parametre tanımlama (Parameter Identification)
- Model belirsizlikleri ile başa çıkma
- Direkt ve dolaylı adaptif kontrol
- Kazanç ayarlama mekanizmaları

#### 3.3.2 Adaptif PID Kontrol
- Otomatik ayarlama yöntemleri
- Çevrimiçi parametre optimizasyonu
- Model öngörülü adaptasyon
- Dayanıklılık (robustness) garantileri

---

## 4. Önerilen Sistem Mimarisi

### 4.1 Genel Mimari

Önerilen sistem, pekiştirmeli öğrenme ajanı, adaptif PID kontrolörü ve sensör-aktüatör sistemlerinden oluşan entegre bir mimari sunmaktadır. Sistem, aşağıdaki ana bileşenlerden oluşmaktadır:

```
[Sensörler] → [Veri İşleme] → [RL Ajanı] → [Adaptif PID] → [Aktüatörler] → [Robot]
                   ↑               ↑             ↓               ↓
                   |               |             |               |
                   └---------------+-------------+---------------┘
                               [Geri Bildirim Döngüsü]
```

### 4.2 Yazılım Bileşenleri

#### 4.2.1 OUNoise Sınıfı
```python
class OUNoise:
    """Ornstein-Uhlenbeck gürültü süreci - sürekli eylem uzayında keşif için"""
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.size = size          
        self.mu = mu * np.ones(size)  
        self.theta = theta        
        self.sigma = sigma        
        self.reset()
        
    def reset(self):
        """Gürültü durumunu başlangıç değerine sıfırla"""
        self.state = np.copy(self.mu)
        
    def sample(self):
        """Yeni bir gürültü örneği üret"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state
```

- **Amaç**: Sürekli eylem uzayında keşif mekanizması sağlamak
- **Parametreler**:
  - `mu`: Ortalama değer (0 varsayılan)
  - `theta`: Ortalamaya dönme hızı (0.15 varsayılan)
  - `sigma`: Gürültü şiddeti (0.2 varsayılan)
- **Fonksiyonlar**:
  - `reset()`: Gürültü durumunu başlangıç değerine döndürür
  - `sample()`: Yeni bir gürültü örneği üretir

#### 4.2.2 RLAgent Sınıfı
```python
class RLAgent:
    """Pekiştirmeli öğrenme ajanı - PID parametrelerini optimize eder"""
    def __init__(self, state_size, action_size):
        # Temel parametreler
        self.state_size = state_size    # Durum uzayı boyutu (gyro + mesafe)
        self.action_size = action_size  # Eylem uzayı boyutu (PID parametreleri)
        
        # Deneyim hafızası ve öğrenme parametreleri
        self.memory = deque(maxlen=2000)  # Son 2000 deneyimi sakla
        self.gamma = 0.95        # Gelecek ödüllerin indirim faktörü
        self.epsilon = 1.0       # Keşif oranı
        self.epsilon_min = 0.01  # Minimum keşif oranı
        self.epsilon_decay = 0.995  # Keşif azalma oranı
        self.learning_rate = 0.001  # Öğrenme hızı
        
        # PyTorch modeli ve optimizasyon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)  # Ana model
        self.target_model = self._build_model().to(self.device)  # Hedef ağ
        
        # Eğitim parametreleri
        self.batch_size = 32  # Mini-batch boyutu
        self.update_target_every = 100  # Hedef ağ güncelleme sıklığı
        self.train_counter = 0  # Eğitim adım sayacı
        
        # Eylem parametre sınırları ve güncelleme
        self.action_bounds = (0, 2)  # PID parametreleri için sınırlar
        self.tau = 0.001  # Yumuşak güncelleme katsayısı
        
        # Keşif mekanizması
        self.noise = OUNoise(action_size)  # Ornstein-Uhlenbeck gürültüsü
        self.min_epsilon = 0.1  # Minimum keşif oranı
```

- **Amaç**: Pekiştirmeli öğrenme algoritmasını çalıştırmak ve PID parametrelerini optimize etmek
- **Yapı**:
  - Sinir ağı modeli ve hafıza mekanizması
  - DQN ve DDPG algoritmaları entegrasyonu
  - Deneyim tekrarı ve hedef ağ mekanizmaları
  - Keşif stratejileri (epsilon-greedy ve OU-noise)
- **Önemli Metotlar**:
  - `_build_model()`: Sinir ağı mimarisini oluşturur
  - `remember()`: Deneyimleri hafızaya kaydeder
  - `train()`: Deneyim tekrarı ile ağı eğitir
  - `get_action()`: Mevcut duruma göre eylem seçer
  - `_soft_update()`: Hedef ağı yumuşak günceller

#### 4.2.3 AdaptivePIDController Sınıfı
```python
class AdaptivePIDController:
    """Adaptif PID kontrolcü - parametreleri RL ajanı tarafından ayarlanır"""
    def __init__(self):
        # PID parametreleri
        self.kp = 0.0  # Oransal kazanç
        self.ki = 0.0  # İntegral kazanç
        self.kd = 0.0  # Türevsel kazanç
        
        # Hata terimleri
        self.last_error = 0  # Son hata
        self.integral = 0    # Toplam hata
        self.target = 0      # Hedef değer
        
        # Kontrol parametreleri
        self.max_integral = 100.0  # İntegral sınırı (anti-windup)
        self.alpha = 0.05         # Parametre güncelleme hızı
        self.max_change_rate = 0.1  # Maksimum değişim oranı
        self.last_params = np.zeros(3)  # Son PID parametreleri
```

- **Amaç**: PID kontrol algoritmasını çalıştırmak ve parametreleri adapte etmek
- **Parametreler**:
  - `kp`, `ki`, `kd`: PID kontrol kazançları
  - `max_integral`: Anti-windup için integral limiti
  - `alpha`: Parametre güncelleme hızı
  - `max_change_rate`: Maksimum parametre değişim oranı
- **Metotlar**:
  - `update_parameters()`: PID parametrelerini yumuşak günceller
  - `compute()`: PID kontrol çıkışını hesaplar

#### 4.2.4 RobotController Sınıfı
```python
class RobotController:
    """Ana kontrol sınıfı - Arduino ile haberleşme ve sistem kontrolü"""
    def __init__(self):
        # Arduino bağlantısı
        self.arduino = serial.Serial('COM3', 9600, timeout=1)
        time.sleep(2)  # Arduino'nun başlaması için bekle
        
        # Sistem boyutları
        self.state_size = 4  # Gyro (3 eksen) + Mesafe
        self.action_size = 3  # PID parametreleri (Kp, Ki, Kd)
        
        # Kontrol bileşenleri
        self.rl_agent = RLAgent(self.state_size, self.action_size)
        self.pid_controller = AdaptivePIDController()
        
        # Veri tamponları
        self.current_state = np.zeros(self.state_size)
        self.reward_buffer = deque(maxlen=100)
```

- **Amaç**: Tüm sistem entegrasyonu ve kontrol döngüsü
- **Bileşenler**:
  - Arduino seri haberleşme bağlantısı
  - RL ajanı ve PID kontrolör entegrasyonu
  - Sensör veri işleme ve filtreleme
  - Ödül hesaplama
- **Ana Metotlar**:
  - `read_sensors()`: Sensör verilerini okur ve filtreler
  - `calculate_reward()`: Sistem performansına göre ödül hesaplar
  - `run_episode()`: Ana kontrol döngüsünü çalıştırır

### 4.3 Donanım Entegrasyonu

#### 4.3.1 Arduino Mikrodenetleyici
- ATmega328P tabanlı mikrodenetleyici
- 16 MHz çalışma frekansı
- 32 KB Flash bellek
- Sensör ve aktüatör arayüzü

#### 4.3.2 Sensör Sistemi
- **IMU (Inertial Measurement Unit)**:
  - 3 eksenli jiroskop (açısal hız)
  - 3 eksenli ivmeölçer (doğrusal ivme)
  - I2C haberleşme protokolü
- **Mesafe Sensörü**:
  - Ultrasonik mesafe ölçümü
  - Çalışma aralığı: 2cm - 400cm
- **Enkoder**:
  - Motorların açısal pozisyon ve hız ölçümü
  - Quadrature kodlama

#### 4.3.3 Aktüatör Sistemi
- **DC Motor Sürücüler**:
  - H-köprü devresi
  - PWM (Pulse Width Modulation) kontrol
  - Yönlendirme kontrolü
- **Motor Özellikleri**:
  - 12V DC motorlar
  - Redüktörlü
  - Enkoder geri bildirimi

---

## 5. Metodoloji

### 5.1 Problem Formülasyonu

Bu çalışmada, pekiştirmeli öğrenme ajanının amacı, robotun belirli bir görevde optimum performans göstermesi için en uygun PID parametrelerini bulmaktır. Bu problemi şu şekilde formüle ediyoruz:

- **Amaç**: Beklenen toplam ödülü maksimize eden optimal politikayı bulmak
- **MDP Formülasyonu**:
  - Durum uzayı: S ⊆ ℝ⁴ (jiroskop + mesafe)
  - Eylem uzayı: A ⊆ ℝ³ (Kp, Ki, Kd)
  - Ödül fonksiyonu: R(s, a, s')
  - Geçiş dinamiği: P(s'|s, a)
  - İndirim faktörü: γ ∈ [0, 1]

### 5.2 Durum ve Eylem Uzayı Tasarımı

#### 5.2.1 Durum Uzayı
Durum uzayı, aşağıdaki sensör verilerinden oluşan 4 boyutlu bir vektördür:
- s₁: X ekseni jiroskop verisi
- s₂: Y ekseni jiroskop verisi
- s₃: Z ekseni jiroskop verisi
- s₄: Mesafe sensörü verisi

Bu durum uzayı, robotun dinamik davranışını ve çevresiyle olan etkileşimini temsil etmektedir. Jiroskop verileri, robotun yönelim ve denge durumunu, mesafe sensörü ise robotun engellere olan uzaklığını göstermektedir.

#### 5.2.2 Eylem Uzayı
Eylem uzayı, PID kontrolörünün üç parametresinden oluşan 3 boyutlu sürekli bir uzaydır:
- a₁: Kp (Oransal kazanç)
- a₂: Ki (İntegral kazanç)
- a₃: Kd (Türevsel kazanç)

Bu eylemler [0, 2] aralığında sınırlandırılmıştır, böylece kontrolör parametrelerinin aşırı büyük değerler alması engellenerek sistem kararlılığı sağlanmaktadır.

### 5.3 Ödül Fonksiyonu Mühendisliği

Ödül fonksiyonu, sistemin performansını değerlendiren kritik bir bileşendir. Önerilen ödül fonksiyonu aşağıdaki bileşenlerden oluşmaktadır:

```python
def calculate_reward(self, error):
    # Ana hata terimi
    normalized_error = self.normalize_error(error)
    error_reward = -abs(normalized_error)
    
    # Stabilite terimi
    normalized_integral = self.normalize_error(self.pid_controller.integral)
    stability_reward = -abs(normalized_integral) * self.stability_weight
    
    # Değişim terimi
    error_change = error - self.pid_controller.last_error
    normalized_change = self.normalize_error(error_change)
    change_reward = -abs(normalized_change) * self.change_weight
    
    # Toplam ödül
    total_reward = (error_reward + stability_reward + change_reward) * self.reward_scale
    
    # Başarı bonusu
    if abs(normalized_error) < 0.1:
        total_reward += 0.5
            
    return total_reward
```

- **Hata Terimi**: Sistemin referans değere ne kadar yakın olduğunu ölçer
- **Stabilite Terimi**: Sistem yanıtının kararlılığını değerlendirir
- **Değişim Terimi**: Hatanın değişim hızını kontrol ederek ani değişimleri cezalandırır
- **Başarı Bonusu**: Sistem hedef değere çok yaklaştığında ek ödül verir

### 5.4 Sinir Ağı Mimarisi ve Eğitimi

#### 5.4.1 Sinir Ağı Mimarisi
```python
def _build_model(self):
    """Sinir ağı modelini oluştur"""
    model = nn.Sequential(
        nn.Linear(self.state_size, 24),  # Giriş katmanı
        nn.ReLU(),                       # Aktivasyon fonksiyonu
        nn.Linear(24, 24),               # Gizli katman
        nn.ReLU(),                       # Aktivasyon fonksiyonu
        nn.Linear(24, self.action_size), # Çıkış katmanı
        nn.Sigmoid()                     # Çıktıyı 0-1 arasına sınırla
    )
    self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
    return model
```

- **Katmanlar**:
  - Giriş katmanı: 4 nöron (durum boyutu)
  - İlk gizli katman: 24 nöron, ReLU aktivasyonu
  - İkinci gizli katman: 24 nöron, ReLU aktivasyonu
  - Çıkış katmanı: 3 nöron (eylem boyutu), Sigmoid aktivasyonu

- **Aktivasyon Fonksiyonları**:
  - ReLU: Gizli katmanlarda doğrusal olmayan özellik çıkarımı için
  - Sigmoid: Çıkış değerlerini [0,1] aralığına normalize etmek için

#### 5.4.2 Eğitim Algoritması
```python
def train(self):
    """Deneyim tekrarı ile ağı eğit"""
    if len(self.memory) < self.batch_size:
        return
    
    # Mini-batch oluştur
    batch = random.sample(self.memory, self.batch_size)
    states = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(self.device)
    actions = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32).to(self.device)
    rewards = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32).to(self.device)
    next_states = torch.tensor(np.array([x[3] for x in batch]), dtype=torch.float32).to(self.device)
    
    # Q değerlerini hesapla
    current_q = self.model(states)
    next_actions = self.target_model(next_states).detach()
    next_q = rewards + self.gamma * torch.sum(next_actions, dim=1)
    
    # Kayıp fonksiyonu ve optimizasyon
    loss = nn.MSELoss()(current_q, next_q.unsqueeze(1).repeat(1, self.action_size))
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    # Hedef ağı güncelle
    self._soft_update()
    
    # Keşif oranını azalt
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay
```

- **Deneyim Tekrarı**: Rastgele örneklenen mini-batch'ler ile korelasyonu azaltma
- **Hedef Q-Değerleri**: Hedef ağ ile hesaplanan gelecekteki Q-değerleri
- **Kayıp Fonksiyonu**: Mean Squared Error (MSE)
- **Optimizasyon**: Adam optimizasyon algoritması
- **Hedef Ağ Güncelleme**: Yumuşak (soft) parametreleri güncelleme
- **Keşif Oranı**: Epsilon-greedy stratejisi ile keşif-sömürü dengesi

### 5.5 Keşif Stratejileri

#### 5.5.1 Epsilon-Greedy Yaklaşımı
```python
# Keşif stratejisi
if np.random.random() < self.epsilon:
    noise = self.noise.sample()
    raw_actions = np.clip(raw_actions + noise, 0, 1)
```

- Eğitimin başında yüksek keşif oranı (ε = 1.0)
- Zamanla azalan keşif oranı (ε = ε * 0.995)
- Minimum keşif oranı (ε_min = 0.01)

#### 5.5.2 Ornstein-Uhlenbeck Gürültüsü
Sürekli eylem uzayında keşif için Ornstein-Uhlenbeck (OU) gürültü süreci kullanılmaktadır. OU gürültüsü, zamana bağlı korelasyonu olan bir gürültü sürecidir ve bu özelliği sayesinde fiziksel sistemlerin kontrol edilmesinde daha doğal bir keşif sağlar.

```python
dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
self.state += dx
```

- **Parametreler**:
  - `mu`: Ortalama değer (genellikle 0)
  - `theta`: Ortalamaya dönme hızı (0.15)
  - `sigma`: Gürültü şiddeti (0.2)

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
