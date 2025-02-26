import serial
import time
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import random

class OUNoise:
    """Ornstein-Uhlenbeck gürültü süreci - sürekli eylem uzayında keşif için"""
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """
        Parametreler:
            size: Gürültü vektörünün boyutu
            mu: Ortalama değer (varsayılan: 0)
            theta: Ortalamaya dönme hızı (varsayılan: 0.15)
            sigma: Gürültü şiddeti (varsayılan: 0.2)
        """
        self.size = size          
        self.mu = mu * np.ones(size)  
        self.theta = theta        
        self.sigma = sigma        
        self.reset()
        
    def reset(self):
        """Gürültü durumunu başlangıç değerine sıfırla"""
        self.state = np.copy(self.mu)
        
    def sample(self):
        """
        Yeni bir gürültü örneği üret
        Returns:
            Ornstein-Uhlenbeck gürültü vektörü
        """
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state

class RLAgent:
    """Pekiştirmeli öğrenme ajanı - PID parametrelerini optimize eder"""
    def __init__(self, state_size, action_size):
        # Temel parametreler
        self.state_size = state_size    # Durum uzayı boyutu (gyro + mesafe)
        self.action_size = action_size  # Eylem uzayı boyutu (PID parametreleri)
        
        # Deneyim hafızası
        self.memory = deque(maxlen=2000)  # Son 2000 deneyimi sakla
        
        # Öğrenme parametreleri
        self.gamma = 0.95        # Gelecek ödüllerin indirim faktörü
        self.epsilon = 1.0       # Keşif oranı
        self.epsilon_min = 0.01  # Minimum keşif oranı
        self.epsilon_decay = 0.995  # Keşif azalma oranı
        self.learning_rate = 0.001  # Öğrenme hızı
        
        # PyTorch cihaz seçimi (GPU varsa kullan)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ana ve hedef ağlar
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        
        # Eğitim parametreleri
        self.batch_size = 32  # Mini-batch boyutu
        self.update_target_every = 100  # Hedef ağ güncelleme sıklığı
        self.train_counter = 0  # Eğitim adım sayacı
        
        # Eylem sınırları ve güncelleme parametreleri
        self.action_bounds = (0, 2)  # PID parametreleri için alt ve üst sınırlar
        self.tau = 0.001  # Yumuşak güncelleme katsayısı
        
        # Keşif mekanizması
        self.noise = OUNoise(action_size)  # Ornstein-Uhlenbeck gürültüsü
        self.min_epsilon = 0.1  # Minimum keşif oranı
        
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

    def remember(self, state, action, reward, next_state):
        """Deneyimi hafızaya kaydet"""
        self.memory.append((state, action, reward, next_state))
    
    def _soft_update(self):
        """Hedef ağı yumuşak güncelle"""
        for target_param, local_param in zip(self.target_model.parameters(), 
                                           self.model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
    
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

    def get_action(self, state):
        """Mevcut duruma göre eylem seç"""
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            raw_actions = self.model(state_tensor).cpu().numpy()
            
        # Keşif stratejisi
        if np.random.random() < self.epsilon:
            noise = self.noise.sample()
            raw_actions = np.clip(raw_actions + noise, 0, 1)
            
        # Eylemleri PID parametre aralığına ölçekle
        scaled_actions = raw_actions * (self.action_bounds[1] - self.action_bounds[0]) + self.action_bounds[0]
        return raw_actions, scaled_actions

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
        
    def update_parameters(self, new_params):
        """PID parametrelerini yumuşak güncelle"""
        delta = new_params - self.last_params
        clipped_delta = np.clip(delta, -self.max_change_rate, self.max_change_rate)
        smooth_params = self.last_params + self.alpha * clipped_delta
        
        self.last_params = smooth_params
        self.kp, self.ki, self.kd = smooth_params
        
    def compute(self, current_state):
        """PID kontrol çıkışını hesapla"""
        error = self.target - current_state[0]
        
        # Anti-windup ile integral hesaplama
        self.integral = np.clip(self.integral + error, -self.max_integral, self.max_integral)
        
        # Türev hesaplama
        derivative = error - self.last_error
        
        # PID çıkışı
        output = (self.kp * error + 
                 self.ki * self.integral + 
                 self.kd * derivative)
        
        self.last_error = error
        return output

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
        
        # Ödül parametreleri
        self.reward_scale = 1.0
        self.stability_weight = 0.3
        self.change_weight = 0.2
        self.error_threshold = 10.0
        
        # Veri okuma tamponu
        self.data_buffer = []
        self.max_buffer_size = 5
        self.min_valid_readings = 3
        
    def read_sensors(self):
        """Sensör verilerini oku ve filtrele"""
        try:
            if self.arduino.in_waiting > 0:
                data = self.arduino.readline().decode().strip()
                values = list(map(float, data.split(',')))
                
                if len(values) == self.state_size:
                    # Tampon belleğe ekle
                    self.data_buffer.append(values)
                    if len(self.data_buffer) > self.max_buffer_size:
                        self.data_buffer.pop(0)
                    
                    # Medyan filtreleme
                    if len(self.data_buffer) >= self.min_valid_readings:
                        median_values = np.median(self.data_buffer, axis=0)
                        self.current_state = np.array(median_values)
                        return True
                else:
                    print(f"Beklenmeyen veri boyutu: {len(values)}, beklenen: {self.state_size}")
                    
        except (ValueError, IndexError, UnicodeDecodeError) as e:
            print(f"Veri okuma hatası: {e}")
        except Exception as e:
            print(f"Beklenmeyen hata: {e}")
            
        return False
    
    def normalize_error(self, error):
        """Hatayı normalize et"""
        return np.clip(error / self.error_threshold, -1, 1)
    
    def calculate_reward(self, error):
        """Ödül fonksiyonu"""
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
    
    def run_episode(self):
        """Ana kontrol döngüsü"""
        try:
            last_state = None
            episode_steps = 0
            
            while True:
                if self.read_sensors():
                    current_state = self.current_state.copy()
                    episode_steps += 1
                    
                    # RL ajanından eylem al
                    raw_actions, pid_params = self.rl_agent.get_action(current_state)
                    
                    # PID kontrolü
                    self.pid_controller.update_parameters(pid_params)
                    control_output = self.pid_controller.compute(current_state)
                    
                    # Arduino'ya kontrol sinyali gönder
                    self.arduino.write(str(control_output).encode() + b'\n')
                    
                    # Öğrenme
                    if last_state is not None:
                        reward = self.calculate_reward(self.pid_controller.last_error)
                        self.rl_agent.remember(last_state, raw_actions, reward, current_state)
                        self.rl_agent.train()
                    
                    # Keşif oranını güncelle
                    if episode_steps % 100 == 0:
                        self.rl_agent.epsilon = max(
                            self.rl_agent.min_epsilon,
                            self.rl_agent.epsilon * self.rl_agent.epsilon_decay
                        )
                        self.rl_agent.noise.reset()
                    
                    last_state = current_state
                    
                    # Durum bilgilerini yazdır
                    print(f"Durum: {current_state}")
                    print(f"PID Parametreleri: {pid_params}")
                    print(f"Kontrol Çıkışı: {control_output:.2f}")
                    print(f"Ödül: {reward:.2f}")
                    print(f"Epsilon: {self.rl_agent.epsilon:.3f}")
                    
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nProgram sonlandırıldı")
            self.arduino.close()

# Ana program
if __name__ == "__main__":
    controller = RobotController()
    controller.run_episode() 