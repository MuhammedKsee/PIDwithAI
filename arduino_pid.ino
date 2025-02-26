// Gerekli kütüphaneleri içe aktar
#include <Wire.h>      // I2C haberleşme için
#include <MPU6050.h>   // Gyro sensörü için
#include <NewPing.h>   // Ultrasonik sensör için

// MPU6050 sensör nesnesi oluştur
MPU6050 mpu;

// Ultrasonik sensör pinlerini tanımla
#define TRIGGER_PIN  12  // Ultrasonik sensör tetikleme pini
#define ECHO_PIN     11  // Ultrasonik sensör echo pini
#define MAX_DISTANCE 200 // Maksimum ölçüm mesafesi (cm)

// Motor kontrol pini
#define MOTOR_PIN 9     // PWM çıkış pini

// Ultrasonik sensör nesnesi oluştur
NewPing sonar(TRIGGER_PIN, ECHO_PIN, MAX_DISTANCE);

void setup() {
    // Seri haberleşmeyi başlat
    Serial.begin(9600);
    
    // I2C haberleşmeyi başlat
    Wire.begin();
    
    // MPU6050'yi başlat ve yapılandır
    mpu.initialize();
    
    // Motor pinini çıkış olarak ayarla
    pinMode(MOTOR_PIN, OUTPUT);
}

void loop() {
    // Gyro verilerini oku
    int16_t ax, ay, az;  // İvmeölçer değişkenleri
    mpu.getAcceleration(&ax, &ay, &az);  // İvmeölçer verilerini al
    
    // Mesafe ölçümü yap
    float distance = sonar.ping_cm();  // Mesafeyi cm cinsinden oku
    
    // Verileri seri porta gönder (Gyro x,y,z + Mesafe)
    Serial.print(ax); Serial.print(",");
    Serial.print(ay); Serial.print(",");
    Serial.print(az); Serial.print(",");
    Serial.println(distance);
    
    // Python'dan kontrol sinyalini al ve uygula
    if (Serial.available() > 0) {
        // Kontrol çıkışını oku
        float controlOutput = Serial.parseFloat();
        
        // PWM çıkışını sınırla (0-255)
        int pwmOutput = constrain(controlOutput, 0, 255);
        
        // Motora PWM sinyali uygula
        analogWrite(MOTOR_PIN, pwmOutput);
    }
    
    // Döngü gecikmesi
    delay(100);  // 100ms bekleme (10Hz örnekleme)
} 