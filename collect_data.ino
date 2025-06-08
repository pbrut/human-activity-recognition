#include <Arduino_LSM9DS1.h>

// Acceleration threshold in Gs
const float accelerationThreshold = 1.0; 
bool isCapturing = false;

void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
}

void loop() {
  float aX, aY, aZ;

  // wait for significant motion
  while (!isCapturing) {
    if (IMU.accelerationAvailable()) {
      
      // read the acceleration
      IMU.readAcceleration(aX, aY, aZ);
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      if (aSum >= accelerationThreshold) {
        break;
      }
    }
  }

  while (isCapturing) {

    if (IMU.accelerationAvailable()) {
      
      // read the acceleration
      IMU.readAcceleration(aX, aY, aZ);

      // print the data in CSV format
      Serial.print(aX);
      Serial.print(' ');
      Serial.print(aY);
      Serial.print(' ');
      Serial.print(aZ);
      Serial.println();
    }
  }
}