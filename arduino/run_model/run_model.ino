#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <math.h>

#include "model.h"

#define MOTION_THRESHOLD 1.0
#define NUM_SAMPLES 75

BLEService bluetoothService("19B10010-E8F2-537E-4F6C-D104768A1214");
BLEByteCharacteristic predictionCharacteristic("19B10012-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify);

bool isCapturing = false;

int numSamplesRead = 0;

const char *ACTIVITIES[] = {
  "Star-jumping", "Shoulder-front-rotations", "Lateral-raises", "Shoulder-external-rotations", "Shoulder-internal-rotations", "Standing"
};

const int NUMBER_OF_ACTIVITIES = sizeof(ACTIVITIES) / sizeof(int);

namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  constexpr int kTensorArenaSize = 40 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];

}

void setup() {

  // Initialize IMU sensors
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // Initialize Bluetooth
  if (!BLE.begin()) {
    Serial.println("starting Bluetooth® Low Energy module failed!");
    while (1);
  }

  BLE.setLocalName("Testing");
  BLE.setAdvertisedService(bluetoothService);
  bluetoothService.addCharacteristic(predictionCharacteristic);
  BLE.addService(bluetoothService);
  BLE.advertise();
  Serial.println("Bluetooth® device active, waiting for connections...");

  // Error logging
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(quantized_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

}

void loop() {

  BLE.poll();

  // Variables to hold IMU data
  float aX, aY, aZ;

  // Wait for motion above the threshold setting
  while (!isCapturing) {
    if (IMU.accelerationAvailable()) {

      IMU.readAcceleration(aX, aY, aZ);

      float average = fabs(aX) + fabs(aY) + fabs(aZ);

      if (average >= MOTION_THRESHOLD) {
        isCapturing = true;
        numSamplesRead = 0;
        break;
      }
    }
  }

     while (isCapturing) {

    // Check if acceleration data is available
    if (IMU.accelerationAvailable()) {

      IMU.readAcceleration(aX, aY, aZ);

      float normalizedX = aX / 4.0;
      float normalizedY = aY / 4.0;
      float normalizedZ = aZ / 4.0;

      int8_t quantizedX = (normalizedX / model_input->params.scale) + model_input->params.zero_point;
      int8_t quantizedY = (normalizedY / model_input->params.scale) + model_input->params.zero_point;
      int8_t quantizedZ = (normalizedZ / model_input->params.scale) + model_input->params.zero_point;

      model_input->data.int8[numSamplesRead * 3 + 0] = quantizedX;
      model_input->data.int8[numSamplesRead * 3 + 1] = quantizedY;
      model_input->data.int8[numSamplesRead * 3 + 2] = quantizedZ;
    
      numSamplesRead++;

      if (numSamplesRead == NUM_SAMPLES) {

        isCapturing = false;
        
        // Run inference
        TfLiteStatus invokeStatus = interpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          Serial.println("Error: Invoke failed!");
          while (1);
          return;
        }

        int maxIndex = 0;
        float maxValue = 0;
        for (int i = 0; i < NUMBER_OF_ACTIVITIES; i++) {
          int8_t value = model_output->data.int8[i];
          float dequantizedValue = (value - model_output->params.zero_point) * model_output->params.scale;

          if(dequantizedValue > maxValue){
            maxValue = dequantizedValue;
            maxIndex = i;
          }

        }
        
        predictionCharacteristic.writeValue(maxIndex);

        Serial.print(ACTIVITIES[maxIndex]);
        Serial.print(": ");
        Serial.println(maxValue);

      }
    }
  }
}