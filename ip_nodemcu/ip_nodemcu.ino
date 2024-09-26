#include <ESP8266WiFi.h>

// Replace with your network credentials
const char* ssid = "realme P1 5G";
const char* password = "h9epyfaa";

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  // Wait for connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  // Print the IP address
  Serial.println();
  Serial.print("Connected to WiFi. IP address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  // Nothing to do herex
}