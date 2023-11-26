#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
int start = 0;
int turn = 1;
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x47);
int speeds = 3000;
int echoPin = 13;
int trigPin = 12;
char blu;
int Ultrasonic_Ranging() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  int distance = pulseIn(echoPin, HIGH);  // reading the duration of high
  // level
  distance = distance / 58;  // Transform pulse time to distance
  delay(50);
  return distance;
}
void advance()  // going forward
{
  pwm.setPWM(0, 0, 4095);
  pwm.setPWM(1, 0, speeds);
  pwm.setPWM(2, 0, 4095);
  pwm.setPWM(3, 0, speeds);
  pwm.setPWM(4, 0, 4095);
  pwm.setPWM(5, 0, speeds);
  pwm.setPWM(6, 0, 4095);
  pwm.setPWM(7, 0, speeds);
}
void turnR() {
  pwm.setPWM(0, 0, 4095);
  pwm.setPWM(1, 0, 4000);
  pwm.setPWM(2, 0, 4095);
  pwm.setPWM(3, 0, 4000);
  pwm.setPWM(4, 0, 0);
  pwm.setPWM(5, 0, 4000);
  pwm.setPWM(6, 0, 0);
  pwm.setPWM(7, 0, 4000);
}
void turnL() {
  pwm.setPWM(0, 0, 0);
  pwm.setPWM(1, 0, 4000);
  pwm.setPWM(2, 0, 0);
  pwm.setPWM(3, 0, 4000);
  pwm.setPWM(4, 0, 4095);
  pwm.setPWM(5, 0, 4000);
  pwm.setPWM(6, 0, 4095);
  pwm.setPWM(7, 0, 4000);
}
void stopp()  //stop
{
  pwm.setPWM(1, 0, 0);
  pwm.setPWM(3, 0, 0);
  pwm.setPWM(5, 0, 0);
  pwm.setPWM(7, 0, 0);
}
void back()  //back
{
  pwm.setPWM(0, 0, 0);
  pwm.setPWM(1, 0, speeds);
  pwm.setPWM(2, 0, 0);
  pwm.setPWM(3, 0, speeds);
  pwm.setPWM(4, 0, 0);
  pwm.setPWM(5, 0, speeds);
  pwm.setPWM(6, 0, 0);
  pwm.setPWM(7, 0, speeds);
}


void setup() {
  Serial.setTimeout(1);
  pinMode(3, OUTPUT);
  Serial.begin(115200);  //set baud rate to 9600
  pwm.begin();
  pwm.setPWMFreq(60);
  stopp();
}
void loop() {
  digitalWrite(3, HIGH);
  start = 0;
  turn = 1;
  int distance = Ultrasonic_Ranging();
  bool can = true;
  // Serial.println(distance);
  // advance(); //go forward for 1s
  if (Serial.available() > 0) {
    // if (distance <= 15) {
    //   // back();
    //   // delay(1000);
    //   stopp();
    //   can = false;
    // }
    // }
    // String in = Serial.readString();
    // println(in);
    String x = Serial.readString();
    turn = x.substring(0, 1).toInt();
    start = x.substring(2).toInt();
    Serial.println(start);
    Serial.println(turn);
    // blu = Serial.read();
    // Serial.println(blu);
    if (start == 1) {
      digitalWrite(3, LOW);
      // if (can == true){
      advance();
      // }

    }

    else if (start == 0) {
      stopp();
    }

    else if (start == 2) {
      back();
    }

    if (turn == 0) {
      turnR();
    } else if (turn == 2) {
      turnL();
    }
    // Serial.println(in);
    // Serial.println(speeds);
  }
}
