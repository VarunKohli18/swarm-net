#include<Wire.h>
#include <SPI.h>

float randsens[3] = {0.0,0.0,0.0};
float min[3] = {0.0,5.0,10.0};
float max[3] = {4.0,9.0,14.0};
const int LED[6] = {3,5,7,8,9,2};
char data[32];
byte receive;
int flag = -1;

void setup() {
  Serial.begin(115200);
  Serial.flush();

  for(int i = 0; i < 6; i++){
    pinMode(LED[i],OUTPUT);
  }

  pinMode(MISO,OUTPUT);
  SPCR |= _BV(SPE);
  SPI.attachInterrupt();

  Wire.begin(12);
  Wire.flush();
  Wire.onReceive(receiveEvent);
}

ISR(SPI_STC_vect){
  receive = SPDR;
  sram_dump();
  flag*=-1;
}

void loop() {
  if(flag == 1) {
    updateVariables();
    flag*=-1;
  }
}

void receiveEvent() {
  int index = 0;
  while(Wire.available() > 0){
    data[index] = Wire.read();
    index++;
  }
  // Serial.println(data);
}

void updateVariables() {

  char temp0[4], temp1[4], temp2[5];

  for(int i = 0; i < 32; i++){
    if(i <= 3) temp0[i] = data[i];
    if(i >=5 && i <= 8) temp1[i-5] = data[i];
    if(i >= 10 && i <= 14) temp2[i-10] = data[i];
  }

  randsens[0] = atof(temp0);
  randsens[1] = atof(temp1);
  randsens[2] = atof(temp2);

  for(int i = 3; i < 6; i++) {
    float threshold = (min[i]+max[i])/2;
    if(randsens[i] < threshold && randsens[i] >= min[i]){
      digitalWrite(LED[i],LOW);
    }
    if(randsens[i] >= threshold && randsens[i] <= max[i]+0.5){
      digitalWrite(LED[i],HIGH);
    }
    if(randsens[i] < min[i] || randsens[i] > max[i]+0.5){
      digitalWrite(LED[i],LOW);
    }
  }
}

void sram_dump() {
  uint16_t address;
  uint8_t byte_at_address, new_line;
  address = 0x0100;
  new_line = 1;

  while (address <= 0x08FF) {
    byte_at_address = *(byte *)address;

    if (((byte_at_address >> 4) & 0x0F) > 9)UDR0 = 55 + (byte_at_address >> 4 & 0x0F);
    else UDR0 = 48 + ((byte_at_address >> 4) & 0x0F);
    while ( !( UCSR0A & (1 << UDRE0)) );

    if ((byte_at_address & 0x0F) > 9)UDR0 = 55 + (byte_at_address & 0x0F);
    else UDR0 = 48 + (byte_at_address & 0x0F);
    while ( !( UCSR0A & (1 << UDRE0)) );

    UDR0 = 0x20;
    while ( !( UCSR0A & (1 << UDRE0)) );

    if (new_line == 64) {
      new_line = 0;
      UDR0 = 0x0A;
      while ( !( UCSR0A & (1 << UDRE0)) );
      UDR0 = 0x0D;
      while ( !( UCSR0A & (1 << UDRE0)) );
    }

    address ++;
    new_line ++;
  }
}
