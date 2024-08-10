#include<Wire.h>
#include <SPI.h>

float randsens[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
char control[6] = {'0','0','0','0','0','0'};
float min[6] = {0.0,3.0,5.0,10.0,13.0,15.0};
float max[6] = {4.0,7.0,9.0,14.0,17.0,19.0};
char data[32];
byte receive;
int flag = -1;

void setup() {
  Serial.begin(115200);
  Serial.flush();

  pinMode(MISO,OUTPUT);
  SPCR |= _BV(SPE);
  SPI.attachInterrupt();

  Wire.begin(9);
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
    sendControl(10);
    flag*=-1;
  }
}

void receiveEvent() {
  int index = 0;
  while(Wire.available() > 0){
    data[index] = Wire.read();
    index++;
  }
}

void updateVariables() {

  char temp0[4], temp1[4], temp2[4], temp3[5], temp4[5], temp5[5];

  for(int i = 0; i < 32; i++){
    if(i <= 3) temp0[i] = data[i];
    if(i >=5 && i <= 8) temp1[i-5] = data[i];
    if(i >= 10 && i <= 13) temp2[i-10] = data[i];
    if(i >= 15 && i <= 19) temp3[i-15] = data[i];
    if(i >= 21 && i <= 25) temp4[i-21] = data[i];
    if(i >= 27 && i <= 31) temp5[i-27] = data[i];
  }

  randsens[0] = atof(temp0);
  randsens[1] = atof(temp1);
  randsens[2] = atof(temp2);
  randsens[3] = atof(temp3);
  randsens[4] = atof(temp4);
  randsens[5] = atof(temp5);

  for(int i = 0; i < 6; i++) {
    float threshold = (min[i]+max[i])/2;
    if(randsens[i] < threshold && randsens[i] >= min[i]){
      control[i] = '0';
    }
    if(randsens[i] >= threshold && randsens[i] <= max[i]+0.5){
      control[i] = '1';
    }
    if(randsens[i] < min[i] || randsens[i] > max[i]+0.5){
      control[i] = 'z';
    }
  }
}

void sendControl(int address) {
  Wire.beginTransmission(address);
  Wire.write(control);
  Wire.endTransmission();
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
