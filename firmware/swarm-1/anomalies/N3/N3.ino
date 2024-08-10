#include <Wire.h>
#include <SPI.h>

const int LED[6] = {3,5,7,2,4,6};
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

  Wire.begin(10);
  Wire.flush();
  Wire.onReceive(receiveEvent);
}

ISR(SPI_STC_vect){
  receive = SPDR;
  sram_dump();
  flag*=-1;
}

void loop() {
  if(flag == 1){
    for(int i = 0; i < 6; i++){
      int r = rand() % 2;
      if(r == 0) {
        digitalWrite(LED[i],LOW);
      }
      if(r == 1) {
        digitalWrite(LED[i],HIGH);
      }
    }
    flag*=-1;
  }
}

void receiveEvent() {
  int index = 0;
  while (Wire.available()) {
    Wire.read();
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