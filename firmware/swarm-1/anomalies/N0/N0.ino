#include <SPI.h>

// SCK 13
// MISO 12
// MOSI 11
// SS 10,9,7

int ss[3] = {10,9,7};
int data[4];

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Serial.flush();

  for(int i = 0; i < 3; i++) {
    pinMode(ss[i],OUTPUT);
    digitalWrite(ss[i],HIGH);
  }

  SPI.begin();
  SPI.setClockDivider(SPI_CLOCK_DIV8);
}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available()>0){  
    char send = Serial.read();
    if(send == 'r'){
      for(int i = 0; i < 3; i++) {
        digitalWrite(ss[i],LOW);
        byte receive=SPI.transfer(send);
        digitalWrite(ss[i],HIGH);
      }
      sram_dump();
      update_variables();
      send = 'n';
    }
  }
}

void update_variables(){
  for(int i=0; i < 4; i++){
    data[i] = rand() % 128;
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