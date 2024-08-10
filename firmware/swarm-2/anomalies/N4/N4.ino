#include <Wire.h>
#include <SPI.h>

const int randsensThreshold[3] = {0,5,10};
float randsens[3] = {0.0,0.0,0.0};
char buffer[4];
byte receive;
int flag = -1;

void setup() {
  Serial.begin(115200);
  Serial.flush();

  Wire.begin(11);

  pinMode(MISO,OUTPUT);
  SPCR |= _BV(SPE);
  SPI.attachInterrupt();
}

ISR(SPI_STC_vect){
  receive = SPDR;
  sram_dump();
  flag*=-1;
}

void loop() {
  if(flag == 1){
    updateRandsens();
    // sendData(12);
    flag*=-1;
  }
}

void updateRandsens() {
  for(int i = 0; i<3; i++){
    randsens[i] = (rand() % 5) + randsensThreshold[i];
  }
}

void sendData(int address) {
    Wire.beginTransmission(address);
    for(int i = 0; i<3; i++){
      Wire.write(dtostrf(randsens[i],4,2,buffer));
      if(i<2){
        Wire.write(" ");
      }
    }
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
