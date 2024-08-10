#include<Wire.h>
#include <SPI.h>

char control[6] = {'0','0','0','0'};
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
  while(Wire.available() > 0){
    Wire.read();
  }
}

void updateVariables() {
  for(int i = 0; i < 4; i++){
    control[i] = '!' + (rand() % 80);
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
