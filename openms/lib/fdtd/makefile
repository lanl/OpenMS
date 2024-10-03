CC = gcc

CCFLAG = -c -O2 -I/${HOME}/polariton/OpenMS/openms/lib/deps/include/
OBJECTS = incoherent.o metalobj.o memory.o parameter.o input.o sigma.o timeupdate.o source.o output.o farfield.o energy.o transmap.o
TARGET = libpFDTD.a
SDIR = ./

$(TARGET) : $(OBJECTS)
	$(AR) rcv $@ $(OBJECTS)
	ranlib $@

incoherent.o :  $(SDIR)incoherent.c
	$(CC)  $(CCFLAG) $(SDIR)incoherent.c
metalobj.o :  $(SDIR)metalobj.c
	$(CC)  $(CCFLAG) $(SDIR)metalobj.c
memory.o :  $(SDIR)memory.c
	$(CC)  $(CCFLAG) $(SDIR)memory.c
parameter.o :  $(SDIR)parameter.c
	$(CC)  $(CCFLAG)  $(SDIR)parameter.c
input.o :  $(SDIR)input.c
	$(CC)  $(CCFLAG)  $(SDIR)input.c
sigma.o :  $(SDIR)sigma.c
	$(CC)  $(CCFLAG)  $(SDIR)sigma.c
timeupdate.o :  $(SDIR)timeupdate.c
	$(CC)  $(CCFLAG)  $(SDIR)timeupdate.c
source.o :  $(SDIR)source.c
	$(CC)  $(CCFLAG)  $(SDIR)source.c
output.o :  $(SDIR)output.c
	$(CC)  $(CCFLAG)  $(SDIR)output.c
farfield.o :  $(SDIR)farfield.c
	$(CC)  $(CCFLAG)  $(SDIR)farfield.c
energy.o :  $(SDIR)energy.c
	$(CC)  $(CCFLAG)  $(SDIR)energy.c
transmap.o :  $(SDIR)transmap.c
	$(CC)  $(CCFLAG)  $(SDIR)transmap.c

clean :
	rm $(OBJECTS)

install :
	cp libpFDTD.a bin/
	cp $(SDIR)pFDTD.h bin/
	cp $(SDIR)pFDTDvar.h bin/
	rm *.o
	chmod +x $(SDIR)pFDTD
	cp $(SDIR)pFDTD bin/
