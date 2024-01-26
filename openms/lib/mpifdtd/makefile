MAKE_DIR   := $(PWD)

INCL_PATH  := -I$(MAKE_DIR)/
INCL_PATH  += -I$(MAKE_DIR)/include
INCL_PATH  += -I/${HOME}/local/gsl-2.4-gnu9.4/include

#CC = gcc
CC = mpicc

#CCFLAG = -c -O3 -pthread -Wall -Wextra
# debug
CCFLAGS = -g -O0 -pthread -Wall -Wextra
CCFLAGS   += -DMPION
CCFLAGS   += $(INCL_PATH)

TARGET = libpFDTD.a

AR = ar

OBJ_DIR = $(MAKE_DIR)/obj

CPP_FILES_SRC    := $(wildcard src/*.c)
CPP_FILES_SRC    += $(wildcard src/*.cpp)
OBJ_FILES        += $(patsubst src/%.c, $(MAKE_DIR)/obj/%.o, $(CPP_FILES_SRC))
OBJ_FILES        += $(patsubst src/%.cpp, $(MAKE_DIR)/obj/%.o, $(CPP_FILES_SRC))

all: obj  $(TARGET)

obj:
	@mkdir -p $@


$(TARGET) : $(OBJ_FILES)
	$(AR) rcv $@ $(OBJ_FILES)
	ranlib $@

$(MAKE_DIR)/obj/%.o   : src/%.c
	@echo $< " --> " $@
	@$(CC) $(CCFLAGS) -c $< -o $@

$(MAKE_DIR)/obj/%.o   : src/%.cpp
	@echo $< " --> " $@
	@$(CC) $(CCFLAGS) -c $< -o $@

clean :
	@$(RM) -f obj/*.o

install :
	cp libpFDTD.a bin/
	cp $(MAKE_DIR)/include/*.hpp bin/
	cp $(MAKE_DIR)/include/pFDTDvar.h bin/
	#chmod +x $(MAKE_DIR)pFDTD
	#cp $(MAKE_DIR)pFDTD bin/



