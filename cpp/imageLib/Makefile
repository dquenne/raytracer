SRC = Convert.cpp Convolve.cpp Image.cpp ImageIO.cpp ImageIOpng.cpp RefCntMem.cpp flowIO.cpp

# codeIO.cpp

DBG = -g
#CXX = g++34
WARN = -W -Wall
OPT ?= -O3
CPPFLAGS = $(OPT) $(WARN) $(DBG)
ARCH := $(shell arch)
IMLIB = libImg.$(ARCH)$(DBG).a

OBJ = $(SRC:.cpp=.o)

all: $(IMLIB)

$(IMLIB): $(OBJ)
	rm -f $(IMLIB)
	ar ruc $(IMLIB) $(OBJ)
	ranlib $(IMLIB)

clean: 
	rm -f $(OBJ) core* *.stackdump *.bak

allclean: clean
	rm -f libImg*.a

depend:
	@makedepend -Y -- $(CPPFLAGS) -- $(SRC) 2>> /dev/null

# DO NOT DELETE THIS LINE -- make depend depends on it.

Convert.o: Image.h RefCntMem.h Error.h Convert.h
Convolve.o: Image.h RefCntMem.h Error.h Convert.h Convolve.h
Image.o: Image.h RefCntMem.h Error.h
ImageIO.o: Image.h RefCntMem.h Error.h ImageIO.h
ImageIOpng.o: Image.h RefCntMem.h Error.h
RefCntMem.o: RefCntMem.h
flowIO.o: imageLib.h Error.h Image.h RefCntMem.h ImageIO.h Convert.h flowIO.h
