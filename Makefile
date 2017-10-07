# Makefile for ray tracing image generator
# type "make" to compile

SRC = ray-tracer.cpp

DBG = -g
WARN = -W -Wall
OPT ?= -O3
CPPFLAGS = $(OPT) $(WARN) $(DBG) -I$(IMGLIB) -std=c++11
ARCH := $(shell arch)
IMGLIB = ./imageLib/
LDLIBS = -L$(IMGLIB) -lImg.$(ARCH)$(DBG) -lpng -lz

BIN = $(SRC:.cpp=)

all: $(BIN)

clean:
	rm -f $(BIN)
