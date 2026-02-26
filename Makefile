SWIFTC = swiftc
MODULE = Pgm
SRC = pgm.swift
LIB = lib$(MODULE).dylib

all: $(LIB)

$(LIB): $(SRC)
	$(SWIFTC) -emit-library -o $(LIB) $(SRC)

clean:
	rm -f *.dylib *.swiftmodule *.swiftdoc *.swiftsourceinfo