SWIFTC := "swiftc"
MODULE := "Pgm"
SRC := "pgm.swift"

default: build

build:
    {{SWIFTC}} -emit-library -o lib{{MODULE}}.dylib {{SRC}}

clean:
    rm -f *.dylib *.swiftmodule *.swiftdoc *.swiftsourceinfo