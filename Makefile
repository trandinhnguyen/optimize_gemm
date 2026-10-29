SOURCES := $(wildcard src/*.cpp)

all: bench

bench: run
	HIP_VISIBLE_DEVICES=7 ./run 6016

run: $(SOURCES)
	hipcc -o run $(SOURCES) -Iinclude

clean:
	rm -rf run