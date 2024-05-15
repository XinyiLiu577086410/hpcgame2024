all: generate baseline

generate: generate.cpp
	g++ -O3 generate.cpp -o generate

baseline: baseline.cpp
	g++ -O3 baseline.cpp -o baseline

.PHONY: clean
clean:
	rm generate baseline
