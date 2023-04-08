CXXFLAGS += -O3 -std=c++20 -I../src
BINARIES = solve_challenge2

.PHONY: all clean

all: $(BINARIES)

%: %.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -fv $(BINARIES)