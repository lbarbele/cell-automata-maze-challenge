CXXFLAGS += -O3 -std=c++20 -I../src
ALL_TESTS = lvl1_pt1

.PHONY: all clean

all: $(ALL_TESTS)

%: %.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -fv $(ALL_TESTS)