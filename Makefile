
CXXFLAGS := -std=c++17 -Werror $(CXXFLAGS)
LDFLAGS := -fuse-ld=gold $(shell pkg-config opencv --libs) $(LDFLAGS)

all: camera-feed

run: | camera-feed
	./camera-feed

camera-feed: camera-feed.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@