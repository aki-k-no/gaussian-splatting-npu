# compiler and flags
CC = g++
CFLAGS = -Wall -O2

#include paths
INCLUDES = -I/usr/include/eigen3 -I/usr/include/opencv4

# linker flags
LD_LIBS = `pkg-config --cflags --libs opencv4`

# source and object files
SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)

# target executable
TARGET = base

# rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LD_LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

run:
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean