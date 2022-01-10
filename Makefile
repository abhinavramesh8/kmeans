CC = nvcc 
CPP_SRCS = ./src/*.cpp
CU_SRCS = ./src/*.cu
INC = ./src/
OPTS = --std c++14

EXEC = bin/kmeans

all: clean compile

compile:
	$(CC) $(CPP_SRCS) $(CU_SRCS) $(OPTS) -I$(INC) -o $(EXEC)

clean:
	rm -f $(EXEC)