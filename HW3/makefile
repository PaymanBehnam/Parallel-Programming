CC=/usr/local/cuda/bin/nvcc
INCLUDE=-I/usr/local/cuda/include \

SOURCE=sobel.cu
EXECUTABLE=sobel
FLAGS=

#FOR DEBUGGING
#FLAGS=-g

$(EXECUTABLE): $(SOURCE)
	$(CC) $(FLAGS) $(INCLUDE) $< -o $@

clean:
