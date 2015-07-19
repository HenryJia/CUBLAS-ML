OutDir = ./Debug/
CXX = g++
NVCC = nvcc
DEBUG = -g
CXXFLAGS = $(DEBUG) -O1 -O -O3 -O0 -O2 -std=c++11 -Wall
IncludePath = -I. -I/opt/cuda/include/
LibPaths = -L. -L/opt/cuda/lib64/
Libs = -lcuda -lcudart -lcurand -lcublas 
LFLAGS = $(LibPaths) $(Libs)

CUBLAS-ML: Directories kernels.cu.o cublasNN.cpp.o main.cpp.o
	$(CXX) -o $(OutDir)CUBLAS-ML $(OutDir)main.cpp.o $(OutDir)cublasNN.cpp.o $(OutDir)kernels.cu.o $(LFLAGS)

Directories:
	mkdir -p $(OutDir)

kernels.cu.o: #kernels.cu
	$(NVCC) -c kernels.cu -o $(OutDir)kernels.cu.o

main.cpp.o: main.cpp
	$(CXX) -c "/home/henry/Coding/C++/CUBLAS-ML/main.cpp" $(CXXFLAGS) -o $(OutDir)main.cpp.o $(IncludePath)

cublasNN.cpp.o: cublasNN.cpp
	$(CXX) -c "/home/henry/Coding/C++/CUBLAS-ML/cublasNN.cpp" $(CXXFLAGS) -o $(OutDir)cublasNN.cpp.o $(IncludePath)

##
## Clean
##
clean:
	$(RM) -r $(OutDir)