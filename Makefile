OutDir = ./Debug/
CXX = g++
NVCC = nvcc
DEBUG = -g
NVCCFLAGS = -O3 -arch=sm_52
CXXFLAGS = $(DEBUG) -O -O0 -O1 -O2 -O3 -std=c++11 -Wall
IncludePath = -I. -I/opt/cuda/include/
LibPaths = -L. -L/opt/cuda/lib64/
Libs = -lcuda -lcudart -lcurand -lcublas 
LFLAGS = $(LibPaths) $(Libs)

CUBLAS-ML: Directories randinitweights.cu.o costfunctions.cu.o activations.cu.o kernels.cu.o cublasNN.cpp.o main.cpp.o
	$(CXX) -o $(OutDir)CUBLAS-ML $(OutDir)main.cpp.o $(OutDir)cublasNN.cpp.o $(OutDir)randinitweights.cu.o $(OutDir)costfunctions.cu.o \
	$(OutDir)activations.cu.o $(OutDir)kernels.cu.o $(LFLAGS)

Directories:
	mkdir -p $(OutDir)

kernels.cu.o: #kernels.cu
	$(NVCC) -c $(NVCCFLAGS) kernels.cu -o $(OutDir)kernels.cu.o

activations.cu.o: #kernels.cu
	$(NVCC) -c $(NVCCFLAGS) activations.cu -o $(OutDir)activations.cu.o

costfunctions.cu.o: #costfunctions.cu
	$(NVCC) -c $(NVCCFLAGS) costfunctions.cu -o $(OutDir)costfunctions.cu.o

randinitweights.cu.o: #randinitweights.cu
	$(NVCC) -c $(NVCCFLAGS) randinitweights.cu -o $(OutDir)randinitweights.cu.o

main.cpp.o: main.cpp
	$(CXX) -c "/home/henry/Coding/C++/CUBLAS-ML/main.cpp" $(CXXFLAGS) -o $(OutDir)main.cpp.o $(IncludePath)

cublasNN.cpp.o: cublasNN.cpp
	$(CXX) -c "/home/henry/Coding/C++/CUBLAS-ML/cublasNN.cpp" $(CXXFLAGS) -o $(OutDir)cublasNN.cpp.o $(IncludePath)

##
## Clean
##
clean:
	$(RM) -r $(OutDir)
