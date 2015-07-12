##
## Auto Generated makefile by CodeLite IDE
## any manual changes will be erased      
##
## Debug
ProjectName            :=CUBLAS-ML
ConfigurationName      :=Debug
WorkspacePath          := "/home/henry/Coding/C++"
ProjectPath            := "/home/henry/Coding/C++/CUBLAS-ML"
IntermediateDirectory  :=./Debug
OutDir                 := $(IntermediateDirectory)
CurrentFileName        :=
CurrentFilePath        :=
CurrentFileFullPath    :=
User                   :=
Date                   :=12/07/15
CodeLitePath           :="/home/henry/.codelite"
LinkerName             :=/usr/bin/g++
SharedObjectLinkerName :=/usr/bin/g++ -shared -fPIC
ObjectSuffix           :=.o
DependSuffix           :=.o.d
PreprocessSuffix       :=.i
DebugSwitch            :=-g 
IncludeSwitch          :=-I
LibrarySwitch          :=-l
OutputSwitch           :=-o 
LibraryPathSwitch      :=-L
PreprocessorSwitch     :=-D
SourceSwitch           :=-c 
OutputFile             :=$(IntermediateDirectory)/$(ProjectName)
Preprocessors          :=
ObjectSwitch           :=-o 
ArchiveOutputSwitch    := 
PreprocessOnlySwitch   :=-E
ObjectsFileList        :="CUBLAS-ML.txt"
PCHCompileFlags        :=
MakeDirCommand         :=mkdir -p
LinkOptions            :=  ./Debug/kernels.cu.o 
IncludePath            :=  $(IncludeSwitch). $(IncludeSwitch). 
IncludePCH             := 
RcIncludePath          := $(IncludeSwitch)Debug/ 
Libs                   := $(LibrarySwitch)cuda $(LibrarySwitch)cudart $(LibrarySwitch)curand $(LibrarySwitch)cublas 
ArLibs                 :=  "cuda" "cudart" "curand" "cublas" 
LibPath                := $(LibraryPathSwitch). $(LibraryPathSwitch)/opt/cuda/lib64/ 

##
## Common variables
## AR, CXX, CC, AS, CXXFLAGS and CFLAGS can be overriden using an environment variables
##
AR       := /usr/bin/ar rcu
CXX      := /usr/bin/g++
CC       := /usr/bin/gcc
CXXFLAGS :=  -g -O1 -O -O3 -O0 -O2 -std=c++11 -Wall -I /opt/cuda/lib64/ -I /opt/cuda/include/  $(Preprocessors)
CFLAGS   :=  -g -O1 -O -O3 -O0 -O2 -std=c++11 -Wall -I /opt/cuda/lib64/ -I /opt/cuda/include/  $(Preprocessors)
ASFLAGS  := 
AS       := /usr/bin/as


##
## User defined environment variables
##
CodeLiteDir:=/usr/share/codelite
Objects0=$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IntermediateDirectory)/cublasNN.cpp$(ObjectSuffix) 



Objects=$(Objects0) 

##
## Main Build Targets 
##
.PHONY: all clean PreBuild PrePreBuild PostBuild
all: $(OutputFile)

$(OutputFile): $(IntermediateDirectory)/.d $(Objects) 
	@$(MakeDirCommand) $(@D)
	@echo "" > $(IntermediateDirectory)/.d
	@echo $(Objects0)  > $(ObjectsFileList)
	$(LinkerName) $(OutputSwitch)$(OutputFile) @$(ObjectsFileList) $(LibPath) $(Libs) $(LinkOptions)

$(IntermediateDirectory)/.d:
	@test -d ./Debug || $(MakeDirCommand) ./Debug

PreBuild:
	@echo Executing Pre Build commands ...
	mkdir -p ./Debug/
	
	nvcc -c kernels.cu -o ./Debug/kernels.cu.o
	@echo Done


##
## Objects
##
$(IntermediateDirectory)/main.cpp$(ObjectSuffix): main.cpp $(IntermediateDirectory)/main.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/henry/Coding/C++/CUBLAS-ML/main.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/main.cpp$(DependSuffix): main.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/main.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/main.cpp$(DependSuffix) -MM "main.cpp"

$(IntermediateDirectory)/main.cpp$(PreprocessSuffix): main.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/main.cpp$(PreprocessSuffix) "main.cpp"

$(IntermediateDirectory)/cublasNN.cpp$(ObjectSuffix): cublasNN.cpp $(IntermediateDirectory)/cublasNN.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/henry/Coding/C++/CUBLAS-ML/cublasNN.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/cublasNN.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/cublasNN.cpp$(DependSuffix): cublasNN.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/cublasNN.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/cublasNN.cpp$(DependSuffix) -MM "cublasNN.cpp"

$(IntermediateDirectory)/cublasNN.cpp$(PreprocessSuffix): cublasNN.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/cublasNN.cpp$(PreprocessSuffix) "cublasNN.cpp"


-include $(IntermediateDirectory)/*$(DependSuffix)
##
## Clean
##
clean:
	$(RM) -r ./Debug/


