.PHONY: all test clean deps tags 

CXX=g++
CXXFLAGS += -g -Wall -O -std=c++11                 

OPENCVLIBS = `pkg-config opencv --cflags --libs`

DEPS_INCLUDE_PATH= $(DLIB_PATH)  -I /usr/local/cuda-10.0/include/ -I /home/asd/Project/MobileNet-YOLO-master2/include 



TARGET = retinaface

LIBS=  -lboost_system -lcaffe -lglog  -lprotobuf -lcudart  -lgflags 

OBJS := $(patsubst %.cpp,%.o,$(wildcard *.cpp))     

  
$(TARGET): $(OBJS) 
	$(CXX)  -o $@  $^ $(LIBS)  $(OPENCVLIBS)  $(DEPS_LIB_PATH) 


%.o:%.cpp
	$(CXX) -c $(CXXFLAGS)  $< $(DEPS_INCLUDE_PATH) 


clean:
	rm -f *.o $(TARGET)
