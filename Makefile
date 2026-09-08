CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Iinclude
SOURCES = src/main.cpp src/mfcc.cpp src/kws_engine.cpp src/telemetry.cpp

# OS Detection
ifeq ($(OS),Windows_NT)
    TARGET = aura.exe
    LDFLAGS += -static -static-libgcc -static-libstdc++ -lole32 -lwinmm -lpsapi
    RM = del /Q /F
else
    UNAME_S := $(shell uname -s)
    TARGET = aura
    ifeq ($(UNAME_S),Darwin)
        LDFLAGS += -framework CoreAudio -framework AudioToolbox -framework CoreFoundation -lpthread -lm
    else
        LDFLAGS += -lpthread -ldl -lm
    endif
    RM = rm -f
endif

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS)

clean:
	$(RM) $(TARGET)

.PHONY: all clean
