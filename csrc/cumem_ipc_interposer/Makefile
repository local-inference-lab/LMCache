CUDA_HOME ?= /usr/local/cuda
CC ?= cc
CFLAGS ?= -O2
MIN_CUDA_VERSION ?= 13000
CPPFLAGS += -I$(CUDA_HOME)/include
CFLAGS += -fPIC -Wall -Wextra -Werror
LDFLAGS += -shared
LDLIBS += -ldl -pthread

TARGET := liblmcache_cumem_shareable.so
SOURCE := cumem_shareable_interposer.c

.PHONY: all clean check-cuda-version

all: check-cuda-version $(TARGET)

check-cuda-version:
	@test -f "$(CUDA_HOME)/include/cuda.h"
	@$(CC) $(CPPFLAGS) -dM -E -x c -include cuda.h /dev/null | \
		awk -v minimum="$(MIN_CUDA_VERSION)" \
		'/CUDA_VERSION/ { if ($$3 < minimum) exit 1; found=1 } END { exit !found }'

$(TARGET): $(SOURCE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f $(TARGET)
