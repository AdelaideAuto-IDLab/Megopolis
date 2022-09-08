NVCCFLAGS = -std=c++14 -use_fast_math --compiler-options -Wall
GPPFLAGS = -std=c++14 -lpthread -Wall

RESAMPLETEST = resample_test
RESAMPLETESTL1 = resample_test_l1
FILTER_BENCH = filter_bench

RESAMPLETEST_FILES = src/benchmarks/resample_testing.cu
RESAMPLETEST_REQ = src/filter/*.cuh src/util/*.cuh src/resampling/* src/benchmarks/resample_config.hpp

FILTER_FILES = src/filter_tests/filter_app.cu
FILTER_REQ = src/filter/*.cuh src/util/*.cuh src/resampling/* src/benchmarks/resample_config.hpp src/filter_tests/*.cuh

all: $(RESAMPLETEST) $(FILTER_BENCH)

$(RESAMPLETEST): $(RESAMPLETEST_FILES) $(RESAMPLETEST_REQ)
	nvcc -o target/$(RESAMPLETEST) $(RESAMPLETEST_FILES) $(NVCCFLAGS)

$(RESAMPLETESTL1): $(RESAMPLETEST_FILES) $(RESAMPLETEST_REQ)
	nvcc -o target/$(RESAMPLETESTL1) $(RESAMPLETEST_FILES) $(NVCCFLAGS) -Xptxas -dlcm=ca

$(FILTER_BENCH): $(FILTER_FILES) $(FILTER_REQ)
	nvcc -o target/$(FILTER_BENCH) $(FILTER_FILES) $(NVCCFLAGS)

clean:
	$(RM) target/$(RESAMPLETEST) target/$(RESAMPLETESTL1) target/$(FILTER_BENCH)
