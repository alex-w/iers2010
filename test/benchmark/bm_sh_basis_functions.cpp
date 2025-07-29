#include "costg_utils.hpp"
#include "datetime/calendar.hpp"
#include "eigen3/Eigen/Eigen"
#include "gravity.hpp"
#include "icgemio.hpp"
#include <benchmark/benchmark.h>

using namespace costg;

/* degree and order, global */
constexpr const int DEGREE = 180;
constexpr const int ORDER = 180;

/* global data for benchmarking */
std::vector<BmOrbit> orbvec;
dso::StokesCoeffs stokes;

/* allocate scratch space for computations */
dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> W(DEGREE + 3,
                                                                  DEGREE + 3);
dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> M(DEGREE + 3,
                                                                  DEGREE + 3);

int load_data() {
  /* read orbit from input file */
  orbvec = parse_orbit("00orbit_itrf.txt");

  /* read gravity model into a StokesCoeffs instance */
  dso::Icgem icgem("EIGEN6-C4.gfc");
  dso::Datetime<dso::nanoseconds> t(
      dso::from_mjdepoch<dso::nanoseconds>(orbvec[0].epoch));

  if (icgem.parse_data(DEGREE, ORDER, t, stokes)) {
    fprintf(stderr, "ERROR Failed reading gravity model!\n");
    return 1;
  }

  return 0;
}

int functionA(
    const std::vector<BmOrbit> &_orbvec, /*const dso::StokesCoeffs &_stokes*/
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> &C,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> &S) {
  int error = 0;
  for (const auto &in : _orbvec) {
    error += dso::gravity::sh_basis_cs_exterior(in.xyz / ::iers2010::Re, DEGREE + 2,
                                           ORDER + 2, C, S);
  }
  return error;
}

int functionB(
    const std::vector<BmOrbit> &_orbvec, /*const dso::StokesCoeffs &_stokes*/
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> &C,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> &S) {
  int error = 0;
  for (const auto &in : _orbvec) {
    error += dso::gravity::sh_basis_cs_exterior2(in.xyz / ::iers2010::Re, DEGREE + 2,
                                            ORDER + 2, C, S);
  }
  return error;
}

/* Benchmarks use the global data */
static void BM_FunctionA(benchmark::State &state) {
  for (auto _ : state) {
    int error = functionA(orbvec, M, W);
    if (error) {
      state.SkipWithError("functionA failed with non-zero code");
    }
  }
}
BENCHMARK(BM_FunctionA);

static void BM_FunctionB(benchmark::State &state) {
  for (auto _ : state) {
    int error = functionB(orbvec, M, W);
    if (error) {
      state.SkipWithError("functionB failed with non-zero code");
    }
  }
}
BENCHMARK(BM_FunctionB);

/* Custom main */
int main(int argc, char **argv) {
  /* 1. Load data BEFORE running benchmarks */
  load_data();

  /* 2. Initialize and run benchmarks */
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
