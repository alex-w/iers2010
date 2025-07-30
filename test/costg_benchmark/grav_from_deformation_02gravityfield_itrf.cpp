#include "costg_utils.hpp"
#include "datetime/calendar.hpp"
#include "eigen3/Eigen/Eigen"
#include "gravity.hpp"
#include "icgemio.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

/* This test is a copy of the 02(itrf) check, but instead of using the
 * 'fitting' function to compute gravity, it uses the 'deformation' function.
 * This makes sure that gravity computed in the deformation function is
 * the same as the one computed in the cunningham_normalized function.
 */

constexpr const int DEGREE = 180;
constexpr const int ORDER = 180;
constexpr const double TOLERANCE = 1e-11; /* [m/sec**2] */

using namespace costg;

int main(int argc, char *argv[]) {
  if (argc != 4) {
    fprintf(stderr,
            "Usage: %s [00orbit_itrf.txt] [EIGEN6-C4.gfc] "
            "[02gravityfield_itrf.txt]\n",
            argv[0]);
    return 1;
  }

  /* read orbit from input file */
  const auto orbvec = parse_orbit(argv[1]);

  /* read accleration from input file */
  const auto accvec = parse_acceleration(argv[3]);

  /* read gravity model into a StokesCoeffs instance */
  dso::Icgem icgem(argv[2]);
  dso::StokesCoeffs stokes;
  dso::Datetime<dso::nanoseconds> t(
      dso::from_mjdepoch<dso::nanoseconds>(orbvec[0].epoch));
  if (icgem.parse_data(DEGREE, ORDER, t, stokes)) {
    fprintf(stderr, "ERROR Failed reading gravity model!\n");
    return 1;
  }

  /* checks */
  assert(stokes.max_degree() == DEGREE);
  assert(stokes.max_order() == ORDER);

  /* allocate scratch space for computations */
  dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> W(DEGREE + 3,
                                                                    DEGREE + 3);
  dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> M(DEGREE + 3,
                                                                    DEGREE + 3);

  /* compare results epoch by epoch */
  Eigen::Matrix<double, 3, 1> a;
  Eigen::Matrix<double, 3, 3> g;
  auto acc = accvec.begin();
  for (const auto &in : orbvec) {
    /* for the test we are computing an SH expansion starting from (2,0) */
    stokes.C(0, 0) = 0e0;
    stokes.C(1, 0) = stokes.C(1, 1) = stokes.S(1, 1) = 0e0;

    /* compute acceleration for given epoch/position; this is the 'correct'
     * call, i.e. what we should be doing.
     */
    if (dso::sh2gradient_cunningham(stokes, in.xyz, a, g, DEGREE, ORDER, -1, -1,
                                    &W, &M)) {
      fprintf(stderr, "ERROR Failed computing acceleration/gradient\n");
      return 1;
    }

    printf("[Results (1)] %.12f %.12f %.12f\n", std::abs(acc->axyz(0) - a(0)),
           std::abs(acc->axyz(1) - a(1)), std::abs(acc->axyz(2) - a(2)));

    assert(std::abs(acc->axyz(0) - a(0)) < TOLERANCE);
    assert(std::abs(acc->axyz(1) - a(1)) < TOLERANCE);
    assert(std::abs(acc->axyz(2) - a(2)) < TOLERANCE);

    /* compute acceleration for given epoch/position (we use the deformation
     * function here, but will only check the gravity results)
     */
    double potential;
    Eigen::Vector3d dr;
    if (dso::sh_deformation(in.xyz, stokes, a, potential, dr, DEGREE, ORDER, &W,
                            &M)) {
      fprintf(stderr, "ERROR Failed computing acceleration/gradient\n");
      return 1;
    }

    printf(
        "[Results (2)] %.12f %.12f %.12f V=%.12f dr=(%.2f, %.2f, %.2f)[mm]\n",
        std::abs(acc->axyz(0) - a(0)), std::abs(acc->axyz(1) - a(1)),
        std::abs(acc->axyz(2) - a(2)), potential, dr(0) * 1e3, dr(1) * 1e3,
        dr(2) * 1e3);

    /* get COSTG result */
    if (acc->epoch != in.epoch) {
      fprintf(stderr, "ERROR Failed to match epochs in input files\n");
      return 1;
    }

    assert(std::abs(acc->axyz(0) - a(0)) < TOLERANCE);
    assert(std::abs(acc->axyz(1) - a(1)) < TOLERANCE);
    assert(std::abs(acc->axyz(2) - a(2)) < TOLERANCE);

    ++acc;
  }

  return 0;
}
