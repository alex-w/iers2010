#include "gravity.hpp"
#include <array>
#include <cassert>
#include <cmath>

namespace {

/** Spherical harmonics of Earth's gravity potential to acceleration and
 *  gradient using the algorithm due to Cunningham. The acceleration and
 *  gradient are computed in Cartesian components, i.e.
 *
 *  acceleration = (ax, ay, az)
 *
 * , and
 *
 *             | dax/dx dax/dy dax/dz |
 *  gradient = | day/dx day/dy day/dz |
 *             | daz/dx daz/dy daz/dz |
 *
 * @param[in] cs Normalized Stokes coefficients of spherical harmonics
 * @param[in] r  Position vector of satellite (aka point of computation) in
 *               an ECEF frame (e.g. ITRF)
 * @param[in] max_degree Max degree of spherical harmonics expansion
 * @param[in] max_order  Max order of spherical harmonics expansion
 * @param[in] Re Equatorial radius of the Earth
 * @param[in] GM Gravitational constant of Earth
 * @param[out] acc Acceleration in cartesian components in [m/s^2]
 * @param[out] gradient Gradient of acceleration in cartesian components
 * @param[in] W   A convinient storage space, as Column-Wise Lower Triangular
 *                matrix off dimensions at least (max_degree+2, max_degree+2)
 * @param[in] M   A convinient storage space, as Column-Wise Lower Triangular
 *                matrix off dimensions at least (max_degree+2, max_degree+2)
 */
int sh2gradient_cunningham_impl(
    const dso::StokesCoeffs &cs, const Eigen::Matrix<double, 3, 1> &r,
    int max_degree, int max_order, double Re, double GM,
    Eigen::Vector3d &acc, Eigen::Matrix<double, 3, 3> &gradient,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> &W,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>
        &M) noexcept {

  if (max_degree >
      dso::NormalizedLegendreFactors::MAX_SIZE_FOR_ALF_FACTORS - 3) {
    fprintf(stderr,
            "[ERROR] (Static) Size for NormalizedLegendreFactors must be "
            "augmented to perform computation (traceback: %s)\n",
            __func__);
    assert(false);
  }

#ifdef DEBUG
  assert(cs.max_degree() >= cs.max_order());
  assert(max_degree <= cs.max_degree());
  assert(max_order <= cs.max_order());
  assert(W.rows() >= max_degree + 2);
  assert(W.cols() >= max_degree + 2);
  assert(M.rows() >= max_degree + 2);
  assert(M.cols() >= max_degree + 2);
#endif

  /* Factors up to degree/order MAX_SIZE_FOR_ALF_FACTORS. Constructed only on
   * the first function call
   */
  static const dso::NormalizedLegendreFactors F;

  /* For the computations, we will need to use a larger degree (due to
   * gradient)
   */
  const int degree = max_degree;
  const int order = max_order;

  /* compute SH basis coefficients, Cnm->M and Snm->W */
  if (dso::gravity::sh_basis_cs_exterior(r / Re, degree + 2, order + 2, M, W)) {
    fprintf(stderr,
            "[ERROR] Failed computing spherical harmonics basis "
            "functions! (traceback: %s)\n",
            __func__);
    return 2;
  }

  /* acceleration and gradient in cartesian components */
  acc = Eigen::Matrix<double, 3, 1>::Zero();
  gradient = Eigen::Matrix<double, 3, 3>::Zero();

  /* start from smaller terms. Note that for degrees m=0,1, we are using
   * seperate loops
   */
  for (int m = order; m >= 2; --m) {
    for (int n = degree; n >= m; --n) {
      {
        /* acceleration */
        const double wm1 =
            std::sqrt(static_cast<double>(n - m + 1) * (n - m + 2));
        const double wm0 =
            std::sqrt(static_cast<double>(n - m + 1) * (n + m + 1));
        const double wp1 =
            std::sqrt(static_cast<double>(n + m + 1) * (n + m + 2));

        const double Cm1 = wm1 * M(n + 1, m - 1);
        const double Sm1 = wm1 * W(n + 1, m - 1);
        const double Cm0 = wm0 * M(n + 1, m);
        const double Sm0 = wm0 * W(n + 1, m);
        const double Cp1 = wp1 * M(n + 1, m + 1);
        const double Sp1 = wp1 * W(n + 1, m + 1);

        const double ax = cs.C(n, m) * (Cm1 - Cp1) + cs.S(n, m) * (Sm1 - Sp1);
        const double ay = cs.C(n, m) * (-Sm1 - Sp1) + cs.S(n, m) * (Cm1 + Cp1);
        const double az = cs.C(n, m) * (-2 * Cm0) + cs.S(n, m) * (-2 * Sm0);

        acc += Eigen::Matrix<double, 3, 1>(ax, ay, az) *
               std::sqrt((2e0 * n + 1e0) / (2e0 * n + 3e0));
      }
      {
        /* gradient */
        const double wm2 = std::sqrt(static_cast<double>(n - m + 1) *
                                     (n - m + 2) * (n - m + 3) * (n - m + 4)) *
                           ((m == 2) ? std::sqrt(2.0) : 1.0);
        const double wm1 = std::sqrt(static_cast<double>(n - m + 1) *
                                     (n - m + 2) * (n - m + 3) * (n + m + 1));
        const double wm0 = std::sqrt(static_cast<double>(n - m + 1) *
                                     (n - m + 2) * (n + m + 1) * (n + m + 2));
        const double wp1 = std::sqrt(static_cast<double>(n - m + 1) *
                                     (n + m + 1) * (n + m + 2) * (n + m + 3));
        const double wp2 = std::sqrt(static_cast<double>(n + m + 1) *
                                     (n + m + 2) * (n + m + 3) * (n + m + 4));

        const double Cm2 = wm2 * M(n + 2, m - 2);
        const double Sm2 = wm2 * W(n + 2, m - 2);
        const double Cm1 = wm1 * M(n + 2, m - 1);
        const double Sm1 = wm1 * W(n + 2, m - 1);
        const double Cm0 = wm0 * M(n + 2, m);
        const double Sm0 = wm0 * W(n + 2, m);
        const double Cp1 = wp1 * M(n + 2, m + 1);
        const double Sp1 = wp1 * W(n + 2, m + 1);
        const double Cp2 = wp2 * M(n + 2, m + 2);
        const double Sp2 = wp2 * W(n + 2, m + 2);

        const double gxx = cs.C(n, m) * (Cm2 - 2 * Cm0 + Cp2) +
                           cs.S(n, m) * (Sm2 - 2 * Sm0 + Sp2);
        const double gxy = cs.C(n, m) * (-Sm2 + Sp2) + cs.S(n, m) * (Cm2 - Cp2);
        const double gxz = cs.C(n, m) * (-2 * Cm1 + 2 * Cp1) +
                           cs.S(n, m) * (-2 * Sm1 + 2 * Sp1);
        const double gyy = cs.C(n, m) * (-Cm2 - 2 * Cm0 - Cp2) +
                           cs.S(n, m) * (-Sm2 - 2 * Sm0 - Sp2);
        const double gyz = cs.C(n, m) * (2 * Sm1 + 2 * Sp1) +
                           cs.S(n, m) * (-2 * Cm1 - 2 * Cp1);
        const double gzz = cs.C(n, m) * (4 * Cm0) + cs.S(n, m) * (4 * Sm0);

        gradient += Eigen::Matrix<double, 3, 3>{{gxx, gxy, gxz},
                                                {gxy, gyy, gyz},
                                                {gxz, gyz, gzz}} *
                    std::sqrt((2e0 * n + 1e0) / (2e0 * n + 5e0));
      }
    } /* loop over n */
  } /* loop over m */

  /* order m = 1 (begin summation from smaller terms) */
  for (int n = degree; n >= 1; --n) {
    const int m = 1;
    {
      /* acceleration
       * only difference with the generalized formula (aka for random n,m)
       * is in wm1
       */
      const double wm1 =
          std::sqrt(static_cast<double>(n - m + 1) * (n - m + 2)) *
          std::sqrt(2e0);
      const double wm0 =
          std::sqrt(static_cast<double>(n - m + 1) * (n + m + 1));
      const double wp1 =
          std::sqrt(static_cast<double>(n + m + 1) * (n + m + 2));

      const double Cm1 = wm1 * M(n + 1, m - 1);
      const double Sm1 = wm1 * W(n + 1, m - 1);
      const double Cm0 = wm0 * M(n + 1, m);
      const double Sm0 = wm0 * W(n + 1, m);
      const double Cp1 = wp1 * M(n + 1, m + 1);
      const double Sp1 = wp1 * W(n + 1, m + 1);

      const double ax = cs.C(n, m) * (Cm1 - Cp1) + cs.S(n, m) * (Sm1 - Sp1);
      const double ay = cs.C(n, m) * (-Sm1 - Sp1) + cs.S(n, m) * (Cm1 + Cp1);
      const double az = cs.C(n, m) * (-2e0 * Cm0) + cs.S(n, m) * (-2 * Sm0);

      acc += Eigen::Matrix<double, 3, 1>(ax, ay, az) *
             std::sqrt((2e0 * n + 1e0) / (2e0 * n + 3e0));
    }
    {
      /* gradient */
      const double wm1 = std::sqrt(static_cast<double>(n - m + 1) *
                                   (n - m + 2) * (n - m + 3) * (n + m + 1)) *
                         std::sqrt(2e0);
      const double wm0 = std::sqrt(static_cast<double>(n - m + 1) *
                                   (n - m + 2) * (n + m + 1) * (n + m + 2));
      const double wp1 = std::sqrt(static_cast<double>(n - m + 1) *
                                   (n + m + 1) * (n + m + 2) * (n + m + 3));
      const double wp2 = std::sqrt(static_cast<double>(n + m + 1) *
                                   (n + m + 2) * (n + m + 3) * (n + m + 4));

      const double Cm1 = wm1 * M(n + 2, m - 1);
      const double Sm1 = wm1 * W(n + 2, m - 1);
      const double Cm0 = wm0 * M(n + 2, m);
      const double Sm0 = wm0 * W(n + 2, m);
      const double Cp1 = wp1 * M(n + 2, m + 1);
      const double Sp1 = wp1 * W(n + 2, m + 1);
      const double Cp2 = wp2 * M(n + 2, m + 2);
      const double Sp2 = wp2 * W(n + 2, m + 2);

      const double gxx =
          cs.C(n, m) * (-3 * Cm0 + Cp2) + cs.S(n, m) * (-Sm0 + Sp2);
      const double gxy = cs.C(n, m) * (-Sm0 + Sp2) + cs.S(n, m) * (-Cm0 - Cp2);
      const double gxz =
          cs.C(n, m) * (-2 * Cm1 + 2 * Cp1) + cs.S(n, m) * (-2 * Sm1 + 2 * Sp1);
      const double gyy =
          cs.C(n, m) * (-Cm0 - Cp2) + cs.S(n, m) * (-3 * Sm0 - Sp2);
      const double gyz =
          cs.C(n, m) * (2 * Sp1) + cs.S(n, m) * (-2 * Cm1 - 2 * Cp1);
      const double gzz = cs.C(n, m) * (4 * Cm0) + cs.S(n, m) * (4 * Sm0);

      gradient += Eigen::Matrix<double, 3, 3>{{gxx, gxy, gxz},
                                              {gxy, gyy, gyz},
                                              {gxz, gyz, gzz}} *
                  std::sqrt((2e0 * n + 1e0) / (2e0 * n + 5e0));
    }
  } /* loop over all n's for m=1 */

  /* order m = 0 */
  for (int n = degree; n >= 0; --n) {
    [[maybe_unused]] const int m = 0;
    {
      /* acceleration */
      double wm0 = std::sqrt(static_cast<double>(n + 1) * (n + 1));
      double wp1 =
          std::sqrt(static_cast<double>(n + 1) * (n + 2)) / std::sqrt(2e0);

      double Cm0 = wm0 * M(n + 1, 0);
      double Cp1 = wp1 * M(n + 1, 1);
      double Sp1 = wp1 * W(n + 1, 1);

      const double ax = cs.C(n, 0) * (-2e0 * Cp1);
      const double ay = cs.C(n, 0) * (-2e0 * Sp1);
      const double az = cs.C(n, 0) * (-2e0 * Cm0);

      acc += Eigen::Matrix<double, 3, 1>(ax, ay, az) *
             std::sqrt((2e0 * n + 1.) / (2e0 * n + 3e0));
    }
    {
      /* gradient */
      const double wm0 =
          std::sqrt(static_cast<double>(n + 1) * (n + 2) * (n + 1) * (n + 2));
      const double wp1 =
          std::sqrt(static_cast<double>(n + 1) * (n + 1) * (n + 2) * (n + 3)) /
          std::sqrt(2e0);
      const double wp2 =
          std::sqrt(static_cast<double>(n + 1) * (n + 2) * (n + 3) * (n + 4)) /
          std::sqrt(2e0);

      const double Cm0 = wm0 * M(n + 2, 0);
      const double Cp1 = wp1 * M(n + 2, 1);
      const double Sp1 = wp1 * W(n + 2, 1);
      const double Cp2 = wp2 * M(n + 2, 2);
      const double Sp2 = wp2 * W(n + 2, 2);

      const double gxx = cs.C(n, 0) * (-2e0 * Cm0 + 2e0 * Cp2);
      const double gxy = cs.C(n, 0) * (2e0 * Sp2);
      const double gxz = cs.C(n, 0) * (4e0 * Cp1);
      const double gyy = cs.C(n, 0) * (-2e0 * Cm0 - 2e0 * Cp2);
      const double gyz = cs.C(n, 0) * (4e0 * Sp1);
      const double gzz = cs.C(n, 0) * (4e0 * Cm0);

      gradient += Eigen::Matrix<double, 3, 3>{{gxx, gxy, gxz},
                                              {gxy, gyy, gyz},
                                              {gxz, gyz, gzz}} *
                  std::sqrt((2e0 * n + 1e0) / (2e0 * n + 5e0));
    }
  } /* loop over all n's for m=0 */

  /* scale ... */
  gradient *= GM / (4e0 * Re * Re * Re);
  acc *= GM / (2e0 * Re * Re);

  return 0;
}
} /* unnamed namespace */

int dso::sh2gradient_cunningham(
    const dso::StokesCoeffs &cs, const Eigen::Matrix<double, 3, 1> &r,
    Eigen::Matrix<double, 3, 1> &acc, Eigen::Matrix<double, 3, 3> &gradient,
    int max_degree, int max_order, double Re, double GM,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> *W,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>
        *M) noexcept {

  /* set (if needed) maximum degree and order of expansion */
  if (max_degree < 0)
    max_degree = cs.max_degree();
  if (max_order < 0)
    max_order = cs.max_order();
  if (max_order > max_degree) {
    fprintf(stderr,
            "[ERROR] Invalid degree/order for spherical harmonics expansion! "
            "(traceback: %s)\n",
            __func__);
    return 1;
  }

  /* check computation degree and order w.r.t. the Stokes coeffs */
  if (max_degree > cs.max_degree()) {
    fprintf(stderr,
            "[ERROR] Requesting computing SH acceleration of degree %d, but "
            "Stokes coefficients are of size %dx%d (traceback: %s)\n",
            max_degree, cs.max_degree(), cs.max_order(), __func__);
    return 1;
  }
  if (max_order > cs.max_order()) {
    fprintf(stderr,
            "[ERROR] Requesting computing SH acceleration of order %d, but "
            "Stokes coefficients are of size %dx%d (traceback: %s)\n",
            max_order, cs.max_degree(), cs.max_order(), __func__);
    return 1;
  }

  /* set (if needed) geometric parameters of expansion */
  if (Re < 0)
    Re = cs.Re();
  if (GM < 0)
    GM = cs.GM();

  /* allocate (if needed) scratch space */
  int delete_mem_pool[] = {0, 0};
  if (!W) {
    W = new dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>(
        max_degree + 2, max_degree + 2);
    delete_mem_pool[0] = 1;
  }
  if (!M) {
    M = new dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>(
        max_degree + 2, max_degree + 2);
    delete_mem_pool[1] = 1;
  }

  /* check scratch space */
  if ((W->rows() < max_degree + 2) || (W->cols() < max_degree + 2) ||
      (M->rows() < max_degree + 2) || (M->cols() < max_degree + 2)) {
    fprintf(stderr,
            "[ERROR] Invalid size of mem pool for spherical harmonics "
            "expansion! (traceback: %s)\n",
            __func__);
    return 1;
  }

  /* call core function */
  int status = sh2gradient_cunningham_impl(cs, r, max_degree, max_order, Re, GM,
                                           acc, gradient, *W, *M);

  /* do we need to free memory ? */
  if (delete_mem_pool[0])
    delete W;
  if (delete_mem_pool[1])
    delete M;

  /* return */
  return status;
}
