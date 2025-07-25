#include "gravity.hpp"
#include <array>
#include <cassert>
#include <cmath>

namespace {

int sh2deformation_impl(
    const dso::StokesCoeffs &cs, const Eigen::Matrix<double, 3, 1> &r,
    int max_degree, int max_order, double Re, double GM,
    Eigen::Matrix<double, 3, 1> &acc, Eigen::Matrix<double, 3, 3> &gradient,
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

  /* Factors up to degree/order MAX_SIZE_FOR_ALF_FACTORS. Constructed only on
   * the first function call
   */
  static const dso::NormalizedLegendreFactors F;

  /* For the computations, we will need to use a larger degree (due to
   * gradient)
   */
  const int degree = max_degree;
  const int order = max_order;

  /* nullify scratch space */
  W.fill_with(0e0);
  M.fill_with(0e0);

  {
    /* (kinda) Associated Legendre Polynomials M and W
     * note that since we are going to compute derivatives (of the gravity
     * potential), we will compute M and W up to degree n+2,m+2
     */
    const Eigen::Matrix<double, 3, 1> rs = r / Re;
    const double rr = std::pow(1e0 / rs.norm(), 2);
    const double x = rs.x() * rr;
    const double y = rs.y() * rr;
    const double z = rs.z() * rr;

    /* start ALF iteration (can use scaling according to Holmes et al, 2002) */
    M(0, 0) = 1e0 / rs.norm();
    // M(0, 0) = 1e280 / rs.norm();

    /* first fill m=0 terms; note that W(n,0) = 0 (already set) */
    M(1, 0) = std::sqrt(3e0) * z * M(0, 0);
    for (int n = 2; n <= degree + 2; n++) {
      M(n, 0) = F.f1(n, 0) * z * M(n - 1, 0) + F.f2(n, 0) * rr * M(n - 2, 0);
    }

    /* fill all elements for order m >= 1 */
    for (int m = 1; m < order + 2; m++) {
      /* M(m,m) and W(m,m) aka, diagonal */
      M(m, m) = F.f1(m, m) * (x * M(m - 1, m - 1) - y * W(m - 1, m - 1));
      W(m, m) = F.f1(m, m) * (y * M(m - 1, m - 1) + x * W(m - 1, m - 1));

      /* if n=m+1 , we do not have a M(n-2,...) aka sub-diagonal term */
      M(m + 1, m) = F.f1(m + 1, m) * z * M(m, m);
      W(m + 1, m) = F.f1(m + 1, m) * z * W(m, m);

      /* go on .... */
      for (int n = m + 2; n <= degree + 2; n++) {
        M(n, m) = F.f1(n, m) * z * M(n - 1, m) + F.f2(n, m) * rr * M(n - 2, m);
        W(n, m) = F.f1(n, m) * z * W(n - 1, m) + F.f2(n, m) * rr * W(n - 2, m);
      }
    }

    {
      /* well, we've left the lst term uncomputed */
      const int m = order + 2;
      M(m, m) = F.f1(m, m) * (x * M(m - 1, m - 1) - y * W(m - 1, m - 1));
      W(m, m) = F.f1(m, m) * (y * M(m - 1, m - 1) + x * W(m - 1, m - 1));
    }

    // M.multiply(1e-280);
    // W.multiply(1e-280);
  } /* end computing ALF factors M and W */

  /* acceleration and gradient in cartesian components */
  acc = Eigen::Matrix<double, 3, 1>::Zero();
  gradient = Eigen::Matrix<double, 3, 3>::Zero();

  /* Load Love numbers */
  const dso::LoadLoveNumbers &love = dso::groopsLoadLoveNumbers;

  for (int n = 0; n <= max_degree; n++) {
    Eigen::Vector3d am0, am1, amn;
    am0 << 0e0, 0e0, 0e0;
    am1 << 0e0, 0e0, 0e0;
    amn << 0e0, 0e0, 0e0;
    double Vm0 = 0e0, Vm1 = 0e0, Vmn = 0e0;

    /* order m=0 */
    {
      const int m = 0;
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

      am0 += Eigen::Matrix<double, 3, 1>(ax, ay, az) *
             std::sqrt((2e0 * n + 1.) / (2e0 * n + 3e0));

      /* potential */
      Vm0 = CS.C(n, m) * cs.C(n, m);
    }

    /* potential */
    Vm0 = CS.C(n, m) * cs.C(n, m);
  }

  /* order m=1 */
  {
    /* only difference with the generalized formula (aka for random n,m) is in
     * wm1
     */
    const int m = 1;

    /* potential */
    Vm1 = CS.C(n, m) * cs.C(n, m) + CS.S(n, m) * cs.S(n, m);

    /* acceleration i.e grad(V) */
    const double wm1 = std::sqrt(static_cast<double>(n - m + 1) * (n - m + 2)) *
                       std::sqrt(2e0);
    const double wm0 = std::sqrt(static_cast<double>(n - m + 1) * (n + m + 1));
    const double wp1 = std::sqrt(static_cast<double>(n + m + 1) * (n + m + 2));

    const double Cm1 = wm1 * CS.C(n + 1, m - 1);
    const double Sm1 = wm1 * CS.S(n + 1, m - 1);
    const double Cm0 = wm0 * CS.C(n + 1, m);
    const double Sm0 = wm0 * CS.S(n + 1, m);
    const double Cp1 = wp1 * CS.C(n + 1, m + 1);
    const double Sp1 = wp1 * CS.S(n + 1, m + 1);

    const double ax = cs.C(n, m) * (Cm1 - Cp1) + cs.S(n, m) * (Sm1 - Sp1);
    const double ay = cs.C(n, m) * (-Sm1 - Sp1) + cs.S(n, m) * (Cm1 + Cp1);
    const double az = cs.C(n, m) * (-2e0 * Cm0) + cs.S(n, m) * (-2e0 * Sm0);

    am1 = Eigen::Vector3d(ax, ay, az);
  }

  /* all other orders, m=2,...,max_order */
  {
    for (int m = 2; m <= std::min(n, max_order); m++) {
      /* potential */
      Vmn += CS.C(n, m) * cs.C(n, m) + CS.S(n, m) * cs.S(n, m);

      /* acceleration i.e grad(V) */
      const double wm1 =
          std::sqrt(static_cast<double>(n - m + 1) * (n - m + 2));
      const double wm0 =
          std::sqrt(static_cast<double>(n - m + 1) * (n + m + 1));
      const double wp1 =
          std::sqrt(static_cast<double>(n + m + 1) * (n + m + 2));

      const double Cm1 = wm1 * CS.C(n + 1, m - 1);
      const double Sm1 = wm1 * CS.S(n + 1, m - 1);
      const double Cm0 = wm0 * CS.C(n + 1, m);
      const double Sm0 = wm0 * CS.S(n + 1, m);
      const double Cp1 = wp1 * CS.C(n + 1, m + 1);
      const double Sp1 = wp1 * CS.S(n + 1, m + 1);

      const double ax = cs.C(n, m) * (Cm1 - Cp1) + cs.S(n, m) * (Sm1 - Sp1);
      const double ay = cs.C(n, m) * (-Sm1 - Sp1) + cs.S(n, m) * (Cm1 + Cp1);
      const double az = cs.C(n, m) * (-2 * Cm0) + cs.S(n, m) * (-2 * Sm0);

      amn += Eigen::Vector3d(ax, ay, az);
    }
  }
