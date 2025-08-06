#include "gravity.hpp"
#include <array>
#include <cassert>
#include <cmath>

int dso::gravity::sh_deformation(
    const Eigen::Vector3d &point, const dso::StokesCoeffs &cs,
    Eigen::Vector3d &dr, Eigen::Vector3d &gravity, double &potential,
    int max_degree, int max_order,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> &M,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>
        &W) noexcept {

  /* compute basis functions, cnm and snm at given point; store at M, W */
  if (dso::gravity::sh_basis_cs_exterior(point / cs.Re(), max_degree + 1,
                                         max_order + 1, M, W)) {
    fprintf(stderr,
            "[ERROR] Failed computing spherical harmonics basis "
            "functions! (traceback: %s)\n",
            __func__);
    return 2;
  }

  /* displacement vector */
  dr << 0e0, 0e0, 0e0;

  /* (total) gravity at point */
  gravity << 0e0, 0e0, 0e0;

  /* unit vector at point */
  const Eigen::Vector3d r = point.normalized();

  /* initialize potential */
  potential = 0e0;

  /* Load Love numbers */
  const dso::LoadLoveNumbers &love = dso::groopsLoadLoveNumbers;

  for (int n = 0; n <= max_degree; n++) {
    double Vn = 0e0;
    Eigen::Vector3d gradVn = Eigen::Vector3d::Zero();
    /* order m=0 */
    {
      [[maybe_unused]] const int m = 0;

      /* potential */
      Vn = M(n, 0) * cs.C(n, 0);

      /* acceleration i.e grad(V) */
      const double wm0 = std::sqrt(static_cast<double>(n + 1) * (n + 1));
      const double wp1 =
          std::sqrt(static_cast<double>(n + 1) * (n + 2)) / std::sqrt(2e0);

      const double Cm0 = wm0 * M(n + 1, 0);
      const double Cp1 = wp1 * M(n + 1, 1);
      const double Sp1 = wp1 * W(n + 1, 1);

      gradVn.x() = cs.C(n, 0) * (-2e0 * Cp1);
      gradVn.y() = cs.C(n, 0) * (-2e0 * Sp1);
      gradVn.z() = cs.C(n, 0) * (-2e0 * Cm0);
    }

    /* all other orders, m=1,...,max_order */
    {
      for (int m = 1; m <= std::min(n, max_order); m++) {
        /* potential */
        Vn += M(n, m) * cs.C(n, m) + W(n, m) * cs.S(n, m);

        /* acceleration i.e grad(V) */
        const double wm1 =
            std::sqrt(static_cast<double>(n - m + 1) * (n - m + 2)) *
            ((m == 1) ? std::sqrt(2.0) : 1.0);
        const double wm0 =
            std::sqrt(static_cast<double>(n - m + 1) * (n + m + 1));
        const double wp1 =
            std::sqrt(static_cast<double>(n + m + 1) * (n + m + 2));

        const double Cm1 = wm1 * M(n + 1, m - 1);
        const double Cm0 = wm0 * M(n + 1, m);
        const double Cp1 = wp1 * M(n + 1, m + 1);
        const double Sm1 = wm1 * W(n + 1, m - 1);
        const double Sm0 = wm0 * W(n + 1, m);
        const double Sp1 = wp1 * W(n + 1, m + 1);

        gradVn.x() += cs.C(n, m) * (Cm1 - Cp1) + cs.S(n, m) * (Sm1 - Sp1);
        gradVn.y() += cs.C(n, m) * (-Sm1 - Sp1) + cs.S(n, m) * (Cm1 + Cp1);
        gradVn.z() += cs.C(n, m) * (-2 * Cm0) + cs.S(n, m) * (-2 * Sm0);
      }
    } /* done with m's ... */

    gradVn *= std::sqrt((2. * n + 1.) / (2. * n + 3.)) / (2e0 * cs.Re());
    /* total gravity vec. (unscalled) */
    gravity += gradVn;
    /* total potential (scalar, unscalled) */
    potential += Vn;
    /* displacement vector */
    dr += (love.h[n] * Vn) * r + love.l[n] * (gradVn - gradVn.dot(r) * r);
  } /* loop through n's */

  /* total gravity at point */
  gravity *= cs.GM() / cs.Re();

  /* total potential at point */
  potential *= cs.GM() / cs.Re();

  /* scale displacement */
  dr *= (1.e0 / gravity.norm());

  return 0;
}

int dso::sh_deformation(
    const Eigen::Vector3d &rsta, const dso::StokesCoeffs &cs,
    Eigen::Vector3d &gravity, double &potential, Eigen::Vector3d &dr,
    int max_degree, int max_order,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> *M,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>
        *W) noexcept {

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

  /* allocate (if needed) scratch space */
  int delete_mem_pool[] = {0, 0};
  if (!M) {
    M = new dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>(
        max_degree + 2, max_degree + 2);
    delete_mem_pool[0] = 1;
  }
  if (!W) {
    W = new dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>(
        max_degree + 2, max_degree + 2);
    delete_mem_pool[1] = 1;
  }

  /* The coefficients of C and S (i.e. M, W) should span the range
   * [0, max_degree+1] and [0, max_order+1], but note that here we are
   * allocating matrices, hence their size should be (max_degree+2,
   * max_order+2)
   */
  if (((M->rows() < max_degree + 2) || (W->rows() < max_degree + 2)) ||
      ((M->cols() < max_order + 2) || (W->cols() < max_order + 2))) {
    fprintf(stderr,
            "[ERROR] Invalid size(s) for (scratch) coefficients arrays C/S; "
            "requested degree, order is (%d,%d), C=(%d,%d) and S=(%d,%d) "
            "(traceback: %s)\n",
            max_degree, max_order, M->rows(), M->cols(), W->rows(), W->cols(),
            __func__);
    return 1;
  }

  /* compute deformation */
  int error = 0;
  if ((error = dso::gravity::sh_deformation(rsta, cs, dr, gravity, potential,
                                            max_degree, max_order, *M, *W))) {
    fprintf(stderr,
            "[ERROR] Failed computing (surface) deformation from SH "
            "expansion! (traceback: %s)\n",
            __func__);
  }

  /* do we need to free memory ? */
  if (delete_mem_pool[0])
    delete M;
  if (delete_mem_pool[1])
    delete W;

  return error;
}