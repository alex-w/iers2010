#include "geodesy/transformations.hpp"
#include "gravity.hpp"
#include <array>
#include <cassert>
#include <cmath>

/** @brief Compute Yn coefficients of the potential expansion:
 * Yn = GM/R * Σ(cnm*Cnm + snm*Snm)
 * where Cnm, Snm are the model coefficients and cnm, snm are the SH basis
 * functions.
 *
 * @param[in] point Point of computation in cartesian ECEF, [m].
 * @param[in] CS    Cnm and Snm (stokes) coefficients of the model. The degree
 * and order of the model must be less or equal to max_degree and max_order.
 * @param[out] y    The computed Yn coefficients, one per n (i.e. max_degree).
 * @param[out] cs   Scratch StokesCoeffs to perform internal computations. It
 * will be resized to hold enough elements for the computation, so it can be
 * of any size. After the call, it's size will be at least as large as
 * (max_degree, max_order), so that it can be used in subsequent calls (if
 * needed) and avoid (re)allocation.
 * @param[in] max_degree Max degree of SH expansion. max_degree should be <=
 * CS.max_degree(). If a negative number is give, then CS.max_degree() will be
 * used.
 * @param[in] max_order Max order of SH expansion. max_degree should be <=
 * CS.max_order(). If a negative number is given, then CS.max_order() will be
 * used.
 * @return Anything other than zero denotes an error.
 */
int dso::gravity::ynm(const Eigen::Vector3d &point, const dso::StokesCoeffs &CS,
                      std::vector<double> &y, dso::StokesCoeffs &cs,
                      int max_degree, int max_order) noexcept {
  /* set/check max (n,m) */
  if (max_degree < 0)
    max_degree = CS.max_degree();
  if (max_order < 0)
    max_order = CS.max_order();

  if ((max_degree > CS.max_degree()) || (max_order >= CS.max_order())) {
    fprintf(stderr,
            "[ERROR] Invalid degree/order for Ynm computation; mismatch with "
            "given Cnm, Snm model coeffs (traceback: %s)\n",
            __func__);
    return 1;
  }

  /* resize cs (to be computed). */
  cs.resize(max_degree, max_degree);
  [[maybe_unused]] const Eigen::Vector3d rsta = point;

  /* compute basis functions, cnm and snm at given point [a]*/
  // if (sh_basis_cs_exterior(point / CS.Re(), cs, max_degree, max_order)) {
  //   fprintf(stderr,
  //           "[ERROR] Failed computing spherical harmonics basis "
  //           "functions! (traceback: %s)\n",
  //           __func__);
  //   return 2;
  // }

  /* Yn = GM/R * sum (cnm*Cnm + snm*Snm) */
  y.reserve(max_degree);
  for (int n = 0; n <= max_degree; n++) {
    double yn = cs.C(n, 0) * CS.C(n, 0);
    for (int m = 1; m <= n; m++) {
      yn += cs.C(n, m) * CS.C(n, m) + cs.S(n, m) * CS.S(n, m);
    }
    y.emplace_back(yn * (CS.GM() / CS.Re()));
  }

  return 0;
  /* Note [a], What's up with the scaling ?
   * sh_basis_cs_exterior computes cnm = (1/r^{n+1}) * Pnm * cos(ml)
   * and snm = (1/r^{n+1}) * Pnm * sin(ml)
   * but the potential reads:
   * V = (GM/r) Σ_{n} (R/n)^{n} Σ_{m} [Cnm cos(ml) Pnm + Snm sin(ml) Pnm]
   * = GM Σ_{n} R^{n} Σ_{m} [(1/r)^{n+1} * Cnm * cos(ml) * Pnm
   * + (1/r)^{n+1} * Snm * Pnm * sin(ml) ]
   * = GM Σ_{n} R^{n} Σ_{m} [Cnm * cnm + Snm * snm] (Eq. 1a)
   *
   * so, giving (point / R) in sh_basis_cs_exterior means that we will normalize
   * the radius as r' = r / R. In turn, the basis functions (cnm, snm) will be
   * computed with a scaling factor:
   *  (1/r')^{n+1} = (1/(r/R))^{n+1} = (R/r)^{n+1}.
   * In this case, Eq. 1a, becomes V = GM/R Σ_{m} [Cnm * cnm + Snm * snm]
   * which is what we actually compute (per degree n).
   */
}

int dso::gravity::sh_deformation(const Eigen::Vector3d &point,
                                 const dso::StokesCoeffs &cs,
                                 Eigen::Vector3d &dr, Eigen::Vector3d &gravity,
                                 double &potential, int max_degree,
                                 int max_order) noexcept {

  /* set/check max (n,m) */
  if (max_degree < 0)
    max_degree = cs.max_degree();
  if (max_order < 0)
    max_order = cs.max_order();

  if (((max_degree > cs.max_degree()) || (max_order > cs.max_order())) ||
      (max_degree < max_order)) {
    fprintf(stderr,
            "[ERROR] Invalid degree/order for deformation computation; "
            "mismatch with "
            "given Cnm, Snm model coeffs (traceback: %s)\n",
            __func__);
    fprintf(stderr,
            "[ERROR] (contd) (N,M)=(%d,%d), Model=(%d,%d), Scratch=(%d,%d) "
            "(traceback: %s)\n",
            max_degree, max_order, cs.max_degree(), cs.max_order(),
            cs.max_degree(), cs.max_order(), __func__);
    return 1;
  }

  /* We need to allocate scratch space for SH coefficients. The coefficients
   * should span the range [0, max_degree+1] and [0, max_order+1], but note that
   * here we are allocating matrices, hence their size should be (max_degree+2,
   * max_order+2)
   */
  dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> M(
      max_degree + 2, max_degree + 2);
  dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> W(
      max_degree + 2, max_degree + 2);

  /* compute basis functions, cnm and snm at given point [a] */
  if (sh_basis_cs_exterior(point / cs.Re(), max_degree + 1, max_order + 1, M,
                           W)) {
    fprintf(stderr,
            "[ERROR] Failed computing spherical harmonics basis "
            "functions! (traceback: %s)\n",
            __func__);
    return 2;
  }

  /* displacement vector */
  dr << 0e0, 0e0, 0e0;

  /* (total) gravity at point */
  Eigen::Vector3d grav, g;
  grav << 0e0, 0e0, 0e0;

  /* unit vector at point */
  const Eigen::Vector3d r = point.normalized();

  /* initialize potential */
  potential = 0e0;

  /* Load Love numbers */
  const dso::LoadLoveNumbers &love = dso::groopsLoadLoveNumbers;

  for (int n = 0; n <= max_degree; n++) {
    /* order m=0 */
    {
      const int m = 0;

      /* potential */
      potential = M(n, m) * cs.C(n, m);

      /* acceleration i.e grad(V) */
      const double wm0 = std::sqrt(static_cast<double>(n + 1) * (n + 1));
      const double wp1 =
          std::sqrt(static_cast<double>(n + 1) * (n + 2)) / std::sqrt(2e0);

      const double Cm0 = wm0 * M(n + 1, 0);
      const double Cp1 = wp1 * M(n + 1, 1);
      const double Sp1 = wp1 * W(n + 1, 1);

      g.x() = cs.C(n, 0) * (-2e0 * Cp1);
      g.y() = cs.C(n, 0) * (-2e0 * Sp1);
      g.z() = cs.C(n, 0) * (-2e0 * Cm0);
    }

    /* all other orders, m=1,...,max_order */
    {
      for (int m = 1; m <= std::min(n, max_order); m++) {
        /* potential */
        potential += M(n, m) * cs.C(n, m) + W(n, m) * cs.S(n, m);

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

        g.x() += cs.C(n, m) * (Cm1 - Cp1) + cs.S(n, m) * (Sm1 - Sp1);
        g.y() += cs.C(n, m) * (-Sm1 - Sp1) + cs.S(n, m) * (Cm1 + Cp1);
        g.z() += cs.C(n, m) * (-2 * Cm0) + cs.S(n, m) * (-2 * Sm0);
      }
    } /* done with m's ... */
    grav += std::sqrt((2. * n + 1.) / (2. * n + 3.)) * g;
    dr += (love.h[n] * potential) * r + love.l[n] * (grav - grav.dot(r) * r);
  } /* loop through n's */

  /* scale with gravity at point */
  dr *= (1e0 / grav.norm());

  /* assign for output */
  grav *= cs.GM() / (2 * cs.Re() * cs.Re());
  potential *= (cs.GM() / cs.Re());

  gravity = grav;
  return 0;
}
