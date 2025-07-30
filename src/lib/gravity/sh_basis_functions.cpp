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