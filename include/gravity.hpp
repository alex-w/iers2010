/** @file
 * Computation of gravity acceleration using a geopotential model.
 */

#ifndef __DSO_GEOPOTENTIAL_ACCELERATAION_HPP__
#define __DSO_GEOPOTENTIAL_ACCELERATAION_HPP__

#include "eigen3/Eigen/Eigen"
#include "geodesy/transformations.hpp"
#include "load_love_numbers.hpp"
#include "stokes_coefficients.hpp"

namespace dso {

struct NormalizedLegendreFactors {
  /* Max size for ALF factors; if degree is more than this (-2), then it must
   * be augmented. For now, OK
   */
  static constexpr const int MAX_SIZE_FOR_ALF_FACTORS = 201;
  dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> f1;
  dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> f2;
  std::array<double, MAX_SIZE_FOR_ALF_FACTORS> f3;
  NormalizedLegendreFactors() noexcept;
}; /* struct NormalizedLegendreFactors */

/** Acceleration due to point mass at r_cb on a mass at r.
 *
 * Compute the aceleration induced on a body at the position vector r,
 * caused by a point mass with gravitational constant GM at position rcb.
 * This is normally used to compute third-body acceleration on a
 * satellite, induced by e.g. the Sun or Moon.
 *
 * @param[in] r Geocentric position of attracted mass, i.e. satellite in
 * [m]. The vector holds Cartesian components, i.e. r=(x,y,z).
 * @param[in] rcb Geocentric position vector of the attracting body, i.e.
 * the 'third body' (e.g. Sun or Moon) in [m].
 * @param[in] GMcb gravitational constant of the 'third body', i.e. G *
 * M_cb, in [m^3/ sec^2]
 * @return Acceleration induced on the mass, in Cartesian components, in
 *         units of [m/s^2]. I.e. a = (a_x, a_y, a_z)
 *
 * @note Instead of using [m] as input units, [km] can also be used, as
 * long as it is used consistently for ALL inputs (r, rcb and GM). in this
 *       case, the resulting acceleration will be given in units of
 * [km/s^2].
 */
Eigen::Matrix<double, 3, 1>
point_mass_acceleration(const Eigen::Matrix<double, 3, 1> &r,
                        const Eigen::Matrix<double, 3, 1> &rcb,
                        double GMcb) noexcept;

/** Acceleration due to point mass at r_cb on a mass at r.
 *
 * Same as above, only in this case we also compute and return the Jacobian
 * matrix da/dr, i.e.
 *
 *     | dax/dx dax/dy dax/dz |
 * J = | day/dx day/dy day/dz |
 *     | daz/dx daz/dy daz/dz |
 *
 * @param[in] r Geocentric position of attracted mass, i.e. satellite in [m].
 *              The vector holds Cartesian components, i.e. r=(x,y,z).
 * @param[in] rcb Geocentric position vector of the attracting body, i.e. the
 *              'third body' (e.g. Sun or Moon) in [m].
 * @param[in] GMcb gravitational constant of the 'third body', i.e. G * M_cb,
 *              in [m^3/ sec^2]
 * @param[out] jacobian The Jacobian 3x3 matrix da/dr
 * @return Acceleration induced on the mass, in Cartesian components, in
 *         units of [m/s^2]. I.e. a = (a_x, a_y, a_z)
 *
 * @note Instead of using [m] as input units, [km] can also be used, as long
 *       as it is used consistently for ALL inputs (r, rcb and GM). in this
 *       case, the resulting acceleration will be given in units of [km/s^2].
 */
Eigen::Matrix<double, 3, 1>
point_mass_acceleration(const Eigen::Matrix<double, 3, 1> &r,
                        const Eigen::Matrix<double, 3, 1> &rcb, double GMcb,
                        Eigen::Matrix<double, 3, 3> &jacobian) noexcept;

/** @brief Compute normalised associated Legendre functions Pnm
 *
 * The algorithm employed here to perform the computations is the
 * "forward recursions" method, see Holmes et al, 2002.
 *
 * @param[in] theta Geocentric latitude for expansion in [rad].
 * @param[in] max_degree Max degree for expansion. i.e. n in Pnm, inclusive
 * @param[in] max_order  Max order for expansion, i.e. m in Pnm, inclusive
 * @param[out] Pnm       Matrix to store the computed Pnm values; must be
 *                       at least large enough to hold Pnm's for
 *                       n=[0,max_degree] and m=[0,max_order].
 * @return Always zero.
 *
 * Holmes, S., Featherstone, W. A unified approach to the Clenshaw summation
 * and the recursive computation of very high degree and order normalised
 * associated Legendre functions. Journal of Geodesy 76, 279–299 (2002).
 * https://doi.org/10.1007/s00190-002-0216-2
 */
int normalised_associated_legendre_functions(
    double theta, int max_degree, int max_order,
    CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> &Pnm) noexcept;

/** @brief Compute normalised associated Legendre functions Pnm and its
 *         first derivative.
 *
 * The algorithm employed here to perform the computations is the
 * "forward recursions" method, see Holmes et al, 2002.
 *
 * @param[in] theta Geocentric latitude for expansion in [rad].
 * @param[in] max_degree Max degree for expansion. i.e. n in Pnm, inclusive
 * @param[in] max_order  Max order for expansion, i.e. m in Pnm, inclusive
 * @param[out] Pnm       Matrix to store the computed Pnm values; must be
 *                       at least large enough to hold Pnm's for
 *                       n=[0,max_degree] and m=[0,max_order].
 * @return Always zero.
 *
 * Holmes, S., Featherstone, W. A unified approach to the Clenshaw summation
 * and the recursive computation of very high degree and order normalised
 * associated Legendre functions. Journal of Geodesy 76, 279–299 (2002).
 * https://doi.org/10.1007/s00190-002-0216-2
 */
int normalised_associated_legendre_functions(
    double theta, int max_degree, int max_order,
    CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> &Pnm,
    CoeffMatrix2D<MatrixStorageType::LwTriangularColWise> &dPnm) noexcept;

/** Spherical harmonics of Earth's gravity potential to acceleration and
 *  gradient using the algorithm due to Cunningham. The acceleration and
 *  gradient are computed in Cartesian components, i.e.
 *
 *  acceleration = (ax, ay, az), and
 *
 *             | dax/dx dax/dy dax/dz |
 *  gradient = | day/dx day/dy day/dz |
 *             | daz/dx daz/dy daz/dz |
 *
 * @param[in] cs Normalized Stokes coefficients of spherical harmonics
 * @param[in] r  Position vector of satellite (aka point of computation) in
 *               an ECEF frame (e.g. ITRF)
 * @param[in] max_degree Max degree of spherical harmonics expansion. If set
 *               to a negative number, the degree of expansion will be derived
 *               from the cs input parameter, i.e. cs.max_degree()
 * @param[in] max_order  Max order of spherical harmonics expansion. If set
 *               to a negative number, the order of expansion will be derived
 *               from the cs input parameter, i.e. cs.max_order()
 * @param[in] Re Equatorial radius of the Earth in [m]. If set to a negative
 *               number, then the cs.Re() method will be used to get it.
 * @param[in] GM Gravitational constant of Earth. If set to a negative
 *               number, then the cs.GM() method will be used to get it.
 * @param[out] acc Acceleration in cartesian components in [m/s^2]
 * @param[out] gradient Gradient of acceleration in cartesian components
 * @param[in] W   A convinient storage space, as Column-Wise Lower Triangular
 *                matrix off dimensions at least (max_degree+2, max_degree+2).
 *                If not given, then the function will allocate and free the
 *                required memmory.
 * @param[in] M   A convinient storage space, as Column-Wise Lower Triangular
 *                matrix off dimensions at least (max_degree+2, max_degree+2)
 *                If not given, then the function will allocate and free the
 *                required memmory.
 * @return        Anything other than zero denotes an error.
 */
int sh2gradient_cunningham(
    const dso::StokesCoeffs &cs, const Eigen::Matrix<double, 3, 1> &r,
    Eigen::Matrix<double, 3, 1> &acc, Eigen::Matrix<double, 3, 3> &gradient,
    int max_degree = -1, int max_order = -1, double Re = -1, double GM = -1,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> *W =
        nullptr,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> *M =
        nullptr) noexcept;

/* ----------------------------------------------------------------- */
/* ---------------------------- NEW STUFF -------------------------- */
/* ----------------------------------------------------------------- */
namespace gravity {
int sh_deformation(const Eigen::Vector3d &point, const dso::StokesCoeffs &CS,
                   dso::StokesCoeffs &cs, Eigen::Vector3d &dr,
                   Eigen::Vector3d &gravity, double &potential,
                   int max_degree = -1, int max_order = -1) noexcept;
int ynm(const Eigen::Vector3d &point, const dso::StokesCoeffs &CS,
        std::vector<double> &y, dso::StokesCoeffs &cs, int max_degree = -1,
        int max_order = -1) noexcept;
int sh_basis_cs_exterior(const Eigen::Vector3d &point, dso::StokesCoeffs &cs,
                         int max_degree, int max_order) noexcept;
} /* namespace gravity */

// template <typename C = CartesianCrd>
// [[nodiscard]]
// int sh_basis_cs_exterior(const C &rsta, dso::StokesCoeffs &cs,
inline int sh_basis_cs_exterior(Eigen::Vector3d &rsta, dso::StokesCoeffs &cs,
                                int max_degree = -1,
                                int max_order = -1) noexcept {
  // static_assert(CoordinateTypeTraits<C>::isCartesian);

  /* set (if needed) maximum degree and order of expansion */
  if (max_degree < 0)
    max_degree = cs.max_degree();
  if (max_order < 0)
    max_order = cs.max_order();
  if ((max_order > max_degree) || (max_degree < 0 || max_order < 0)) {
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

  return gravity::sh_basis_cs_exterior(rsta, cs, max_degree, max_order);
}

// template <typename C = CartesianCrd>
//[[no_discard]]
// dso::StokesCoeffs sh_basis_cs_exterior(const C &rsta, int max_degree,
//                                        int max_order) {
//   static_assert(dso::CoordinateTypeTraits<C>::isCartesian);
//
//   /* set (if needed) maximum degree and order of expansion */
//   if ((max_order > max_degree) || (max_degree < 0 || max_order < 0)) {
//     fprintf(stderr,
//             "[ERROR] Invalid degree/order for spherical harmonics expansion!
//             "
//             "(traceback: %s)\n",
//             __func__);
//     throw std::runtime_error(err_msg);
//   }
//
//   dso::StokesCoeffs cs(max_degree, max_order);
//   if (gravity::sh_basis_cs_exterior(rsta.mv, cs, max_degree, max_order)) {
//     throw std::runtime_error(err_msg);
//   }
//
//   /* return the Stokes coeffs */
//   return cs;
// }
inline int sh_deformation(const Eigen::Vector3d &rsta,
                          const dso::StokesCoeffs &cs, int max_degree,
                          int max_order, Eigen::Vector3d &gravity,
                          double &potential, Eigen::Vector3d &dr) noexcept {

  /* set (if needed) maximum degree and order of expansion */
  if (max_degree < 0)
    max_degree = cs.max_degree();
  if (max_order < 0)
    max_order = cs.max_order();
  if ((max_order > max_degree) || (max_degree < 0 || max_order < 0)) {
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

  /* compute deformation */
  dso::StokesCoeffs scratch(max_degree, max_order);
  if (gravity::sh_deformation(rsta, cs, scratch, dr, gravity, potential,
                              max_degree, max_order)) {
    fprintf(stderr,
            "[ERROR] Failed computing (surface) deformation from SH "
            "expansion! (traceback: %s)\n",
            __func__);
    return 5;
  }

  return 0;
}

} /* namespace dso */

#endif
