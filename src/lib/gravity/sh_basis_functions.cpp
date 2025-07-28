#include "geodesy/transformations.hpp"
#include "gravity.hpp"
#include <array>
#include <cassert>
#include <cmath>

/** @brief  Compute the spherical harmonic basis functions Cnm, Snm
 *
 * This function computes the spherical harmonic basis functions Cnm, Snm​ up
 * to a specified degree n, evaluated at a 3D point in space.
 *
 * These functions are the real-valued solid spherical harmonics, commonly
 * used in geodesy and gravity field modeling.
 *
 * Computes the real solid spherical harmonics (4π normalized):
 * Cnm = (1/r**(n+1)) cos(mλ) Pnm(cosθ)
 * Snm = (1/r**(n+1)) cos(mλ) Pnm(cosθ)
 *
 * At the end of the function
 * cs.C(n,m) holds the real-valued cosine basis Cnm
 * cs.S(n,m) holds the real-valued sine basis Snm​
 * These can then be multiplied by the corresponding cnm, snm gravity field
 * coefficients to compute potential, acceleration, etc.
 */
int dso::gravity::sh_basis_cs_exterior(
    const Eigen::Vector3d &rsta, int max_degree, int max_order,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> &C,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>
        &S) noexcept {

  if (((C.rows() < max_degree) || (C.cols() < max_degree)) ||
      ((S.rows() < max_degree) || (S.cols() < max_degree))) {
    fprintf(stderr,
            "[ERROR] Invalid C/S size(s) for computing SH coefficients "
            "(traceback: %s)\n",
            __func__);
    return 1;
  }

  if (max_degree >
      dso::NormalizedLegendreFactors::MAX_SIZE_FOR_ALF_FACTORS - 3) {
    fprintf(stderr,
            "[ERROR] (Static) Size for NormalizedLegendreFactors must be "
            "augmented to perform computation (traceback: %s)\n",
            __func__);
    assert(false);
    return 1;
  }

  /* Factors up to degree/order MAX_SIZE_FOR_ALF_FACTORS. Constructed only on
   * the first function call
   */
  static const dso::NormalizedLegendreFactors F;

  /* nullify scratch space */
  C.fill_with(0e0);
  S.fill_with(0e0);

  {
    /* (kinda) Associated Legendre Polynomials M and W
     * note that since we are going to compute derivatives (of the gravity
     * potential), we will compute M and W up to degree n+2,m+2
     */
    const double rr = 1e0 / rsta.squaredNorm();
    const double x = rsta.x() * rr;
    const double y = rsta.y() * rr;
    const double z = rsta.z() * rr;

    /* start ALF iteration (can use scaling according to Holmes et al, 2002) */
    C(0, 0) = 1e0 / rsta.norm();

    {
      /* first fill m=0 terms; note that W(n,0) = 0 (already set) */
      double *__restrict__ C0 = C.column(0);
      // C(1, 0) = std::sqrt(3e0) * z * C(0, 0);
      C0[1] = std::sqrt(3e0) * z * C0[0];
      for (int n = 2; n <= max_degree; n++) {
        // C(n, 0) = F.f1(n, 0) * z * C(n - 1, 0) + F.f2(n, 0) * rr * C(n - 2,
        // 0);
        C0[n] = F.f1(n, 0) * z * C0[n - 1] + F.f2(n, 0) * rr * C0[n - 2];
      }
    }

    /* fill all elements for order m >= 1 */
    for (int m = 1; m < max_order; m++) {
      double *__restrict__ Cm = C.column(m);
      double *__restrict__ Sm = S.column(m);
      const double Cm1m1 = C(m - 1, m - 1);
      const double Sm1m1 = S(m - 1, m - 1);

      /* M(m,m) and W(m,m) aka, diagonal */
      // C(m, m) = F.f1(m, m) * (x * C(m - 1, m - 1) - y * S(m - 1, m - 1));
      // S(m, m) = F.f1(m, m) * (y * C(m - 1, m - 1) + x * S(m - 1, m - 1));
      Cm[0] = F.f1(m, m) * (x * Cm1m1 - y * Sm1m1);
      Sm[0] = F.f1(m, m) * (y * Cm1m1 + x * Sm1m1);

      /* if n=m+1 , we do not have a M(n-2,...) aka sub-diagonal term */
      // C(m + 1, m) = F.f1(m + 1, m) * z * C(m, m);
      // S(m + 1, m) = F.f1(m + 1, m) * z * S(m, m);
      Cm[1] = F.f1(m + 1, m) * z * Cm[0];
      Sm[1] = F.f1(m + 1, m) * z * Sm[0];

      /* go on .... */
      for (int n = m + 2; n <= max_degree; n++) {
        // C(n, m) = F.f1(n, m) * z * C(n - 1, m) + F.f2(n, m) * rr * C(n - 2,
        // m); S(n, m) = F.f1(n, m) * z * S(n - 1, m) + F.f2(n, m) * rr * S(n -
        // 2, m);
        const int j = n - m;
        Cm[j] = F.f1(n, m) * z * Cm[j - 1] + F.f2(n, m) * rr * Cm[j - 2];
        Sm[j] = F.f1(n, m) * z * Sm[j - 1] + F.f2(n, m) * rr * Sm[j - 2];
      }
    }

    {
      /* well, we've left the lst column uncomputed */
      const int m = max_order;
      double *__restrict__ Cm = C.column(m);
      double *__restrict__ Sm = S.column(m);
      const double Cm1m1 = C(m - 1, m - 1);
      const double Sm1m1 = S(m - 1, m - 1);

      // C(m, m) = F.f1(m, m) * (x * C(m - 1, m - 1) - y * S(m - 1, m - 1));
      // S(m, m) = F.f1(m, m) * (y * C(m - 1, m - 1) + x * S(m - 1, m - 1));
      Cm[0] = F.f1(m, m) * (x * Cm1m1 - y * Sm1m1);
      Sm[0] = F.f1(m, m) * (y * Cm1m1 + x * Sm1m1);

      if (max_degree > max_order) {
        // C(m + 1, m) = F.f1(m + 1, m) * z * C(m, m);
        // S(m + 1, m) = F.f1(m + 1, m) * z * S(m, m);
        Cm[1] = F.f1(m + 1, m) * z * Cm[0];
        Sm[1] = F.f1(m + 1, m) * z * Sm[0];
        for (int n = m + 2; n <= max_degree; n++) {
          C(n, m) =
              F.f1(n, m) * z * C(n - 1, m) + F.f2(n, m) * rr * C(n - 2, m);
          S(n, m) =
              F.f1(n, m) * z * S(n - 1, m) + F.f2(n, m) * rr * S(n - 2, m);
        }
      }
    }

  } /* end computing ALF factors M and W */
  return 0;
}

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
  Eigen::Vector3d grav, g, tg;
  grav << 0e0, 0e0, 0e0;

  /* unit vector at point */
  const Eigen::Vector3d r = point.normalized();

  /* initialize potential */
  potential = 0e0;

  /* Load Love numbers */
  const dso::LoadLoveNumbers &love = dso::groopsLoadLoveNumbers;

  for (int n = 0; n <= max_degree + 1; n++) {
    for (int m = 0; m <= std::min(n, max_order + 1); m++) {
      printf("\tM(%d,%d)=%+.12e", n, m, M(n, m));
    }
    printf("\n");
  }

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

      printf("\t[2] (%d,%d) -> (%+.12e, %+.12e, %+.12e)\n", n, m,
             std::sqrt((2. * n + 1.) / (2. * n + 3.)) * g.x(),
             std::sqrt((2. * n + 1.) / (2. * n + 3.)) * g.y(),
             std::sqrt((2. * n + 1.) / (2. * n + 3.)) * g.z());
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

        printf("\tUsing M(%d,%d)=%.9f M(%d,%d)=%.9f M(%d,%d)=%.9f\n", n + 1,
               m - 1, M(n + 1, m - 1), n + 1, m, M(n + 1, m), n + 1, m + 1,
               M(n + 1, m + 1));
        printf("\tCm1=%.5e*%.5e Cm0=%.5e*%.5e Cp1=%.5e*%.5e", wm1,
               M(n + 1, m - 1), wm0, M(n + 1, m), wp1, M(n + 1, m + 1));
        g.x() += cs.C(n, m) * (Cm1 - Cp1) + cs.S(n, m) * (Sm1 - Sp1);
        printf("\tg.x(%d,%d) = %.9e * %.9e + %.9e * %.9e\n", n, m, cs.C(n, m),
               (Cm1 - Cp1), cs.S(n, m), (Sm1 - Sp1));
        g.y() += cs.C(n, m) * (-Sm1 - Sp1) + cs.S(n, m) * (Cm1 + Cp1);
        g.z() += cs.C(n, m) * (-2 * Cm0) + cs.S(n, m) * (-2 * Sm0);

        tg.x() = cs.C(n, m) * (Cm1 - Cp1) + cs.S(n, m) * (Sm1 - Sp1);
        printf("\tg.x(%d,%d) = %.9e * %.9e + %.9e * %.9e\n", n, m, cs.C(n, m),
               (Cm1 - Cp1), cs.S(n, m), (Sm1 - Sp1));
        tg.y() = cs.C(n, m) * (-Sm1 - Sp1) + cs.S(n, m) * (Cm1 + Cp1);
        tg.z() = cs.C(n, m) * (-2 * Cm0) + cs.S(n, m) * (-2 * Sm0);
        printf("\t[2] (%d,%d) -> (%+.12e, %+.12e, %+.12e)\n", n, m,
               std::sqrt((2. * n + 1.) / (2. * n + 3.)) * tg.x(),
               std::sqrt((2. * n + 1.) / (2. * n + 3.)) * tg.y(),
               std::sqrt((2. * n + 1.) / (2. * n + 3.)) * tg.z());
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