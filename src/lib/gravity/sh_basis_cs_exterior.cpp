#include "gravity.hpp"
#include <cassert>

/* Alternate implementation of the below, which is actually only 1% faster 
 * according to the benchmarks, but quite cleared. So just fuck it and use 
 * this version which can be debugged a lot easier (and can also trigger 
 * errors/warning if the Debug version is buit.
 */
#ifdef ENABLE_BENCHMARKS
int dso::gravity::sh_basis_cs_exterior2(
    const Eigen::Vector3d &rsta, int max_degree, int max_order,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> &C,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>
        &S) noexcept {

  /* check size of matrices */
  if (((C.rows() < max_degree) || (C.cols() < max_degree)) ||
      ((S.rows() < max_degree) || (S.cols() < max_degree))) {
    fprintf(stderr,
            "[ERROR] Invalid C/S size(s) for computing SH coefficients "
            "(traceback: %s)\n",
            __func__);
    return 1;
  }

  /* check if we are ok with the NormalizedLegendreFactors size */
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
      const double *f10 = F.f1.column(0);
      const double *f20 = F.f2.column(0);

      // C(1, 0) = std::sqrt(3e0) * z * C(0, 0);
      C0[1] = std::sqrt(3e0) * z * C0[0];
      
      for (int n = 2; n <= max_degree; n++) {
        // C(n, 0) = F.f1(n, 0) * z * C(n - 1, 0) + F.f2(n, 0) * rr * C(n - 2,
        // 0);
        C0[n] = f10[n] * z * C0[n - 1] + f20[n] * rr * C0[n - 2];
      }
    }

    /* fill all elements for order m >= 1 */
    for (int m = 1; m < max_order; m++) {
      double *__restrict__ Cm = C.column(m);
      double *__restrict__ Sm = S.column(m);
      const double Cm1m1 = C(m - 1, m - 1);
      const double Sm1m1 = S(m - 1, m - 1);
      const double *f1m = F.f1.column(m);
      const double *f2m = F.f2.column(m);

      /* M(m,m) and W(m,m) aka, diagonal */
      // C(m, m) = F.f1(m, m) * (x * C(m - 1, m - 1) - y * S(m - 1, m - 1));
      // S(m, m) = F.f1(m, m) * (y * C(m - 1, m - 1) + x * S(m - 1, m - 1));
      Cm[0] = f1m[0] * (x * Cm1m1 - y * Sm1m1);
      Sm[0] = f1m[0] * (y * Cm1m1 + x * Sm1m1);

      /* if n=m+1 , we do not have a M(n-2,...) aka sub-diagonal term */
      // C(m + 1, m) = F.f1(m + 1, m) * z * C(m, m);
      // S(m + 1, m) = F.f1(m + 1, m) * z * S(m, m);
      Cm[1] = f1m[1] * z * Cm[0];
      Sm[1] = f1m[1] * z * Sm[0];

      /* go on .... */
      for (int n = m + 2; n <= max_degree; n++) {
        // C(n, m) = F.f1(n, m) * z * C(n - 1, m) + F.f2(n, m) * rr * C(n - 2,
        // m); S(n, m) = F.f1(n, m) * z * S(n - 1, m) + F.f2(n, m) * rr * S(n -
        // 2, m);
        const int j = n - m;
        Cm[j] = f1m[j] * z * Cm[j - 1] + f2m[j] * rr * Cm[j - 2];
        Sm[j] = f1m[j] * z * Sm[j - 1] + f2m[j] * rr * Sm[j - 2];
      }
    }

    {
      /* well, we've left the lst column uncomputed */
      const int m = max_order;
      double *__restrict__ Cm = C.column(m);
      double *__restrict__ Sm = S.column(m);
      const double Cm1m1 = C(m - 1, m - 1);
      const double Sm1m1 = S(m - 1, m - 1);
      const double *f1m = F.f1.column(m);
      const double *f2m = F.f2.column(m);

      // C(m, m) = F.f1(m, m) * (x * C(m - 1, m - 1) - y * S(m - 1, m - 1));
      // S(m, m) = F.f1(m, m) * (y * C(m - 1, m - 1) + x * S(m - 1, m - 1));
      Cm[0] = f1m[0] * (x * Cm1m1 - y * Sm1m1);
      Sm[0] = f1m[0] * (y * Cm1m1 + x * Sm1m1);

      if (max_degree > max_order) {
        // C(m + 1, m) = F.f1(m + 1, m) * z * C(m, m);
        // S(m + 1, m) = F.f1(m + 1, m) * z * S(m, m);
        Cm[1] = f1m[1] * z * Cm[0];
        Sm[1] = f1m[1] * z * Sm[0];
        for (int n = m + 2; n <= max_degree; n++) {
          //C(n, m) =
          //    F.f1(n, m) * z * C(n - 1, m) + F.f2(n, m) * rr * C(n - 2, m);
          //S(n, m) =
          //    F.f1(n, m) * z * S(n - 1, m) + F.f2(n, m) * rr * S(n - 2, m);
          const int j = n - m;
          Cm[j] = f1m[j] * z * Cm[j-1] + f2m[j] * rr * Cm[j-2];
          Sm[j] = f1m[j] * z * Sm[j-1] + f2m[j] * rr * Sm[j-2];
        }
      }
    }

  } /* end computing ALF factors M and W */
  return 0;
}
#endif

int dso::gravity::sh_basis_cs_exterior(
    const Eigen::Vector3d &rsta, int max_degree, int max_order,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> &C,
    dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise>
        &S) noexcept {

  /* check size of matrices */
  if (((C.rows() < max_degree) || (C.cols() < max_degree)) ||
      ((S.rows() < max_degree) || (S.cols() < max_degree))) {
    fprintf(stderr,
            "[ERROR] Invalid C/S size(s) for computing SH coefficients "
            "(traceback: %s)\n",
            __func__);
    return 1;
  }

  /* check if we are ok with the NormalizedLegendreFactors size */
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
      C(1, 0) = std::sqrt(3e0) * z * C(0, 0);
      
      for (int n = 2; n <= max_degree; n++) {
        C(n, 0) = F.f1(n, 0) * z * C(n - 1, 0) + F.f2(n, 0) * rr * C(n - 2,
        0);
      }
    }

    /* fill all elements for order m >= 1 */
    for (int m = 1; m < max_order; m++) {

      /* M(m,m) and W(m,m) aka, diagonal */
      C(m, m) = F.f1(m, m) * (x * C(m - 1, m - 1) - y * S(m - 1, m - 1));
      S(m, m) = F.f1(m, m) * (y * C(m - 1, m - 1) + x * S(m - 1, m - 1));

      /* if n=m+1 , we do not have a M(n-2,...) aka sub-diagonal term */
      C(m + 1, m) = F.f1(m + 1, m) * z * C(m, m);
      S(m + 1, m) = F.f1(m + 1, m) * z * S(m, m);

      /* go on .... */
      for (int n = m + 2; n <= max_degree; n++) {
        C(n, m) = F.f1(n, m) * z * C(n - 1, m) + F.f2(n, m) * rr * C(n - 2,
        m); S(n, m) = F.f1(n, m) * z * S(n - 1, m) + F.f2(n, m) * rr * S(n -
        2, m);
      }
    }

    {
      /* well, we've left the lst column uncomputed */
      const int m = max_order;

      C(m, m) = F.f1(m, m) * (x * C(m - 1, m - 1) - y * S(m - 1, m - 1));
      S(m, m) = F.f1(m, m) * (y * C(m - 1, m - 1) + x * S(m - 1, m - 1));

      if (max_degree > max_order) {
        C(m + 1, m) = F.f1(m + 1, m) * z * C(m, m);
        S(m + 1, m) = F.f1(m + 1, m) * z * S(m, m);
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
