#include "datetime/datetime_write.hpp"
#include "earth_rotation.hpp"
#include "fundarg.hpp"
#include "geodesy/transformations.hpp"
#include "gravity.hpp"
#include "iau.hpp"
#include "ocean_tide.hpp"
#include "planets.hpp"
#include "pole_tide.hpp"
#include "solid_earth_tide.hpp"

using namespace dso;
constexpr const int ENU = 1;
constexpr const int MAX_DEGREE = 180 + 2;

int compute_displacement(int argc, char *argv[]) {
  if (argc != 6) {
    fprintf(stderr,
            "Usage: %s [eopc04.1962-now] [de421.bsp] [naif*.tls] "
            "[FES2014b_OCN_001fileList.txt] [FES2014b gfc dir]\n",
            argv[0]);
    return 1;
  }

  /* Approx ITRF coordinates for sites */
  Eigen::Vector3d rsta;
  rsta << 4595212.468e0, 2039473.691e0, 3912617.891e0;

  /* start and end epochs in TT */
  MjdEpoch start(year(2023), month(7), day_of_month(19));
  MjdEpoch end(year(2023), month(7), day_of_month(22));

  /* create an instance to hold EOP series */
  EopSeries eops;

  /* parse EOP values to the EopSeries instance for any t >= t1 and t < t2 */
  if (parse_iers_C04(argv[1], start - modified_julian_day(2),
                     end + modified_julian_day(2), eops)) {
    fprintf(stderr, "ERROR Failed parsing eop file\n");
    return 1;
  }

  /* allocate scratch space for computations */
  dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> W(
      MAX_DEGREE + 3, MAX_DEGREE + 3);
  dso::CoeffMatrix2D<dso::MatrixStorageType::LwTriangularColWise> M(
      MAX_DEGREE + 3, MAX_DEGREE + 3);

  /* Load CSPICE/NAIF Kernels */
  load_spice_kernel(argv[2]);
  load_spice_kernel(argv[3]);

  /* setup a SolidEarthTide instance */
  SolidEarthTide set;

  /* setup Ocean Tide model */
  dso::OceanTide octide = dso::groops_ocean_atlas(argv[4], argv[5]);

  /* print info */
  printf("# Ref. Frame %s\n", ENU ? "enu" : "xyz");

  char buf[64];
  Eigen::Vector3d rmon, rsun;
  auto tt = start;
  /* iterate through [start, end) with step=30[sec] */
  while (tt < end) {

    /* EOPS at time of request */
    EopRecord eop;
    if (EopSeries::out_of_bounds(eops.interpolate(tt, eop))) {
      fprintf(stderr, "Failed to interpolate: Epoch is out of bounds!\n");
      return 1;
    }

    /* compute GMST using an approximate value for UT1 (linear interpolation) */
    double dut1_approx;
    eops.approx_dut1(tt, dut1_approx);
    const double gmst = dso::gmst(tt, tt.tt2ut1(dut1_approx));

    /* compute fundamental arguments at given epoch */
    double fargs[5];
    dso::fundarg(tt, fargs);

    /* add libration effect [micro as] */
    {
      double dxp, dyp, dut1, dlod;
      dso::deop_libration(fargs, gmst, dxp, dyp, dut1, dlod);
      eop.xp() += dxp * 1e-6;
      eop.yp() += dyp * 1e-6;
      eop.dut() += dut1 * 1e-6;
      eop.lod() += dlod * 1e-6;
    }

    /* add ocean tidal effect [micro as] */
    {
      double dxp, dyp, dut1, dlod;
      dso::deop_ocean_tide(fargs, gmst, dxp, dyp, dut1, dlod);
      eop.xp() += dxp * 1e-6;
      eop.yp() += dyp * 1e-6;
      eop.dut() += dut1 * 1e-6;
      eop.lod() += dlod * 1e-6;
    }

    /* get Sun+Moon position in ICRF */
    if (planet_pos(Planet::SUN, tt, rsun)) {
      fprintf(stderr, "ERROR Failed to compute Sun position!\n");
      return 2;
    }
    if (planet_pos(Planet::MOON, tt, rmon)) {
      fprintf(stderr, "ERROR Failed to compute Sun position!\n");
      return 2;
    }

    /* get the rotation matrix, i.e. [TRS] = q * [CRS] */
    Eigen::Quaterniond q = dso::c2i06a(tt, eop);
    rsun = q * rsun;
    rmon = q * rmon;

    /* compute Solid Earth tide deformation (cartesian) */
    auto dr_setide =
        (set.displacement(tt, tt.tt2ut1(eop.dut()), rsta, rmon, rsun, fargs))
            .mv;

    /* compute Stokes coefficients for Ocean Tide at given epoch */
    octide.stokes_coeffs(tt, tt.tt2ut1(eop.dut()), fargs);

    /* compute displacement due to ocean tide */
    Eigen::Vector3d dr_octide, gravity;
    double potential;
    if (sh_deformation(rsta, octide.stokes_coeffs(), gravity, potential,
                       dr_octide, -1, -1, &M, &W)) {
      fprintf(stderr, "ERROR. Failed computing Ocean Tide deformation!\n");
      return 10;
    }

    /* compute displacement due to (solid earth) pole tide */
    auto dr_eptide = (PoleTide::deformation(
                          tt, eop.xp(), eop.yp(),
                          cartesian2spherical(CartesianCrdConstView(rsta))))
                         .mv;

    /* compute displacement due to ocean pole tide */
    auto dr_optide = (OceanPoleTide::deformation(
                          tt, eop.xp(), eop.yp(),
                          cartesian2spherical(CartesianCrdConstView(rsta))))
                         .mv;

    if (ENU) {
      const Eigen::Matrix3d R = lvlh(CartesianCrdConstView(rsta));
      dr_setide = R.transpose() * dr_setide;
      dr_octide = R.transpose() * dr_octide;
      dr_eptide = R.transpose() * dr_eptide;
    }
    printf("%s %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f\n",
           to_char<YMDFormat::YYYYMMDD, HMSFormat::HHMMSS>(tt, buf),
           dr_setide(0), dr_setide(1), dr_setide(2), dr_octide(0), dr_octide(1),
           dr_octide(2), dr_eptide(0), dr_eptide(1), dr_eptide(2));

    /* next date */
    tt.add_seconds_inplace(FractionalSeconds(30));
  }

  return 0;
}

int main(int argc, char *argv[]) { return compute_displacement(argc, argv); }
