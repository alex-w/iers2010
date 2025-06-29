#include "aod1b_data_stream.hpp"
#include "datetime/datetime_write.hpp"
#include <cstdio>

/* To produce reference results for this test, use the script
 * src/aod1b/linear_interpolate_coeffs.py
 */

int main(int argc, char *argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s [AOD1B]\n", argv[0]);
    return 1;
  }

  char buf[64];

  dso::Aod1bDataStream<dso::AOD1BCoefficientType::ATM> stream(argv[1]);
  dso::StokesCoeffs cs(120, 120, 0e0, 0e0);

  if (stream.initialize()) {
    return 1;
  }

  auto t = stream.stream().first_epoch();
  while (t < stream.stream().last_epoch()) {
    if (stream.coefficients_at(t, cs))
      return 8;
    printf("%s %+.15e %+.15e\n",
           dso::to_char<dso::YMDFormat::YYYYMMDD, dso::HMSFormat::HHMMSSF,
                        dso::nanoseconds>(t.tt2gps(), buf),
           cs.C(10, 9), cs.S(10, 9));
    t.add_seconds(dso::seconds(180));
  }

  /* some dummy epochs so that we go back & forth in the file */
  {
    t = stream.stream().first_epoch();
    t.add_seconds(dso::seconds(180));
    if (stream.coefficients_at(t, cs))
      return 8;
  }
  {
    t = stream.stream().first_epoch();
    t.add_seconds(dso::seconds(3600 * 9 + 1));
    if (stream.coefficients_at(t, cs))
      return 8;
  }
  {
    t = stream.stream().first_epoch();
    t.add_seconds(dso::seconds(3600 * 20 + 1));
    if (stream.coefficients_at(t, cs))
      return 8;
  }
  {
    t = stream.stream().first_epoch();
    t.add_seconds(dso::seconds(3600 * 6 + 1));
    if (stream.coefficients_at(t, cs))
      return 8;
  }
  { /* error */
    t = stream.stream().first_epoch();
    t.add_seconds(dso::seconds(-1));
    if (stream.coefficients_at(t, cs)) {
      // printf("Expected an error and got an error! OK\n");
      ;
    } else {
      return 9;
    }
  }
  { /* error */
    t = stream.stream().last_epoch();
    // t.add_seconds(dso::seconds(-1));
    if (stream.coefficients_at(t, cs)) {
      // printf("Expected an error and got an error! OK\n");
      ;
    } else {
      return 9;
    }
  }

  return 0;
}
