#include "hardisp.hpp"

/**
 * @details This function returns the ocean loading displacement amplitude,
 *          frequency, and phase of a set of tidal constituents generated by
 *          the Bos-Scherneck website at http://www.oso.chalmers.se/~loading/.
 *          The variable nin is input as the number wanted, and the variable 
 *          nout is returned as the number provided.  The constituents used
 *          are stored in the arrays idd (Doodson number) and tamp
 *          (Cartwright-Edden amplitude).  The actual amp and phase of each
 *          of these are determined by spline interpolation of the real and
 *          imaginary part of the admittance, as specified at a subset of the
 *          constituents.
 * 
 * @param[in]  ampin  Cartwright-Edden amplitude of tidal constituents
 * @param[in]  idtin  Doodson number of tidal constituents
 * @param[in]  phin   Phase of tidal constituents
 * @param[in]  nin    Number of harmonics used
 * @param[out] amp    Amplitude due to ocean loading
 * @param[out] f      Frequency due to ocean loading
 * @param[out] p      Phase due to ocean loading
 * @param[out] nout   Number of harmonics returned
 * @return            zero
 * 
 * @note
 *     -# The phase is determined for a time set in COMMON block /date/ in
 *        the function TDFRPH.
 *     -# The arrays F and P must be specified as double precision.
 *     -# Status:  Class 1 model
 * 
 * @version 2009 August 19
 * 
 * @cite McCarthy, D. D., Petit, G. (eds.), IERS Conventions (2003),
 *       IERS Technical Note No. 32, BKG (2004)
 * 
 */
int iers2010::hisp::admint (const double& ampin, const int& idtin, const double& phin, const int& nin
            double& amp, double& f, double& p, int& nout)
 {
   
 }