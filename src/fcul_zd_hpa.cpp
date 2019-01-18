#include "iers2010.hpp"

/**
 * @details  This function determines the total zenith delay following 
 *           (Mendes and Pavlis, 2004). 
 *           This function is a translation/wrapper for the fortran FCUL_A
 *           subroutine, found here : 
 *           http://maia.usno.navy.mil/conv2010/software.html
 * 
 * @param[in]  dlat   Geodetic Latitude given in degrees (North Latitude)
 * @param[in]  dhgt   Height above ellipsoid given in meters 
 * @param[in]  pres   Surface pressure given in hPa (mbars) (Note 1)
 * @param[in]  wvp    Water vapor pressure in hPa (mbars) (Note 1)
 * @param[in]  lambda Laser wavelength (micrometers)
 * @param[out] f_ztd  Zenith total delay in meters
 * @param[out] f_zhd  Zenith hydrostatic delay in meters
 * @param[out] f_zwd  Zenith non-hydrostatic delay in meters
 * @return            An integer, always zero
 * 
 * @note
 *    -# The surface pressure provided was converted from inches Hg. 
 *       The water vapor pressure was calculated from the surface 
 *       temperature (Celsius) and Relative Humidity (% R.H.) at the station.
 *    -# Status: Class 1 model
 *
 * @verbatim
 *     given input: LATITUDE  = 30.67166667D0 degrees (McDonald Observatory)
 *                  ELLIP_HT  = 2010.344D0 meters 
 *                  PRESSURE  = 798.4188D0 hPa (August 14, 2009)
 *                  WVP       = 14.322D0 hPa (August 14, 2009)
 *                  LAMBDA_UM = 0.532D0 micrometers (See Mendes et al.)
 *     expected output: FCUL_ZTD = 1.935225924846803114D0 m
 *                      FCUL_ZHD = 1.932992176591644462D0 m
 *                      FCUL_ZWD = 0.2233748255158703871D-02 m
 * @endverbatim
 * 
 * @version 2009 August 14
 *
 * @cite iers2010
 *    Mendes, V.B. and E.C. Pavlis, 2004, 
 *     "High-accuracy zenith delay prediction at optical wavelengths,"
 *     Geophysical Res. Lett., 31, L14602, doi:10.1029/2004GL020308, 2004
 * 
 */
int iers2010::fcul_zd_hpa (const double& dlat,const double& dhgt,
        const double& pres,const double& wvp,const double& lambda,
        double& f_ztd,double& f_zhd,double& f_zwd)
{
    #ifdef USE_EXTERNAL_CONSTS
        constexpr double PI   (DPI);
        //constexpr double C    (DC);
    #else
        constexpr double PI   (3.14159265358979323846e0);
        // speed of light in vacuum (m/s)
        //constexpr double C    (2.99792458e8);
    #endif
    
    // CO2 content in ppm
    constexpr double xc ( 375.0e0 );
    // constant values to be used in Equation (20)
    // k1 and k3 are k1* and k3* 
    constexpr double k0 ( 238.0185e0  );
    constexpr double k1 ( 19990.975e0 );
    constexpr double k2 ( 57.362e0    );
    constexpr double k3 ( 579.55174e0 );
    
    // constant values to be used in Equation (32)
    constexpr double w0 ( 295.235e0   );
    constexpr double w1 ( 2.6422e0    );
    constexpr double w2 ( -0.032380e0 );
    constexpr double w3 ( 0.004028e0  );
    
    //  Wave number
    double sigma ( 1e0/lambda );
    
    // correction factor - Equation (24)
    double f ( 1e0 - 0.00266e0*cos(2e0*PI/180e0*dlat) - 0.00028e-3*dhgt );
    
    // correction for CO2 content
    double corr ( 1.0e0 + 0.534e-6*(xc-450e0) );
    
    // dispersion equation for the hydrostatic component - Equation (20)
    double sigma2 ( sigma * sigma );
    double fh ( 
            0.01e0*corr*(
                (k1*(k0+sigma2))
                /(pow((k0-sigma2),2)) +
                k3*(k2+sigma2)
                /(pow((k2-sigma2),2)) 
                )
            );
    
    // computation of the hydrostatic component - Equation (26)
    // caution: pressure in hectoPascal units
    f_zhd = 2.416579e-3*fh*pres/f;
    
    // dispersion equation for the non-hydrostatic component - Equation (32)
    double fnh ( 0.003101e0*(w0+3.0e0*w1*sigma2 +
          5.0e0*w2*(sigma2*sigma2)+7.0e0*w3*pow(sigma,6)) );
    
    // computation of the non-hydrostatic component - Equation (38)
    // caution: pressure in hectoPascal units
    f_zwd = 1.e-4*(5.316e0*fnh-3.759e0*fh)*wvp/f;
    
    // compute the zenith total delay
    f_ztd = f_zhd + f_zwd;
    
    // return
    return 0;
}
