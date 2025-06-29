      PROGRAM MAIN

      DOUBLE PRECISION T, MJD, MIN_MJD, MAX_MJD, DUT1, DLOD, PM(2)
      INTEGER NUM_TESTS, I,J

      NUM_TESTS = 1000
      MIN_MJD = 47892D0
      MAX_MJD = 66154D0

      DO I=1,NUM_TESTS-1
        CALL RANDOM_NUMBER(T)
        MJD = (MAX_MJD-MIN_MJD)*T + MIN_MJD
        CALL UTLIBR(MJD, DUT1, DLOD)
        CALL PMSDNUT2(MJD, PM)
        WRITE(*,'(E30.22,X,E25.17,X,E25.17,X,E25.17,X,E25.17)') 
     .  MJD, DUT1, DLOD, PM(1), PM(2)
      END DO

      END
