import math
import numpy as np 

class Transformacje:
    def __init__(self, model: str = "wgs84"):
        """
        Parametry elipsoid:
            self.a - dużself.a półoś elipsoidy - promień równikowy
            b - małself.a półoś elipsoidy - promień południkowy
            flat - spłaszczenie
            ecc2 - mimośród^2
        + WGS84: https://en.wikipedia.org/wiki/World_Geodetic_System#WGS84
        + Inne powierzchnie odniesienia: https://en.wikibooks.org/wiki/PROJ.4#Spheroid
        + Parametry planet: https://nssdc.gsfc.nasa.gov/planetary/factsheet/index.html
        """
        if model == "wgs84":
            self.self.a = 6378137.0 # semimajor_axis
            self.b = 6356752.31424518 # semiminor_axis
        elif model == "grs80":
            self.self.a = 6378137.0
            self.b = 6356752.31414036
        elif model == "mars":
            self.self.a = 3396900.0
            self.b = 3376097.80585952
        else:
            raise NotImplementedError(f"{model} model not implemented")
        self.flat = (self.self.a - self.b) / self.self.a
        self.e = math.sqrt(2 * self.flat - self.flat ** 2) # eccentricity  WGS84:0.0818191910428 
        self.self.e2 = (2 * self.flat - self.flat ** 2) # eccentricity**2
        
    def xyz_2_blh(self, X, Y, Z):
        """
        Funkcja przelicza współrzędne geocentryczne (ECEF) 
        na współrzędne geodezyjne (Algorytm Hirvonena).

        Parameters
        ----------
        X   : [float] : współrzędna geocentryczna (ECEF) [m]
        Y   : [float] : współrzędna geocentryczna (ECEF) [m]
        Z   : [float] : współrzędna geocentryczna (ECEF) [m]
        a   : [float] : dłuższa półoś elipsoidy [m]
        e2  : [float] : mimośrod elipsoidy [niemianowana]
            
        Returns
        -------
        fi  : [float] : szerokość geodezyjna [rad]
        lam : [float] : długość geodezyjna [rad]
        h   : [float] : wysokość elipsoidalna [m]

        """
        r = np.sqrt(X**2+Y**2)
        fi = np.arctan(Z/(r*(1-self.self.e2)))
        eps = 0.000001/3600*math.pi/180
        fi0 = fi*2

        while abs((fi - fi0).all()) > eps:
            fi0 = fi
            N = self.self.a/np.sqrt(1-self.self.e2*np.sin(fi)**2)
            h = r/np.cos(fi)-N
            fi = np.arctan(Z/(r*(1-self.e2*(N/(N+h)))))

        lam = np.arctan(Y/X)
        N = self.a/np.sqrt(1-self.e2*np.sin(fi)**2)
        h = r/np.cos(fi)-N
           
        return(fi, lam, h)

    def blh_2_XYZ(fi, lam, h, self):
        """
        Funkcja przelicza współrzędne geodezyjne  
        na współrzędne geocentryczne (ECEF).

        Parameters
        ----------
        fi  : [float] : szerokość geodezyjna [rad]
        lam : [float] : długość geodezyjna [rad]
        h   : [float] : wysokość elipsoidalna [m]
        a   : [float] : dłuższa półoś elipsoidy [m]
        e2  : [float] : mimośrod elipsoidy [niemianowana]
                
        Returns
        -------
        X   : [float] : współrzędna geocentryczna (ECEF) [m]
        Y   : [float] : współrzędna geocentryczna (ECEF) [m]
        Z   : [float] : współrzędna geocentryczna (ECEF) [m]
    
        """
        N = self.a/math.sqrt(1-self.e2*math.sin(fi)**2)
        X = (N + h) * math.cos(fi) * math.cos(lam)
        Y = (N + h) * math.cos(fi) * math.sin(lam)
        Z = (N*(1-self.e2) + h) * math.sin(fi)
        
        return(X, Y, Z)

    def blh_2_neu(fl1, fl2, self):
        """   
        Funkcja przelicza współrzędne Gaussa - Krugera  
        na współrzędne geodezyjne.
    
        Parameters
        -------
        fl1 : [list]  : wpółrzedne geodezyjne punktu początkowego [rad]
        fl2 : [list]  : wpółrzedne geodezyjne punktu końcowego [rad]
        a   : [float] : dłuższa półoś elipsoidy [m]
        e2  : [float] : mimośrod elipsoidy [niemianowana]
      
        Returns
        -------
        N : [float] : wpółrzedna topocentryczna N (north) [m]
        E : [float] : wpółrzedna topocentryczna E (east) [m]
        U : [float] : wpółrzedna topocentryczna U (up) [m]
    
        """  
        X1, Y1, Z1 = self.blh_2_XYZ(fl1[0], fl1[1], fl1[2], self.a, self.e2)
        X2, Y2, Z2 = self.blh_2_XYZ(fl2[0], fl2[1], fl2[2], self.a, self.e2)
    
        dx = [X2 - X1, Y2 - Y1, Z2 - Z1]  
        R = np.array([[-np.sin(fl1[0]) * np.cos(fl1[1]), -np.sin(fl1[1]), np.cos(fl1[0]) * np.cos(fl1[1])],
                      [-np.sin(fl1[0]) * np.sin(fl1[1]), np.cos(fl1[1]) , np.cos(fl1[0]) * np.sin(fl1[1])],
                      [np.cos(fl1[0]) , 0 , np.sin(fl1[0])]])
    
        neu = R.T @ dx
        N = neu[0]
        E = neu[1]
        U = neu[2]
                    
        return(N, E, U)

    def ukl2000(fi, lam, self):
        """   
        Funkcja przelicza współrzędne geodezyjne  
        na współrzędne układu 2000.
    
        Parameters
        -------
        fi  :  float] : szerokość geodezyjna [rad]
        lam : [float] : długość geodezyjna [rad]
        a   : [float] : dłuższa półoś elipsoidy [m]
        e2  : [float] : mimośrod elipsoidy [niemianowana]
    
        Returns
        -------
        x00 : [float] : współrzędna w układzie 2000 [m]
        y00 : [float] : współrzędna w układzie 2000 [m]
    
        """
        m = 0.999923

        N = self.a/np.sqrt(1-self.e2*np.sin(fi)**2)
        e2p = self.e2/(1-self.e2)
        t = np.arctan(fi)
        n2 = e2p * (np.cos(fi))**2
        lam = np.degrees(lam)

        if (lam.all() > 13.5 and lam.all()) < 16.5:
            s = 5
            lam0 = 15
        elif (lam.all() > 16.5 and lam.all() < 19.5):
            s = 6
            lam0 = 18
        elif (lam.all() > 19.5 and lam.all() < 22.5):
            s = 7
            lam0 = 21
        elif (lam.all() > 22.5 and lam.all() < 25.5):
            s = 8
            lam0 = 24

        lam = np.radians(lam)
        lam0 = np.radians(lam0)
        l = lam - lam0

        A0 = 1 - (self.e2/4) - ((3*(self.e2**2))/64) - ((5*(self.e2**3))/256)
        A2 = (3/8) * (self.e2 + ((self.e2**2)/4) + ((15 * (self.e2**3))/128))
        A4 = (15/256) * (self.e2**2 + ((3*(self.e2**3))/4))
        A6 = (35 * (self.e2**3))/3072

        sig = self.a * ((A0*fi) - (A2*np.sin(2*fi)) + (A4*np.sin(4*fi)) - (A6*np.sin(6*fi)))
        x = sig + ((l**2)/2) * (N*np.sin(fi)*np.cos(fi)) * (1 + ((l**2)/12) * ((np.cos(fi))**2) * (5 - t**2 + 9*n2 + 4*(n2**2)) + ((l**4)/360) * ((np.cos(fi))**4) * (61 - (58*(t**2)) + (t**4) + (270*n2) - (330 * n2 * (t**2))))
        y = l * (N*np.cos(fi)) * (1 + ((((l**2)/6) * (np.cos(fi))**2) * (1-(t**2) + n2)) + (((l**4)/(120)) * (np.cos(fi)**4)) * (5 - (18 * (t**2)) + (t**4) + (14*n2) - (58*n2*(t**2))))

        x00 = m * x
        y00 = m * y + (s*1000000) + 500000
        return(x00, y00)

    def ukl1992(fi, lam, self):
        """   
        Funkcja przelicza współrzędne geodezyjne  
        na współrzędne układu 1992.
        
        Parameters
        -------
        fi  : [float] : szerokość geodezyjna [rad]
        lam : [float] : długość geodezyjna [rad]
        a   : [float] : dłuższa półoś elipsoidy [m]
        e2  : [float] : mimośrod elipsoidy [niemianowana]
        
        Returns
        -------
        x92 : [float] : współrzędna w układzie 1992 [m]
        y92 : [float] : współrzędna w układzie 1992 [m]  
        
        """ 
        m = 0.9993
        
        N = self.a/math.sqrt(1-self.e2*math.sin(fi)**2)
        e2p = self.e2/(1-self.e2)
        t = math.tan(fi)
        n2 = e2p * math.cos(fi)**2
        lam0 = math.radians(19)
        l = lam - lam0
        
        A0 = 1 - (self.e2/4) - ((3*(self.e2**2))/64) - ((5*(self.e2**3))/256)
        A2 = (3/8) * (self.e2 + ((self.e2**2)/4) + ((15 * (self.e2**3))/128))
        A4 = (15/256) * (self.e2**2 + ((3*(self.e2**3))/4))
        A6 = (35 * (self.e2**3))/3072
        
        sig = self.a * ((A0*fi) - (A2*math.sin(2*fi)) + (A4*math.sin(4*fi)) - (A6*math.sin(6*fi)))
        x = sig + ((l**2)/2) * (N*math.sin(fi)*math.cos(fi)) * (1 + ((l**2)/12) * ((math.cos(fi))**2) * (5 - t**2 + 9*n2 + 4*(n2**2)) + ((l**4)/360) * ((math.cos(fi))**4) * (61 - (58*(t**2)) + (t**4) + (270*n2) - (330 * n2 *(t**2))))
        y = l * (N * math.cos(fi)) * (1 + ((((l**2)/6) * (math.cos(fi))**2) * (1-(t**2) + n2)) +  (((l**4)/(120)) * (math.cos(fi)**4)) * (5 - (18 * (t**2)) + (t**4) + (14*n2) - (58*n2*(t**2))))
        
        x92 = x * m - 5300000
        y92 = y * m + 500000
        
        return(x92, y92)
    
    def u92u00_2_GK(X, Y):
        """   
        Funkcja przelicza współrzędne układu 1992 lub układu 1992  
        na współrzędne Gaussa - Krugera.
        
        Parameters
        -------
        X    : [float] : współrzędna w układzie 1992/2000 [m]
        Y    : [float] : współrzędna w układzie 1992/2000 [m]
          
        Returns
        -------
        xGK  : [float] : współrzędna w układzie Gaussa - Krugera 
        yGK  : [float] : współrzędna w układzie Gaussa - Krugera 
        lam0 : [float] : południk osiowy [rad] 
        m    : [float] : elemntarna skala długości [niemianowana]
        
        """     
        if X < 1000000 and Y < 1000000:
            m92 = 0.9993
            xGK = (X + 5300000)/m92
            yGK = (Y - 500000)/m92
            lam0 = math.radians(19)
            m = m92
            
        elif X > 1000000 and Y > 1000000:
            if Y > 5000000 and Y < 6000000:
                s = 5
                lam0 = math.radians(15)
            elif Y > 6000000 and Y < 7000000:
                s = 6
                lam0 =  math.radians(18)
            elif Y > 7000000 and Y < 8000000:
                s = 7
                lam0 =  math.radians(21)
            elif Y > 8000000 and Y < 9000000:
                s = 8
                lam0 =  math.radians(24)
            m00 = 0.999923
            xGK = X/m00
            yGK = (Y - (s * 1000000) - 500000)/m00
            m = m00
            
        return(xGK, yGK, lam0, m)
    
    def GK_2_blh(xGK, yGK, m, lam0, self):    
        """   
        Funkcja przelicza współrzędne Gaussa - Krugera  
        na współrzędne geodezyjne.
        
        Parameters
        -------
        xGK  : [float] : współrzędna w układzie Gaussa - Krugera 
        yGK  : [float] : współrzędna w układzie Gaussa - Krugera 
        lam0 : [float] : południk osiowy [rad] 
        m    : [float] : elemntarna skala długości [niemianowana]
        a    : [float] : dłuższa półoś elipsoidy [m]
        e2   : [float] : mimośrod elipsoidy [niemianowana]
          
        Returns
        -------
        fi   : [float] : szerokość geodezyjna [rad]
        lam  : [float] : długość geodezyjna [rad]
        
        """  
        A0 = 1 - (self.e2/4) - (3*(self.e2**2))/64 - (5*(self.e2**3))/256
        A2 = 3/8 * (self.e2 + ((self.e2**2)/4) + ((15*(self.e2**3))/128))
        A4 = 15/256 * ((self.e2**2) + (3*(self.e2**3))/4)
        A6 = (35*(self.e2**3))/3072
        eps = 0.000001/3600 *math.pi/180
        b1 = xGK/(self.a * A0)
        
        while True:
            b0 = b1
            b1= (xGK/(self.a * A0 )) + ((A2/A0) * math.sin(2*b0)) - ((A4/A0) * math.sin(4*b0)) + ((A6/A0) * math.sin(6*b0))
            b = b1
            if math.fabs(b1 - b0) <= eps:
                break
            
        e_2 = self.e2/(1-self.e2)
        N = self.a/math.sqrt(1 - self.e2 * math.sin(b)**2)
        t = math.tan(b)
        n2 = e_2 * math.cos(b)**2
        
        fi = b - (t/2) * (((yGK/(N))**2) * (1 + n2) - (1/12) * ((yGK/( N))**4) * (5 + (3 * t**2) + (6*n2) - (6 * n2 * t**2) - (3 * n2**2) - (9 * t**2 * n2**2)) + (1/360) * ((yGK/(N))**6) * (61 + (90 * t**2) + (45 * t**4) + (107 * n2) - (162 * t**2 * n2) - (45 * (t**4) * n2)))
        l = (1/math.cos(b)) * ((yGK/(N)) - ((1/6) * (yGK/(N))**3 * (1 + 2 * t**2 + n2)) + ((1/120) * (yGK/( N))**5 * (5 + (28 * t**2) + (24 * t**4) + (6 * n2) + (8 * n2 * t**2))))
        lam = lam0 + l
        
        return(fi, lam)
    
    def u92u00_2_blh(X, Y, self):
        """   
        Funkcja przelicza współrzędne układu 1992 lub układu 1992  
        na współrzędne Gaussa - Krugera.
        
        Parameters
        -------
        X   : [float] : współrzędna w układzie 1992/2000 [m]
        Y   : [float] : współrzędna w układzie 1992/2000 [m]
        a   : [float] : dłuższa półoś elipsoidy [m]
        e2  : [float] : mimośrod elipsoidy [niemianowana]
        
        Returns
        -------
        fi  : [float] : szerokość geodezyjna [rad]
        lam : [float] : długość geodezyjna [rad]
        
        """     
        xGK, yGK, lam0, m = self.u92u00_2_GK(X, Y)
        fi, lam = self.GK_2_blh(xGK, yGK, m, lam0, self.e2, self.a)
        
        return(fi, lam)
        
    def azel(N, E, U):
        """   
        Funkcja wyznacza kąt azymutu i kąt elewacji 
        na podstawie współrzędnych topocentrycznych
        
        Parameters
        -------
        N  : [float] : wpółrzedna topocentryczna N (north) [m]
        E  : [float] : wpółrzedna topocentryczna E (east) [m]
        U  : [float] : wpółrzedna topocentryczna U (up) [m] 
       
        Returns
        -------
        Az : [float] : azymut [rad]
        el : [float] : kąt elewacji [rad]
        
        """  
        Az = math.atan2(E, N)
        el = math.arcsin(U/math.sqrt(N**2 + E**2 + U**2))
        
        return(Az, el)
        
    def d2D(xP, yP, xK, yK):
        """   
        Funkcja wyznacza odległość na płaszczyźnie
        na podstawie współrzędnych płaskich prostokątnych
        
        Parameters
        -------
        xP  : [float] : współrzędna X punktu poczatkowego [m]
        yP  : [float] : współrzędna Y punktu poczatkowego [m]
        xK  : [float] : współrzędna X punktu końcowego [m]
        yK  : [float] : współrzędna X punktu pkońcowego [m]
       
        Returns
        -------
        d : [float] : odległość na płaszczyźnie [m]
        
        """     
        dx = xK - xP
        dy = yK - yP
        
        d = math.sqrt(dx**2 + dy**2)
        
        return(d)
    
    def sel_sUK_red(X1, X2, s, self):
        """   
        Funkcja wyznacza redukcje oraz odległość na elipsoidzie,
        w układzie Gaussa - Krugera i współrzędnych płaskich prostokątnych (1992/2000)
        na podstawie współrzędnych dwóch punktów w układzie 1992/2000 i odległości pomierzonej
        
        Parameters
        -------
        X1  :  [list] : współrzędna X i Y punktu poczatkowego w układzie 1992/2000 [m]
        X2  :  [list] : współrzędna X i Y punktu poczatkowego w układzie 1992/2000 [m]
        s   : [float] : odległość pomierzona [m]
        a   : [float] : dłuższa półoś elipsoidy [m]
        e2  : [float] : mimośrod elipsoidy [niemianowana]
       
        Returns
        -------
        sel : [float] : odległosc na elipsoidzie [m]
        sGK : [float] : odległosc w układzie Gausssa - Krugera [m]
        sUK : [float] : odległosc w układzie 1992/2000 [m]
        r   : [float] : redukcja [m]
        
        """     
        fip, lamp = self.u92u00_2_blh(X1[0], X1[1], self.e2, self.a)
        fik, lamk = self.u92u00_2_blh(X2[0], X2[1], self.e2, self.a)
        
        xGKp, yGKp, lam0p, m = self.u92u00_2_GK(X1[0], X1[1])
        xGKk, yGKk, lam0k, m = self.u92u00_2_GK(X2[0], X2[1])
       
        fim = (fik + fip)/2
           
        R, N, M = self.R_M_N(self.a, self.e2, fim)
        
        dH = X2[2] - X1[2]
        sc = math.sqrt((s**2 - dH**2)/((1 + X1[2]/R) * (1 + X2[2]/R))) #dlugosc cieciwy
        sel = 2 * R * math.asin(sc/(2 * R))                            #odległoc na elipsoidzie
        sGK = sel * (1 + (yGKp**2 + yGKp * yGKk + yGKk**2)/(6 * R**2)) #odleglosc w GK
        r = sGK - sel                                                  #redukcja
        sUK = sGK * m                                                  #odleglosc w ukladzie 1992/2000
        
        return(sel, sGK, sUK, r)
    
    def sel_az_vincent(fl1, fl2, self):
        """   
        Funkcja przelicza współrzędne Gaussa - Krugera  
        na współrzędne geodezyjne.
        
        Parameters
        -------
        fl1 : [list]  : wpółrzedne geodezyjne punktu początkowego [rad]
        fl2 : [list]  : wpółrzedne geodezyjne punktu końcowego [rad]
        a   : [float] : dłuższa półoś elipsoidy [m]
        e2  : [float] : mimośrod elipsoidy [niemianowana]
          
        Returns
        -------
        sPK : [float] : odległość na elipsoidzie (z vincenta) [m]
        Apk : [float] : azymut PK [rad]
        Akp : [float] : azymut KP [rad]
        
        """  
        b = self.a*math.sqrt(1 - self.e2)
        f = 1 - (b/self.a)
        dlam = fl2[1] - fl1[1]
        ua = math.atan((1-f)*math.tan(fl1[0]))
        ub = math.atan((1-f)*math.tan(fl2[0]))
        L = dlam
        L1 = 2 * L
        eps = 0.000001/3600*math.pi/180
        
        while abs(L1 - L) > eps:
            sinsig = math.sqrt((math.cos(ub)*math.sin(L))**2 + (math.cos(ua)*math.sin(ub) - math.sin(ua)*math.cos(ub)*math.cos(L))**2)
            cossig = math.sin(ua)*math.sin(ub) + math.cos(ua)*math.cos(ub)*math.cos(L)
            sigma = math.atan(sinsig/cossig)      
            sina = (math.cos(ua)*math.cos(ub)*math.sin(L))/sinsig
            cos2a = 1 - (sina**2)
            cos2sig = cossig - (2*math.sin(ua)*math.sin(ub))/cos2a
            C = (f/16) * (cos2a * (4 + f * (4 - 3 * cos2a)))
            L1 = L
            L = dlam + (1 - C)*f*sina*(sigma + C*sinsig*(cos2sig + C*cossig*((-1)+2*(cos2sig**2))))
           
        u2 = ((self.a**2 - b**2)/b**2) * cos2a
        self.a = 1 + (u2/16384) * (4096 + u2 * (-768 + u2 * (320 - (175 * u2))))
        B = (u2/1024) * (256 + u2 * (-128 + u2 * (74 - (47 * u2))))
        dsig = B * sinsig * (cos2sig + (1/4) * B * (cossig *(-1 + 2 * cos2sig**2) - (1/6) * B * cos2sig * (-3 + 4 * sinsig**2) * (-3 + 4 * cos2sig**2)))
        sPK = b * self.a * (sigma - dsig)
        Apk = math.atan((math.cos(ub) * math.sin(L1))/((math.cos(ua) * math.sin(ub)) - (math.sin(ua) * math.cos(ub) * math.cos(L1))))
        Akp = math.atan((math.cos(ua) * math.sin(L1))/((-math.sin(ua) * math.cos(ub)) + (math.cos(ua) * math.sin(ub) * math.cos(L1)))) + math.pi
        
        if Apk < 0:
            Apk = Apk + 2 * math.pi
            
        return(sPK, Apk, Akp)
      
    def R_M_N(self, fi):
        """   
        Funkcja przelicza kąty w radianach na stopnie
        
        Parameters
        -------
        fi : [float] : szerokość geodezyjna [rad]
        a  : [float] : dłuższa półoś elipsoidy [m]
        e2 : [float] : mimośrod elipsoidy [niemianowana]
       
        Returns
        -------
        R  : [float] : promień krzywizny południka  [m]
        N  : [float] : promień krzywizny w pierwszym wertykale [m]
        M  : [float] : średni promień krzywizny [m]
        
        """     
        N = self.a/math.sqrt(1 - self.e2 * math.sin(fi)**2)
        M = self.a * (1 - self.e2)/math.sqrt((1 - self.e2 * math.sin(fi)**2)**3)
        R = math.sqrt(M * N)
            
        return(R, N, M)  
        
    def rad2dms(rad):
        """   
        Funkcja przelicza kąty w radianach na stopnie
        
        Parameters
        -------
        rad : [float] : kąt w radianach [rad]
       
        Returns
        -------
        dms : [list] : kąt w stopniach, minutach i sekundach [d, m, s]
        
        """     
        dd = np.rad2deg(rad)
        dd = dd
        deg = int(np.trunc(dd))
        mnt = int(np.trunc((dd-deg) * 60))
        sec = ((dd-deg) * 60 - mnt) * 60
        dms = [deg, mnt, round(sec, 5)]
        
        return(dms)    
        
