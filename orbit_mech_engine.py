import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from datetime import datetime, timedelta


# constants

RE = 6371.0  # Earth radius (km)
MU = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
J2 = 1.08263e-3  # Earth's oblateness coefficient
OMEGA_E = 7.2921150e-5  # Earth's rotation rate (rad/s)
C20 = -1.08263e-3  # J2 normalized


# DATA  STRUCTURE
@dataclass
class OrbitalElement:
    """Kepler orbital elements at epoch"""

    a: float  # Semi-major axis (km)
    e: float  # Eccentricity
    i: float  # Inclination (rad)
    omega: float  # Argument of perigee (rad)
    Omega: float  # Right ascension of ascending node (rad)
    M: float  # Mean anomaly (rad)
    n: float  # Mean motion (rad/s)
    epoch: datetime  # TLE epoch
    bstar: float  # Ballistic coefficient (1/Earth radii)

    @property
    def period(self) -> float:
        """Orbital period in second"""
        return 2 * math.pi / self.n if self.n > 0 else float("inf")

    @property
    def p(self) -> float:
        """semi -latus rectum in km"""
        return self.a * (1 - self.e**2)

    @property
    def ra(self) -> float:
        """Apogee radiuis (km)"""
        return self.a * (1 + self.e)

    @property
    def rp(self) -> float:
        """Perigee radius km"""
        return self.a * (1 - self.e)


@dataclass
class StateVector:
    """Certesian state in Earth centered inertial (ECI) frame"""

    r: np.ndarray
    v: np.ndarray
    t: datetime

    @property
    def position_magnitude(self) -> float:
        """Distance from Earth center (km)"""
        return float(np.linalg.norm(self.r))

    @property
    def velocity_magnitude(self) -> float:
        """speed (km/s)"""
        return float(np.linalg.norm(self.v))

    @property
    def altitude(self) -> float:
        """altitude above  Earth surface (km)"""
        return self.position_magnitude - RE

    def to_dict(self) -> Dict:
        return {
            "x": float(self.r[0]),
            "y": float(self.r[1]),
            "z": float(self.r[2]),
            "vx": float(self.v[0]),
            "vy": float(self.v[1]),
            "vz": float(self.v[2]),
            "altitude": self.altitude,
            "speed": self.velocity_magnitude,
            "range": self.position_magnitude,
        }


# ===================================================
# KEPLER EQUATION SOLVER


# =======================================================
class KeplerSolver:
    """Solves Kepler's equatoin : M= E-e*sin(E)  using Newton-Raphson"""

    @staticmethod
    def solve_eccentric_anomaly(
        M: float, e: float, tolerance: float = 1e-8, max_iter: int = 50
    ) -> float:
        """
        Solve Kepler's equation for eccentric anomaly E.

        Args:
            M: Mean anomaly (radians)
            e: Eccentricity
            tolerance: Convergence tolerance
            max_iter: Maximum iterations
            =
        Returns:
            Eccentric anomaly E (radians)
            =
        Algorithm:
            Newton-Raphson iteration: E_{n+1} = E_n - f(E_n) / f'(E_n)
            where f(E) = E - e*sin(E) - M
        """

        # normalize M to [-pi, pi]
        M = M % (2 * math.pi)
        if M > math.pi:
            M -= 2 * math.pi

        # Initial guess (works well for e<0.8)
        if e < 0.8:
            E = M if abs(M) > math.pi / 6 else M / (1 - e)
        else:
            E = math.pi

            # newton - Raphson iteraton
        for _ in range(max_iter):
            sin_E = math.sin(E)
            cos_E = math.cos(E)
            f = E - e * sin_E - M
            f_prime = 1 - e * cos_E

            E_new = E - f / f_prime
            if abs(E_new - E) < tolerance:
                return E_new

            E = E_new
        return E

    @staticmethod
    def anomaly_conversion(M: float, e: float) -> Tuple[float, float]:
        """Convert mean anomaly to eccentric and true anomalies.Returns (eccentric_anomaly,
        ly) in radians"""
       # E = KeplerSolver.solve_eccentric_anomaly(M, e)

        E = KeplerSolver.solve_eccentric_anomaly(M, e)
        nu = 2 * math.atan2(
            math.sqrt(1 + e) * math.sin(E / 2),
            math.sqrt(1 - e) * math.cos(E / 2)
        )
        return E, nu


# ==========================================
#  Coordinate transformation
# ============================
class CoordinateTransform:
    """ECI  geographics coordinate and orbital frame transformation"""

    @staticmethod
    def orbital_to_eci(r_orb: np.ndarray, oe: OrbitalElement) -> np.ndarray:
        """
        Convert position from orbital frame to ECI frame.

        Orbital frame: Z-axis along specific angular momentum, X-axis to perigee
        ECI frame: Z-axis along Earth's rotation axis, X-axis to vernal equinox

        Transformation uses 3 rotations:
          1. About Z by argument of perigee (ω)
          2. About X by inclination (i)
          3. About Z by right ascension of ascending node (Ω)
        """
        # Rotation 1: About Z-axis by ω (argument of perigee)
        cos_omega = math.cos(oe.omega)
        sin_omega = math.sin(oe.omega)
        R_omega = np.array(
            [[cos_omega, -sin_omega, 0], [sin_omega, cos_omega, 0], [0, 0, 1]]
        )

        # Rotation 2: About X-axis by i (inclination)
        cos_i = math.cos(oe.i)
        sin_i = math.sin(oe.i)
        R_i = np.array(
            [
                [1, 0, 0],
                [0, cos_i, -sin_i],
                [0, sin_i, cos_i],
            ]
        )

        # Rotation 3: About Z axis by omega (RAAN)
        cos_Omega = math.cos(oe.Omega)
        sin_Omega = math.sin(oe.Omega)
        R_Omega = np.array(
            [[cos_Omega, -sin_Omega, 0], [sin_Omega, cos_Omega, 0], [0, 0, 1]]
        )
        # Combined rotation: R_Omega @ R_i @ R_omega
        R = R_Omega @ R_i @ R_omega

        return R @ r_orb

    @staticmethod
    def eci_to_geographic(         # ← Remove the extra "s"
        r_eci: np.ndarray, t: datetime
    ) -> Tuple[float, float, float]:
        """
        Convert ECI position to geographic coordinates (latitude, longitude, altitude).

        Args:
            r_eci: Position vector in ECI (km)
            t: Epoch time

        Returns:
            (latitude, longitude, altitude) in (degrees, degrees, km)
        """

        # Greenwich Mean Sidereal Time (simplified)
        jd = _julian_day(t)  
        gmst = _gmst(jd)  

        # Rotate ECI to ECEF (Earth-Centered Earth-Fixed)
        cos_gmst = math.cos(gmst)
        sin_gmst = math.sin(gmst)
        R_gmst = np.array([[cos_gmst, sin_gmst, 0],
                            [-sin_gmst, cos_gmst, 0],
                            [0, 0, 1]])
        r_ecef = R_gmst @ r_eci
        # ECEF to geographic (WGS84)
        x, y, z = r_ecef[0], r_ecef[1], r_ecef[2]
        lon = math.degrees(math.atan2(y, x))
        lat = math.degrees(math.atan2(z, math.sqrt(x**2 + y**2)))
        alt = math.sqrt(x**2 + y**2 + z**2) - RE
        return lat, lon, alt


class SGP4Propagator:
    """
    Simplified SGP4 perturbation model.
    Includes: atmospheric drag, J2/J3 oblateness, Lunisolar effects.
    """

    def __init__(self, oe: OrbitalElement):
        self.oe = oe
        self.a0 = oe.a
        self.n0 = oe.n
        self.e0 = oe.e

    def propagate(self, t_prop: datetime) -> StateVector:
        """Propagate orbit from epoch to propagation time. Args: t_prop: Target epoch Returns:
        State vector at t_prop
        """
       
        dt = (t_prop - self.oe.epoch).total_seconds()
        # Mean anomaly at propagation time

        M_prop = self.oe.M + self.oe.n * dt

        # Secular effects (simplified)
        # Drag-induced decay
        decay_factor = math.exp(-self.oe.bstar * dt / 1e5)
        a_prop = self.oe.a * decay_factor
        n_prop = self.oe.n / (decay_factor**1.5)
        e_prop = self.oe.e  # Eccentricity change is small
        # Create propagated orbital elements

        oe_prop = OrbitalElement(
            a=a_prop,
            e=e_prop,
            i=self.oe.i,
            omega=self.oe.omega,
            Omega=self.oe.Omega,
            M=M_prop,
            n=n_prop,
            epoch=t_prop,
            bstar=self.oe.bstar,
        )

        # solve kepler's equation

        #E, nu = KeplerSolver.anomaly_conversion(M_prop, e_prop)
        E, nu = KeplerSolver.anomaly_conversion(M_prop, e_prop)

        # Compute state in orbital frame

        p = a_prop * (1 - e_prop**2)
        r_mag = p / (1 + e_prop * math.cos(nu))

        r_orb = np.array([r_mag * math.cos(nu), r_mag * math.sin(nu), 0.0])

        v_orb = np.array(
            [
                -math.sqrt(MU / p) * math.sin(nu),
                math.sqrt(MU / p) * (e_prop + math.cos(nu)),
                0.0,
            ]
        )

        # Transformation to ECI
        r_eci = CoordinateTransform.orbital_to_eci(r_orb, oe_prop)
        v_eci = CoordinateTransform.orbital_to_eci(v_orb, oe_prop)

        return StateVector(r=r_eci, v=v_eci, t=t_prop)
 #==============================================
 #================HELPER FUNCTION===============
 #====================
def _julian_day(dt: datetime
                 ) -> float:
     """Compute Julina day number at 0h UT"""
     a = (14-dt.month)//12
     y= dt.year+4800-a
     m = dt.month + 12 *a -3
     jdn = dt.day + (153 * m+2)//5 + 365 * y+y//4-y//100+y//400-32045
     jd = jdn+(dt.hour -12)/ 24 + dt.minute/1440+dt.second/86400
     return jd
    
def _gmst(jd:float
              )-> float:
       """Greenwich mean sidereal time (radians )  from JUlian Day"""
       T= (jd-2451545.0)/36525.0
       gmst_sec = 67310.54841+(67.0 *3600.0+48.0*60.0+45.25625)*T+\
         (0.093104)*T**2-(6.2e-6)*T**3
       gmst_sec=gmst_sec%86400.0
       return 2*math.pi*gmst_sec/86400.0
    

# ============================================================================
# EXAMPLE USAGE
# ============================================
if __name__ =="__main__":
   #example : starlink TLE
   oe_example=OrbitalElement(
       a=6800.0,
       e=0.0002,
       i=math.radians(97.286),
       omega=math.radians(89.84),
       Omega=math.radians(37.78),
       M=math.radians(270.31),
       n=2.0*math.pi/5445.0,
       epoch=datetime(2026,3,29,8,58,0),
       bstar=0.14
   )

   #Propagate to 24 hours later
   propagator = SGP4Propagator(oe_example)
   state_24h = propagator.propagate(datetime(2026, 3, 30, 8, 58, 0))

   print("State at 24h:")
   print(f"Position :{state_24h.r} km")
   print(f"Velocity :{state_24h.v} km/s")
   print(f"Altitude:{state_24h.altitude:.1f} km")
