
"""
Conjunction Detection & Collision Assessment for SSA
Mahalanobis distance, hard-body radius models, probabilistic risk scoring

UPDATED:
- Major speedup: precompute tracks + KDTree candidate generation
- Parallel pair refinement (ProcessPool or Threads)
- Keeps structure and outputs compatible with original main script
- Maintains Chan-1D Pc method by default (same downstream behavior)
"""

import os
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from scipy.optimize import minimize_scalar

try:
    from scipy.spatial import cKDTree
    _HAS_KDTREE = True
except Exception:
    _HAS_KDTREE = False

from orbit_mech_engine import SGP4Propagator, StateVector, OrbitalElement


# ===========================================
# DATA STRUCTURE
# ===========================================
@dataclass
class ConjunctionEvent:
    """Represents a potential collision event"""
    sat1_id: str
    sat1_name: str
    sat2_id: str
    sat2_name: str
    tca: datetime  # Time of closest approach
    min_distance: float  # Minimum approach distance (km)
    mahalanobis_distance: float  # Mahalanobis distance at TCA
    probability_of_collision: float  # Collision probability [0, 1]
    risk_level: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW", "NEGLIGIBLE"
    sat1_hbr: float  # Hard-body radius (km)
    sat2_hbr: float  # Hard-body radius (km)
    combined_hbr: float  # Sum of hard-body radii

    def to_dict(self) -> Dict:
        return {
            'sat1_id': self.sat1_id,
            'sat1_name': self.sat1_name,
            'sat2_id': self.sat2_id,
            'sat2_name': self.sat2_name,
            'tca': self.tca.isoformat(),
            'min_distance_km': round(self.min_distance, 3),
            'mahalanobis_distance': round(self.mahalanobis_distance, 3),
            'probability_of_collision': round(self.probability_of_collision, 6),
            'risk_level': self.risk_level,
            'combined_hbr_km': round(self.combined_hbr, 4),
        }


# ===========================================
# HARD-BODY RADIUS MODELS
# ===========================================
class HardBodyRadiusModel:
    """
    Estimates hard-body radius (collision avoidance radius) based on satellite class.
    """
    MODELS = {
        'starlink': 0.00565,   # km
        'oneweb': 0.00400,
        'iridium': 0.00390,
        'cubesat': 0.00050,
        'debris': 0.00010,
        'generic': 0.00300,
    }

    @classmethod
    def estimate(cls, object_name: str, norad_id: Optional[str] = None) -> float:
        name_lower = (object_name or "").lower()
        for keyword, hbr in cls.MODELS.items():
            if keyword in name_lower:
                return hbr
        return cls.MODELS['generic']


# ===========================================
# COVARIANCE MODEL (kept consistent with original)
# ===========================================
class CovarianceModel:
    @staticmethod
    def position_covariance(time_diff_seconds: float) -> float:
        days = abs(time_diff_seconds) / 86400
        sigma_0 = 0.5      # km
        growth_rate = 2.0  # km/day
        sigma = sigma_0 + growth_rate * days
        return max(sigma, sigma_0)

    @staticmethod
    def velocity_covariance(time_diff_seconds: float) -> float:
        days = abs(time_diff_seconds) / 86400
        sigma_v0 = 0.001
        growth_rate = 0.001
        sigma_v = sigma_v0 + growth_rate * days
        return max(sigma_v, sigma_v0)

    @staticmethod
    def build_covariance_matrix(time_diff_seconds: float, is_velocity: bool = False) -> np.ndarray:
        sigma = CovarianceModel.velocity_covariance(time_diff_seconds) if is_velocity else CovarianceModel.position_covariance(time_diff_seconds)
        return np.eye(3) * (sigma ** 2)


# ===========================================
# CONJUNCTION DETECTION
# ===========================================
class ConjunctionDetector:
    """
    Detects and assesses conjunction events between satellite pairs.
    """

    def __init__(
        self,
        time_horizon_days: int = 7,
        search_step_hours: float = 1.0,
        initial_screen_km: float = 100.0,
        refine_window_hours: float = 6.0,
        pc_min_filter: float = 1e-12
    ):
        self.time_horizon_days = time_horizon_days
        self.search_step_hours = search_step_hours
        self.initial_screen_km = initial_screen_km
        self.refine_window_hours = refine_window_hours
        self.pc_min_filter = pc_min_filter

    def assess_pair(
        self,
        oe1: OrbitalElement,
        oe2: OrbitalElement,
        name1: str,
        name2: str,
        id1: str,
        id2: str,
        start_epoch: Optional[datetime] = None,
        tca_candidate: Optional[datetime] = None
    ) -> Optional[ConjunctionEvent]:
        """
        Assess collision risk between two satellites.

        NOTE (updated):
        - If tca_candidate is provided, skips full grid search and refines locally.
        - If not provided, uses the original grid search + refinement.
        """
        if start_epoch is None:
            start_epoch = datetime.utcnow()

        prop1 = SGP4Propagator(oe1)
        prop2 = SGP4Propagator(oe2)

        # 1) Candidate acquisition (either from caller or from internal grid search)
        if tca_candidate is None:
            min_distance = float('inf')
            tca_candidate = None

            current_time = start_epoch
            end_time = start_epoch + timedelta(days=self.time_horizon_days)

            step = timedelta(hours=self.search_step_hours)
            while current_time < end_time:
                s1 = prop1.propagate(current_time)
                s2 = prop2.propagate(current_time)
                d = np.linalg.norm(s1.r - s2.r)
                if d < min_distance:
                    min_distance = d
                    tca_candidate = current_time
                current_time += step

            if tca_candidate is None or min_distance > self.initial_screen_km:
                return None
        else:
            # quick check at candidate
            s1 = prop1.propagate(tca_candidate)
            s2 = prop2.propagate(tca_candidate)
            min_distance = np.linalg.norm(s1.r - s2.r)
            if min_distance > self.initial_screen_km:
                return None

        # 2) Refinement
        tca = self._refine_tca(prop1, prop2, tca_candidate, window_hours=self.refine_window_hours)

        # 3) Final states at TCA
        s1_tca = prop1.propagate(tca)
        s2_tca = prop2.propagate(tca)

        min_distance = float(np.linalg.norm(s1_tca.r - s2_tca.r))

        # 4) Risk assessment
        hbr1 = HardBodyRadiusModel.estimate(name1, id1)
        hbr2 = HardBodyRadiusModel.estimate(name2, id2)
        combined_hbr = hbr1 + hbr2

        mahal = self._mahalanobis_distance(s1_tca, s2_tca, tca, oe1.epoch, oe2.epoch)
        pc = self._collision_probability(min_distance, combined_hbr, mahal)

        risk_level = self._risk_level(pc)

        if pc < self.pc_min_filter:
            return None

        return ConjunctionEvent(
            sat1_id=id1, sat1_name=name1,
            sat2_id=id2, sat2_name=name2,
            tca=tca,
            min_distance=min_distance,
            mahalanobis_distance=mahal,
            probability_of_collision=pc,
            risk_level=risk_level,
            sat1_hbr=hbr1,
            sat2_hbr=hbr2,
            combined_hbr=combined_hbr
        )

    def _risk_level(self, pc: float) -> str:
        if pc > 1e-4:
            return "CRITICAL"
        elif pc > 1e-5:
            return "HIGH"
        elif pc > 1e-6:
            return "MEDIUM"
        elif pc > 1e-7:
            return "LOW"
        else:
            return "NEGLIGIBLE"

    def _refine_tca(
        self,
        prop1: SGP4Propagator,
        prop2: SGP4Propagator,
        tca_approx: datetime,
        window_hours: float = 12.0
    ) -> datetime:
        """
        Refine time of closest approach using local optimization.
        Kept from original but window can be tuned smaller for speed.
        """
        t0 = tca_approx.timestamp()

        def dist_func(dt_hours: float) -> float:
            t = datetime.fromtimestamp(t0 + dt_hours * 3600.0)
            s1 = prop1.propagate(t)
            s2 = prop2.propagate(t)
            return float(np.linalg.norm(s1.r - s2.r))

        res = minimize_scalar(
            dist_func,
            bounds=(-window_hours, window_hours),
            method='bounded'
        )
        return datetime.fromtimestamp(t0 + float(res.x) * 3600.0)

    def _mahalanobis_distance(
        self,
        state1: StateVector,
        state2: StateVector,
        tca: datetime,
        epoch1: datetime,
        epoch2: datetime
    ) -> float:
        delta_r = state2.r - state1.r

        dt1 = (tca - epoch1).total_seconds()
        dt2 = (tca - epoch2).total_seconds()
        P = CovarianceModel.build_covariance_matrix(dt1, False) + CovarianceModel.build_covariance_matrix(dt2, False)

        try:
            invP = np.linalg.inv(P)
            return float(np.sqrt(delta_r @ invP @ delta_r))
        except np.linalg.LinAlgError:
            return float(np.linalg.norm(delta_r))

    def _collision_probability(self, min_distance: float, combined_hbr: float, mahalanobis_distance: float) -> float:
        """
        Chan 1D approximation — kept identical to preserve output behavior.
        """
        if mahalanobis_distance <= 0 or combined_hbr <= 0 or min_distance <= 0:
            return 0.0

        sigma = min_distance / mahalanobis_distance
        if sigma <= 0:
            return 0.0

        exponent = -0.5 * (min_distance / sigma) ** 2
        pc = (combined_hbr ** 2 / (2.0 * sigma ** 2)) * math.exp(exponent)

        return float(max(0.0, min(1.0, pc)))


# ===========================================
# PARALLEL WORKER (top-level for pickling)
# ===========================================
def _assess_pair_worker(args):
    """
    args: (detector_kwargs, oe1, oe2, name1, name2, id1, id2, start_epoch, tca_candidate)
    """
    detector_kwargs, oe1, oe2, name1, name2, id1, id2, start_epoch, tca_candidate = args
    det = ConjunctionDetector(**detector_kwargs)
    return det.assess_pair(
        oe1, oe2,
        name1, name2,
        id1, id2,
        start_epoch=start_epoch,
        tca_candidate=tca_candidate
    )


# ===========================================
# CONJUNCTION SEARCH (CATALOG)
# ===========================================
class ConjunctionSearch:
    """
    Batch conjunction search across satellite catalog.

    UPDATED:
    - Precompute tracks once
    - KDTree per epoch to generate candidate pairs
    - Parallel refinement + Pc computation
    """

    def __init__(self, detector: ConjunctionDetector, parallel: bool = True, use_processes: bool = True, max_workers: Optional[int] = None):
        self.detector = detector
        self.parallel = parallel
        self.use_processes = use_processes
        self.max_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)

    def search_catalog(self, orbital_elements: Dict[str, Tuple[OrbitalElement, str]]) -> List[ConjunctionEvent]:
        conjunctions: List[ConjunctionEvent] = []

        ids = list(orbital_elements.keys())
        n = len(ids)
        if n < 2:
            return conjunctions

        # Same behavior as your original: use epoch of first object as search start
        start_epoch = orbital_elements[ids[0]][0].epoch

        # Build arrays for speed
        oes = [orbital_elements[_id][0] for _id in ids]
        names = [orbital_elements[_id][1] for _id in ids]

        # 1) Build time grid
        step_sec = int(self.detector.search_step_hours * 3600)
        Nt = int((self.detector.time_horizon_days * 86400) // step_sec) + 1
        times = [start_epoch + timedelta(seconds=k * step_sec) for k in range(Nt)]

        # 2) Precompute tracks (positions only) ONCE: r[i,k,3]
        r = np.empty((n, Nt, 3), dtype=np.float64)
        props = [SGP4Propagator(oe) for oe in oes]

        for i in range(n):
            for k, t in enumerate(times):
                s = props[i].propagate(t)
                r[i, k, :] = s.r

        # 3) Candidate pair generation (KDTree if available, else brute)
        # Keep best (closest) time index for each pair
        best: Dict[Tuple[int, int], Tuple[float, int]] = {}

        screen = float(self.detector.initial_screen_km)

        if _HAS_KDTREE:
            for k in range(Nt):
                tree = cKDTree(r[:, k, :])
                pairs = tree.query_pairs(r=screen)  # set of (i,j)
                for i, j in pairs:
                    d = float(np.linalg.norm(r[i, k] - r[j, k]))
                    key = (i, j) if i < j else (j, i)
                    if key not in best or d < best[key][0]:
                        best[key] = (d, k)
        else:
            # Brute fallback (OK for small N)
            for k in range(Nt):
                for i in range(n):
                    for j in range(i + 1, n):
                        d = float(np.linalg.norm(r[i, k] - r[j, k]))
                        if d <= screen:
                            key = (i, j)
                            if key not in best or d < best[key][0]:
                                best[key] = (d, k)

        # No candidates -> done
        if not best:
            return conjunctions

        # 4) Refine candidates and compute risk (parallel)
        detector_kwargs = dict(
            time_horizon_days=self.detector.time_horizon_days,
            search_step_hours=self.detector.search_step_hours,
            initial_screen_km=self.detector.initial_screen_km,
            refine_window_hours=self.detector.refine_window_hours,
            pc_min_filter=self.detector.pc_min_filter
        )

        tasks = []
        for (i, j), (_, k) in best.items():
            tca_candidate = times[k]
            tasks.append((
                detector_kwargs,
                oes[i], oes[j],
                names[i], names[j],
                ids[i], ids[j],
                start_epoch,
                tca_candidate
            ))

        if self.parallel:
            Executor = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
            with Executor(max_workers=self.max_workers) as ex:
                futures = [ex.submit(_assess_pair_worker, t) for t in tasks]
                for f in as_completed(futures):
                    ev = f.result()
                    if ev is not None:
                        conjunctions.append(ev)
        else:
            for t in tasks:
                ev = _assess_pair_worker(t)
                if ev is not None:
                    conjunctions.append(ev)

        conjunctions.sort(key=lambda x: x.probability_of_collision, reverse=True)
        return conjunctions


# ===========================================
# EXAMPLE USAGE (same as original)
# ===========================================
if __name__ == "__main__":
    from parser_pipeline import CelestrackClient, TLEDatabase
    from sgp4.api import Satrec

    detector = ConjunctionDetector(
        time_horizon_days=7,
        search_step_hours=1.0,
        initial_screen_km=100.0,     # same screening behavior as your original
        refine_window_hours=6.0,     # smaller than 12 for speed (still accurate)
        pc_min_filter=1e-12
    )

    search = ConjunctionSearch(
        detector,
        parallel=True,
        use_processes=True,          # set False if you see pickling issues on your platform
        max_workers=None
    )

    db = TLEDatabase()
    if db.count_tles() == 0:
        print("Fetching TLEs from Celestrak...")
        tles = CelestrackClient.fetch_group(group="last-30-days")
        db.insert_batch(tles)

    all_tles = db.get_all_tles()

    orbital_elements: Dict[str, Tuple[OrbitalElement, str]] = {}

    MU_EARTH = 398600.4418  # km^3/s^2

    for tle in all_tles:
        try:
            sat = Satrec.twoline2rv(tle.line1, tle.line2)
            n_rad_s = sat.no_kozai / 60.0
            a = (MU_EARTH / (n_rad_s ** 2)) ** (1 / 3)

            oe = OrbitalElement(
                a=a,
                e=sat.ecco,
                i=sat.inclo,
                omega=sat.argpo,
                Omega=sat.nodeo,
                M=sat.mo,
                n=n_rad_s,
                epoch=tle.epoch,
                bstar=sat.bstar
            )
        except Exception as e:
            print(f"Skipping {tle.object_name}: {e}")
            continue

        orbital_elements[tle.norad_cat_id] = (oe, tle.object_name)

    print(f"Searching {len(orbital_elements)} objects for conjunctions...")
    conjunctions = search.search_catalog(orbital_elements)

    print(f"\nFound {len(conjunctions)} conjunction events (top 10):\n")
    for i, event in enumerate(conjunctions[:10]):
        print(f"{i+1}. {event.sat1_name} <-> {event.sat2_name}")
        print(f"   TCA: {event.tca}")
        print(f"   Min Distance: {event.min_distance:.2f} km")
        print(f"   Combined HBR: {event.combined_hbr:.4f} km")
        print(f"   Pc: {event.probability_of_collision:.2e}")
        print(f"   Risk: {event.risk_level}\n")
