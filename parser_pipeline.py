"""
TLE Data Pipeline for SSA
Fetches live TLEs from Celestrak, parses, validates, and stores in SQLite
"""

import requests
import sqlite3
import json
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
from orbit_mech_engine import OrbitalElement
from sgp4.api import Satrec


# ============================================================================
# TLE DATACLASS
# ============================================================================

@dataclass
class TLE:
    """TLE record with computed orbital elements"""
    object_name: str
    norad_cat_id: str
    epoch_year: int
    epoch_day: float
    bstar: float
    mean_motion_dot: float
    mean_motion_ddot: float
    element_number: int
    line1: str
    line2: str
    inclination: float
    raan: float
    eccentricity: float
    arg_of_perigee: float
    mean_anomaly: float
    mean_motion: float
    
    @property
    def epoch(self) -> datetime:
        """Convert TLE epoch (YY DDD.FFFFFFF) to datetime"""
        yy = self.epoch_year
        year = 2000 + yy if yy < 70 else 1900 + yy
        jan1 = datetime(year, 1, 1)
        day_of_year = int(self.epoch_day)
        fraction = self.epoch_day - day_of_year
        epoch = jan1 + timedelta(days=day_of_year - 1, seconds=fraction * 86400)
        return epoch
    
    def to_dict(self) -> Dict:
        """Convert TLE to dictionary"""
        return asdict(self)
    
    def to_orbital_element(self):
        MU_EARTH = 398600.4418
        sat = Satrec.twoline2rv(self.line1, self.line2)
        n_rad_s = sat.no_kozai / 60.0
        a = (MU_EARTH / (n_rad_s ** 2)) ** (1.0 / 3.0)
        return OrbitalElement(
            a=a,
            e=sat.ecco,
            i=sat.inclo,
            omega=sat.argpo,
            Omega=sat.nodeo,
            M=sat.mo,
            n=n_rad_s,
            epoch=self.epoch,
            bstar=sat.bstar
        )

# ============================================================================
# TLE PARSER
# ============================================================================

class TLEParser:
    """Parse TLE records from text format"""
    
    @staticmethod
    def _parse_bstar(bstar_str: str) -> float:
        """
        Parse BSTAR drag term from TLE compressed exponential format.
        """
        try:
            bstar_str = bstar_str.strip()
            if not bstar_str:
                return 0.0
            
            sign_pos = -1
            for i in range(len(bstar_str) - 1, -1, -1):
                if bstar_str[i] in ['+', '-']:
                    sign_pos = i
                    break
            
            if sign_pos < 0:
                return 0.0
            
            mantissa_str = bstar_str[:sign_pos].strip()
            exponent_str = bstar_str[sign_pos:]
            
            if not mantissa_str or not exponent_str:
                return 0.0
            
            mantissa = float(mantissa_str) / 1e5
            exponent = int(exponent_str)
            bstar = mantissa * (10 ** exponent)
            return bstar
            
        except (ValueError, IndexError):
            return 0.0
    
    @staticmethod
    def _parse_mean_motion_ddot(mmddot_str: str) -> float:
        """
        Parse mean motion second derivative (positions 44-52 of Line 1).
        Uses same compressed exponential format as BSTAR.
        Example: "00000-0" → 0.0, "12345-6" → 0.0000012345
        """
        try:
            s = mmddot_str.strip()
            if not s or s == '00000-0':
                return 0.0
            
            sign_pos = -1
            for i in range(len(s) - 1, -1, -1):
                if s[i] in ['+', '-']:
                    sign_pos = i
                    break
            
            if sign_pos < 0:
                return 0.0
            
            mantissa_str = s[:sign_pos].strip()
            exponent_str = s[sign_pos:]
            
            if not mantissa_str:
                return 0.0
            
            mantissa = float(mantissa_str) / 1e5
            exponent = int(exponent_str)
            value = mantissa * (10 ** exponent)
            return value
        
        except (ValueError, IndexError):
            return 0.0

    @staticmethod
    def parse_lines(name: str, line1: str, line2: str) -> Optional[TLE]:
        """Parse two-line TLE format."""
        try:
            # Line 1 parsing
            if not line1 or line1[0] != '1':
                return None
        
            try:
                norad_cat_id = line1[2:7].strip()
                epoch_year = int(line1[18:20])
                epoch_day = float(line1[20:32])
                mean_motion_dot = float(line1[33:43])
                
                # Mean motion second derivative - now properly parsed
                mean_motion_ddot = TLEParser._parse_mean_motion_ddot(line1[44:52])
                
                # BSTAR
                bstar = TLEParser._parse_bstar(line1[53:61])
                
                element_number = int(line1[64:68])
                
            except (ValueError, IndexError) as e:
                print(f"  Error parsing Line1: {e}")
                return None

            # Line 2 parsing
            if not line2 or line2[0] != '2':
                return None
        
            try:
                inclination = float(line2[8:16])
                raan = float(line2[17:25])
                eccentricity_raw = float(line2[26:33]) / 1e7
                arg_of_perigee = float(line2[34:42])
                mean_anomaly = float(line2[43:51])
                mean_motion = float(line2[52:63])
            except (ValueError, IndexError) as e:
                print(f"  Error parsing Line2: {e}")
                return None

            tle = TLE(
                object_name=name,
                norad_cat_id=norad_cat_id,
                epoch_year=epoch_year,
                epoch_day=epoch_day,
                bstar=bstar,
                mean_motion_dot=mean_motion_dot,
                mean_motion_ddot=mean_motion_ddot,
                element_number=element_number,
                line1=line1,
                line2=line2,
                inclination=inclination,
                raan=raan,
                eccentricity=eccentricity_raw,
                arg_of_perigee=arg_of_perigee,
                mean_anomaly=mean_anomaly,
                mean_motion=mean_motion
            )
            return tle
            
        except Exception as e:
            print(f"  Unexpected error parsing {name}: {e}")
            return None


# ============================================================================
# CELESTRAK CLIENT
# ============================================================================

class CelestrackClient:
    """Fetch and parse TLEs from Celestrak"""

    BASE_URL = "https://celestrak.org/NORAD/elements/gp.php"
    TIMEOUT = 10
    
    @classmethod
    def fetch_group(cls, group: str = "last-30-days", format_type: str = "tle") -> List[TLE]:
        """Fetch TLE group from Celestrak."""
        params = {"GROUP": group, "FORMAT": format_type}
        
        try:
            response = requests.get(cls.BASE_URL, params=params, timeout=cls.TIMEOUT)
            response.raise_for_status()
            
            print(f"[Celestrak] Status: {response.status_code}")
            print(f"[Celestrak] Response Length: {len(response.text)} chars")
            
            if format_type == "json":
                return cls._parse_json(response.json())
            else:
                return cls._parse_tle_text(response.text)
                
        except requests.RequestException as e:
            print(f"[ERROR] Celestrak fetch failed: {e}")
            return []
    
    @staticmethod
    def _parse_tle_text(text: str) -> List[TLE]:
        """Parse TLE text format (3-line format: NAME, LINE1, LINE2)"""
        lines = text.strip().split('\n')
        print(f"[Parser] Total lines: {len(lines)}")
        
        tles = []
        i = 0
        while i < len(lines) - 1:
            if i + 2 >= len(lines):
                break
            
            name = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()
            
            if not name or not line1 or not line2:
                i += 3
                continue
            
            print(f"  [{i//3 + 1}] Parsing: {name[:30]:<30} | L1[0]={line1[0] if line1 else '?'} L2[0]={line2[0] if line2 else '?'}")
            
            tle = TLEParser.parse_lines(name, line1, line2)
            if tle:
                tles.append(tle)
                print(f"      ✓ Success: {tle.norad_cat_id}")
            else:
                print(f"      ✗ Failed")
            
            i += 3
        
        print(f"\n[Parser] Total TLEs successfully parsed: {len(tles)}/{(len(lines)//3)}")
        return tles

    @staticmethod
    def _parse_json(data: Dict) -> List[TLE]:
        """Parse JSON format from Celestrak"""
        tles = []

        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        
        for record in (data if isinstance(data, list) else []):
            try:
                tle = TLE(
                    object_name=record.get("OBJECT_NAME", ""),
                    norad_cat_id=record.get("NORAD_CAT_ID", ""),
                    epoch_year=int(record.get("EPOCH", "00")[:2]),
                    epoch_day=float(record.get("EPOCH", "0.0")[3:]),
                    bstar=float(record.get("BSTAR", 0)),
                    mean_motion_dot=float(record.get("MEAN_MOTION_DOT", 0)),
                    mean_motion_ddot=float(record.get("MEAN_MOTION_DDOT", 0)),
                    element_number=int(record.get("ELEMENT_NUMBER", 0)),
                    line1=record.get("LINE1", ""),
                    line2=record.get("LINE2", ""),
                    inclination=float(record.get("INCLINATION", 0)),
                    raan=float(record.get("RA_OF_ASC_NODE", 0)),
                    eccentricity=float(record.get("ECCENTRICITY", 0)),
                    arg_of_perigee=float(record.get("ARG_OF_PERICENTER", 0)),
                    mean_anomaly=float(record.get("MEAN_ANOMALY", 0)),
                    mean_motion=float(record.get("MEAN_MOTION", 0)),
                )
                tles.append(tle)
            except (ValueError, KeyError):
                continue
                
        return tles


# ============================================================================
# DATABASE LAYER
# ============================================================================

class TLEDatabase:
    """SQLite storage for TLE records with caching"""

    def __init__(self, db_path: str = "ssa_data.db"):
        self.db_path = db_path
        self._init_schema()
    
    def _init_schema(self):
        """Create database schema and upgrade it if needed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tle (
                id INTEGER PRIMARY KEY,
                norad_cat_id TEXT UNIQUE,
                object_name TEXT,
                epoch TIMESTAMP,
                inclination REAL,
                raan REAL,
                eccentricity REAL,
                arg_of_perigee REAL,
                mean_anomaly REAL,
                mean_motion REAL,
                bstar REAL,
                mean_motion_dot REAL,
                mean_motion_ddot REAL,
                element_number INTEGER,
                line1 TEXT,
                line2 TEXT,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Upgrade existing tables by adding missing columns safely
        columns_to_add = [
            ("epoch", "TIMESTAMP"),
            ("inclination", "REAL"),
            ("raan", "REAL"),
            ("eccentricity", "REAL"),
            ("arg_of_perigee", "REAL"),
            ("mean_anomaly", "REAL"),
            ("mean_motion", "REAL"),
            ("bstar", "REAL"),
            ("mean_motion_dot", "REAL"),
            ("mean_motion_ddot", "REAL"),
            ("element_number", "INTEGER"),
            ("line1", "TEXT"),
            ("line2", "TEXT"),
            ("fetched_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        ]

        for col_name, col_type in columns_to_add:
            try:
                cursor.execute(f"ALTER TABLE tle ADD COLUMN {col_name} {col_type}")
                print(f"[DB] Added missing column: {col_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    print(f"[DB] Warning adding {col_name}: {e}")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"[DB] Schema initialized at: {self.db_path}")

    def insert_tle(self, tle: TLE, replace: bool = True):
        """Insert or update TLE record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            query = f"""
                {'REPLACE' if replace else 'INSERT'} INTO tle 
                (norad_cat_id, object_name, epoch, inclination, raan, 
                 eccentricity, arg_of_perigee, mean_anomaly, mean_motion, 
                 bstar, mean_motion_dot, mean_motion_ddot, element_number, line1, line2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(query, (
                tle.norad_cat_id, tle.object_name, tle.epoch.isoformat(),
                tle.inclination, tle.raan, tle.eccentricity,
                tle.arg_of_perigee, tle.mean_anomaly, tle.mean_motion,
                tle.bstar, tle.mean_motion_dot, tle.mean_motion_ddot,
                tle.element_number, tle.line1, tle.line2
            ))
            conn.commit()
        except sqlite3.IntegrityError as e:
            print(f"[DB] IntegrityError for {tle.norad_cat_id}: {e}")
        finally:
            conn.close()

    def insert_batch(self, tles: List[TLE], replace: bool = True):
        """Batch insert TLE records"""
        print(f"\n[DB] Inserting {len(tles)} TLEs...")
        for i, tle in enumerate(tles):
            self.insert_tle(tle, replace=replace)
            if (i + 1) % 100 == 0:
                print(f"  [{i + 1}/{len(tles)}]")
        print(f"[DB] Insertion complete!")

    def get_tle(self, norad_cat_id: str) -> Optional[TLE]:
        """Retrieve TLE by NORAD catalog ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try: 
            cursor.execute("SELECT * FROM tle WHERE norad_cat_id = ?", (norad_cat_id,))
            row = cursor.fetchone()
            if not row:
                return None
            
            cols = ['id', 'norad_cat_id', 'object_name', 'epoch', 'inclination', 
                    'raan', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 
                    'mean_motion', 'bstar', 'mean_motion_dot', 'mean_motion_ddot',
                    'element_number', 'line1', 'line2', 'fetched_at']

            data = dict(zip(cols, row))
            
            # Handle both string and datetime returned by SQLite
            epoch_val = data['epoch']
            if isinstance(epoch_val, str):
                epoch_dt = datetime.fromisoformat(epoch_val.replace('Z', '+00:00'))
            elif isinstance(epoch_val, datetime):
                epoch_dt = epoch_val
            else:
                return None
            
            epoch_year = epoch_dt.year % 100
            epoch_day = epoch_dt.timetuple().tm_yday + (epoch_dt.hour * 3600 + epoch_dt.minute * 60 + epoch_dt.second) / 86400.0
            
            return TLE(
                object_name=data['object_name'],
                norad_cat_id=data['norad_cat_id'],
                epoch_year=epoch_year,
                epoch_day=epoch_day,
                bstar=data['bstar'],
                mean_motion_dot=data['mean_motion_dot'],
                mean_motion_ddot=data['mean_motion_ddot'],
                element_number=data['element_number'],
                line1=data['line1'],
                line2=data['line2'],
                inclination=data['inclination'],
                raan=data['raan'],
                eccentricity=data['eccentricity'],
                arg_of_perigee=data['arg_of_perigee'],
                mean_anomaly=data['mean_anomaly'],
                mean_motion=data['mean_motion']
            )
        finally:
            conn.close()

    def get_all_tles(self, limit: Optional[int] = None) -> List[TLE]:
        """Retrieve all TLE records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            query = "SELECT * FROM tle"
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            cols = ['id', 'norad_cat_id', 'object_name', 'epoch', 'inclination', 
                    'raan', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 
                    'mean_motion', 'bstar', 'mean_motion_dot', 'mean_motion_ddot',
                    'element_number', 'line1', 'line2', 'fetched_at']
            
            tles = []
            for row in rows:
                data = dict(zip(cols, row))
                
                # Handle both string and datetime
                epoch_val = data['epoch']
                if isinstance(epoch_val, str):
                    epoch_dt = datetime.fromisoformat(epoch_val.replace('Z', '+00:00'))
                elif isinstance(epoch_val, datetime):
                    epoch_dt = epoch_val
                else:
                    continue
                
                epoch_year = epoch_dt.year % 100
                epoch_day = epoch_dt.timetuple().tm_yday + (epoch_dt.hour * 3600 + epoch_dt.minute * 60 + epoch_dt.second) / 86400.0
                
                tles.append(TLE(
                    object_name=data['object_name'],
                    norad_cat_id=data['norad_cat_id'],
                    epoch_year=epoch_year,
                    epoch_day=epoch_day,
                    bstar=data['bstar'],
                    mean_motion_dot=data['mean_motion_dot'],
                    mean_motion_ddot=data['mean_motion_ddot'],
                    element_number=data['element_number'],
                    line1=data['line1'],
                    line2=data['line2'],
                    inclination=data['inclination'],
                    raan=data['raan'],
                    eccentricity=data['eccentricity'],
                    arg_of_perigee=data['arg_of_perigee'],
                    mean_anomaly=data['mean_anomaly'],
                    mean_motion=data['mean_motion']
                ))
                        
            return tles
        finally:
            conn.close()

    def set_cache(self, key: str, value: str):
        """Set cache value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO cache_metadata (key, value) VALUES (?, ?)",
                (key, value)
            )
            conn.commit()
        finally:
            conn.close()

    def get_cache(self, key: str) -> Optional[str]:
        """Get cache value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT value FROM cache_metadata WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def cache_is_fresh(self, key: str, max_age_hours: int = 24) -> bool:
        """Check if cached data is fresh"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                "SELECT updated_at FROM cache_metadata WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            if not row:
                return False

            updated_at = datetime.fromisoformat(row[0])
            age_hours = (datetime.now() - updated_at).total_seconds() / 3600
            return age_hours < max_age_hours
        finally:
            conn.close()

    def count_tles(self) -> int:
        """Count TLE records in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT COUNT(*) FROM tle")
            return cursor.fetchone()[0]
        finally:
            conn.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Use absolute path so DB is always saved next to the script
    project_dir = Path(__file__).parent.absolute()
    db_path = str(project_dir / "ssa_data.db")
    
    # Initialize database
    db = TLEDatabase(db_path=db_path)
    print(f"[Init] Database: {db_path}")
    
    # Fetch live TLEs from Celestrak
    print("\n[Main] Fetching TLEs from Celestrak...")
    tles = CelestrackClient.fetch_group(group="last-30-days")
    print(f"[Main] Fetched {len(tles)} TLE records")
    
    if tles:
        # Store in database
        db.insert_batch(tles)
        
        # Query
        count = db.count_tles()
        print(f"\n[DB] Database contains {count} TLE records total")

        # Retrieve and display first 5
        print("\n[Display] First 5 TLEs:")
        all_tles = db.get_all_tles(limit=5)
        for tle in all_tles:
            print(f"\n  {tle.object_name} ({tle.norad_cat_id})")
            print(f"    Epoch: {tle.epoch}")
            print(f"    Inclination: {tle.inclination:.2f}°")
            print(f"    Mean Motion: {tle.mean_motion:.5f} rev/day")
            print(f"    Eccentricity: {tle.eccentricity:.6f}")
            print(f"    Mean Motion DDOT: {tle.mean_motion_ddot}")
    else:
        print("[ERROR] No TLEs were parsed. Check network/Celestrak availability.")