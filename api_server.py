"""
SSA REST API
FastAPI-based production server for orbital queries and conjunction detection
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import json
import os
import asyncio
from pathlib import Path
import logging
 
from parser_pipeline import CelestrackClient, TLEDatabase, TLE

from orbit_mech_engine import OrbitalElement, SGP4Propagator
from conjunction_detection import ConjunctionDetector, ConjunctionSearch, ConjunctionEvent
from Analytics import OrbitalAnalytics, ExportUtils



# ============================================================================
# SETUP
# ============================================================================
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
app = FastAPI(
    title="Space Situational Awareness API",
    description="Production-grade SSA system: orbital mechanics, TLE data, conjunction detection",
    version="1.0.0"
)
 
db = TLEDatabase()
 
# In-memory cache for expensive computations
_conjunction_cache = {}
_analytics_cache = {}






# ============================================================================
# PYDANTIC MODELS
# ============================================================================
 
class StateVectorResponse(BaseModel):
    """Cartesian state vector response"""
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    altitude: float
    speed: float
    range: float
 
 
class OrbitalElementResponse(BaseModel):
    """Orbital element response"""
    a: float  # Semi-major axis (km)
    e: float  # Eccentricity
    i: float  # Inclination (degrees)
    omega: float  # Argument of perigee (degrees)
    Omega: float  # RAAN (degrees)
    M: float  # Mean anomaly (degrees)
    period_hours: float
    apogee_km: float
    perigee_km: float
 
 
class TLEResponse(BaseModel):
    """TLE record response"""
    norad_cat_id: str
    object_name: str
    epoch: str
    inclination: float
    raan: float
    eccentricity: float
    arg_of_perigee: float
    mean_anomaly: float
    mean_motion: float
 
 
class ConjunctionResponse(BaseModel):
    """Conjunction event response"""
    sat1_id: str
    sat1_name: str
    sat2_id: str
    sat2_name: str
    tca: str
    min_distance_km: float
    probability_of_collision: float
    risk_level: str
 
 
class CatalogStatsResponse(BaseModel):
    """Catalog statistics response"""
    total_objects: int
    apogee_km: Dict
    perigee_km: Dict
    period_hours: Dict
    inclination_deg: Dict
 

# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================
 
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation"""
    return """
    <html>
        <head>
            <title>SSA API</title>
            <style>
                body { font-family: Arial; margin: 40px; background: #f5f5f5; }
                .container { max-width: 900px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
                h1 { color: #333; border-bottom: 3px solid #2196F3; padding-bottom: 10px; }
                h2 { color: #2196F3; margin-top: 30px; }
                .endpoint { background: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #2196F3; }
                code { background: #eee; padding: 2px 6px; border-radius: 3px; font-family: monospace; }
                a { color: #2196F3; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🛰️ Space Situational Awareness API</h1>
                <p>Production-grade SSA system for orbital mechanics, TLE data processing, and conjunction detection.</p>
                
                <h2>📚 Endpoints</h2>
                
                <div class="endpoint">
                    <strong>TLE Management</strong>
                    <ul>
                        <li><code>GET /api/tles</code> - List all TLE records</li>
                        <li><code>GET /api/tle/{norad_id}</code> - Get specific TLE</li>
                        <li><code>POST /api/update-tles</code> - Fetch latest from Celestrak</li>
                    </ul>
                </div>
                
                <div class="endpoint">
                    <strong>Orbital Propagation</strong>
                    <ul>
                        <li><code>GET /api/propagate/{norad_id}</code> - Propagate satellite state</li>
                        <li><code>GET /api/orbital-elements/{norad_id}</code> - Get orbital elements</li>
                    </ul>
                </div>
                
                <div class="endpoint">
                    <strong>Conjunction Detection</strong>
                    <ul>
                        <li><code>GET /api/conjunctions</code> - Search for conjunctions</li>
                        <li><code>GET /api/conjunction-pair/{id1}/{id2}</code> - Assess specific pair</li>
                    </ul>
                </div>
                
                <div class="endpoint">
                    <strong>Analytics</strong>
                    <ul>
                        <li><code>GET /api/statistics</code> - Catalog statistics</li>
                        <li><code>GET /api/export/conjunctions</code> - Export results (CSV/JSON)</li>
                    </ul>
                </div>
                
                <h2>📖 Full Documentation</h2>
                <p>Interactive API docs: <a href="/docs">/docs</a></p>
                <p>Alternative docs: <a href="/redoc">/redoc</a></p>
            </div>
        </body>
    </html>
    """
 
 
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": {
            "tles_count": db.count_tles()
        }
    }
 



 # ============================================================================
# TLE ENDPOINTS
# ============================================================================
 
@app.get("/api/tles", response_model=List[TLEResponse])
async def list_tles(limit: int = Query(100, ge=1, le=5000)):
    """List TLE records from database"""
    tles = db.get_all_tles(limit=limit)
    
    return [
        TLEResponse(
            norad_cat_id=tle.norad_cat_id,
            object_name=tle.object_name,
            epoch=tle.epoch.isoformat(),
            inclination=tle.inclination,
            raan=tle.raan,
            eccentricity=tle.eccentricity,
            arg_of_perigee=tle.arg_of_perigee,
            mean_anomaly=tle.mean_anomaly,
            mean_motion=tle.mean_motion
        )
        for tle in tles
    ]
 




@app.get("/api/tle/{norad_id}", response_model=TLEResponse)
async def get_tle(norad_id: str):
    """Get specific TLE by NORAD catalog ID"""
    tle = db.get_tle(norad_id)
    
    if not tle:
        raise HTTPException(status_code=404, detail=f"TLE {norad_id} not found")
    
    return TLEResponse(
        norad_cat_id=tle.norad_cat_id,
        object_name=tle.object_name,
        epoch=tle.epoch.isoformat(),
        inclination=tle.inclination,
        raan=tle.raan,
        eccentricity=tle.eccentricity,
        arg_of_perigee=tle.arg_of_perigee,
        mean_anomaly=tle.mean_anomaly,
        mean_motion=tle.mean_motion
    )
 
 
@app.post("/api/update-tles")
async def update_tles(background_tasks: BackgroundTasks):
    """Fetch latest TLEs from Celestrak (background task)"""
    
    def fetch_and_store():
        logger.info("Fetching TLEs from Celestrak...")
        tles = CelestrackClient.fetch_group(group="last-30-days")
        logger.info(f"Fetched {len(tles)} TLE records")
        
        db.insert_batch(tles, replace=True)
        db.set_cache("tle_update_time", datetime.utcnow().isoformat())
        logger.info("TLE database updated")
    
    background_tasks.add_task(fetch_and_store)
    
    return {"status": "update started", "message": "TLEs will be fetched in the background"}
 
 
# ============================================================================
# ORBITAL PROPAGATION ENDPOINTS
# ============================================================================

@app.get("/api/propagate/{norad_id}", response_model=StateVectorResponse)
async def propagate_satellite(
    norad_id: str,
    hours_ahead: float = Query(24, ge=0, le=7*24)
):
    """Propagate satellite state vector"""
    tle = db.get_tle(norad_id)
    
    if not tle:
        raise HTTPException(status_code=404, detail=f"TLE {norad_id} not found")
    
    # Propagate
    oe = tle.to_orbital_element()
    prop = SGP4Propagator(oe)
    
    t_prop = oe.epoch + timedelta(hours=hours_ahead)
    state = prop.propagate(t_prop)
    
    return StateVectorResponse(
        x=state.r[0],
        y=state.r[1],
        z=state.r[2],
        vx=state.v[0],
        vy=state.v[1],
        vz=state.v[2],
        altitude=state.altitude,
        speed=state.velocity_magnitude,
        range=state.position_magnitude
    )
 
 
@app.get("/api/orbital-elements/{norad_id}", response_model=OrbitalElementResponse)
async def get_orbital_elements(norad_id: str):
    """Get computed orbital elements"""
    tle = db.get_tle(norad_id)
    
    if not tle:
        raise HTTPException(status_code=404, detail=f"TLE {norad_id} not found")
    
    oe = tle.to_orbital_element()
    apogee, perigee = OrbitalAnalytics.compute_apogee_perigee(oe)
    
    import math
    return OrbitalElementResponse(
        a=oe.a,
        e=oe.e,
        i=math.degrees(oe.i),
        omega=math.degrees(oe.omega),
        Omega=math.degrees(oe.Omega),
        M=math.degrees(oe.M),
        period_hours=OrbitalAnalytics.orbital_period_hours(oe),
        apogee_km=apogee,
        perigee_km=perigee
    )
 


  
# ============================================================================
# CONJUNCTION DETECTION ENDPOINTS
# ============================================================================
 
@app.get("/api/conjunctions", response_model=List[ConjunctionResponse])
async def search_conjunctions(
    time_horizon_days: int = Query(7, ge=1, le=30),
    limit: int = Query(100, ge=1, le=1000),
    search_step_hours: float = Query(2.0, ge=0.5, le=12)
):
    """Search for conjunctions across catalog"""
    
    # Load catalog
    all_tles = db.get_all_tles()
    
    if len(all_tles) < 2:
        raise HTTPException(status_code=400, detail="Insufficient TLEs in database")
    
    orbital_elements = {}
    for tle in all_tles:
        orbital_elements[tle.norad_cat_id] = (tle.to_orbital_element(), tle.object_name)
    
    # Search
    detector = ConjunctionDetector(
        time_horizon_days=time_horizon_days,
        search_step_hours=search_step_hours
    )
    search = ConjunctionSearch(detector)
    conjunctions = search.search_catalog(orbital_elements)
    
    # Return top results
    return [
        ConjunctionResponse(
            sat1_id=c.sat1_id,
            sat1_name=c.sat1_name,
            sat2_id=c.sat2_id,
            sat2_name=c.sat2_name,
            tca=c.tca.isoformat(),
            min_distance_km=c.min_distance,
            probability_of_collision=c.probability_of_collision,
            risk_level=c.risk_level
        )
        for c in conjunctions[:limit]
    ]
 
 
@app.get("/api/conjunction-pair/{id1}/{id2}")
async def assess_conjunction_pair(id1: str, id2: str):
    """Assess collision risk for specific satellite pair"""
    
    tle1 = db.get_tle(id1)
    tle2 = db.get_tle(id2)
    
    if not tle1 or not tle2:
        raise HTTPException(status_code=404, detail="One or both TLEs not found")
    
    detector = ConjunctionDetector(time_horizon_days=7, search_step_hours=1.0)
    
    event = detector.assess_pair(
        tle1.to_orbital_element(),
        tle2.to_orbital_element(),
        tle1.object_name,
        tle2.object_name,
        id1, id2
    )
    
    if not event:
        return {"status": "no_risk", "message": "No conjunction event detected"}
    
    return event.to_dict()



# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================
 
@app.get("/api/statistics")
async def catalog_statistics():
    """Get aggregate catalog statistics"""
    
    all_tles = db.get_all_tles()
    
    orbital_elements = {}
    for tle in all_tles:
        orbital_elements[tle.norad_cat_id] = (tle.to_orbital_element(), tle.object_name)
    
    stats = OrbitalAnalytics.catalog_statistics(orbital_elements)
    
    return {
        "computed_at": datetime.utcnow().isoformat(),
        "statistics": stats
    }
 


 # ============================================================================
# EXPORT ENDPOINTS
# ============================================================================
 
@app.get("/api/export/conjunctions")
async def export_conjunctions(format: str = Query("json", regex="^(json|csv)$")):
    """Export conjunction results"""
    
    # Get conjunctions
    all_tles = db.get_all_tles()
    
    orbital_elements = {}
    for tle in all_tles:
        orbital_elements[tle.norad_cat_id] = (tle.to_orbital_element(), tle.object_name)
    
    detector = ConjunctionDetector(time_horizon_days=3, search_step_hours=2.0)
    search = ConjunctionSearch(detector)
    conjunctions = search.search_catalog(orbital_elements)
    
    # Export
    output_dir = Path("/tmp")
    if format == "json":
        filename = output_dir / "conjunctions.json"
        ExportUtils.export_conjunctions_json(conjunctions, str(filename))
        media_type = "application/json"
    else:
        filename = output_dir / "conjunctions.csv"
        ExportUtils.export_conjunctions_csv(conjunctions, str(filename))
        media_type = "text/csv"
    
    return FileResponse(
        path=filename,
        media_type=media_type,
        filename=filename.name
    )
 
 
# ============================================================================
# EXAMPLE USAGE
# ============================================================================
 
if __name__ == "__main__":
    import uvicorn
    
    # Initialize database with sample data if empty
    if db.count_tles() == 0:
        print("Database empty. Fetching initial TLE data...")
        tles = CelestrackClient.fetch_group(group="last-30-days")
        db.insert_batch(tles)
        print(f"Loaded {len(tles)} TLEs")
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
 






