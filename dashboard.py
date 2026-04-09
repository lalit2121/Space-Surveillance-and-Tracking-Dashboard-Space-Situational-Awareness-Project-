
"""
SSA Streamlit Dashboard
Interactive web interface for orbital analysis, conjunction detection, and analytics
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import streamlit.components.v1 as components
from pathlib import Path
DB_PATH = str(Path(__file__).parent / "ssa_data.db")  #  for path 
def get_db(modules):
    return modules['TLEDatabase'](db_path=DB_PATH)
@st.cache_data(ttl=300) # cache for 5 minutes
def load_orbital_elements(_modules, db_path):
    db = _modules['TLEDatabase'](db_path=db_path)
    all_tles = db.get_all_tles()
    orbital_elements = {}
    for tle in all_tles:
        try:
            orbital_elements[tle.norad_cat_id] = (tle.to_orbital_element(), tle.object_name)
        except Exception:
            continue
    return orbital_elements
@st.cache_data(ttl=300)
def load_tles(_modules, db_path, limit=315):
    db = _modules['TLEDatabase'](db_path=db_path)
    return db.get_all_tles(limit=limit)
# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="SSA Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ============================================================================
# IMPORTS (lazy to avoid import errors)
# ============================================================================
def load_modules():
    try:
        from parser_pipeline import CelestrackClient, TLEDatabase, TLE
        from orbit_mech_engine import SGP4Propagator
        from conjunction_detection import ConjunctionDetector, ConjunctionSearch
        from Analytics import OrbitalAnalytics, OrbitalVisualizer, RiskVisualizer, ExportUtils
        return {
            'CelestrakClient': CelestrackClient,
            'TLEDatabase': TLEDatabase,
            'SGP4Propagator': SGP4Propagator,
            'ConjunctionDetector': ConjunctionDetector,
            'ConjunctionSearch': ConjunctionSearch,
            'OrbitalAnalytics': OrbitalAnalytics,
            'OrbitalVisualizer': OrbitalVisualizer,
            'RiskVisualizer': RiskVisualizer,
            'ExportUtils': ExportUtils,
        }
    except ImportError as e:
        st.error(f"Import error: {e}")
        return None
# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("# 🛰️ SSA Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Conjunction Analysis", "Analytics", "Data Management"],
    index=0
)
# ============================================================================
# PAGE: DASHBOARD (HOME)
# ============================================================================
if page == "Dashboard":
    st.title("🛰️ Space Situational Awareness Dashboard")
   
    col1, col2, col3 = st.columns(3)
   
    modules = load_modules()
    if modules:
        # FIXED — absolute path to your existing database
        db = modules['TLEDatabase'](db_path=r"C:\Users\Lkd\Desktop\proj\ssa_data.db")
       
        with col1:
            count = db.count_tles()
            st.metric("Objects in Catalog", count)
       
        with col2:
            st.metric("Last Update", "Today")
       
        with col3:
            st.metric("API Status", "🟢 Healthy")
   
    st.markdown("---")
   
    st.markdown("---")
    # ── ANIMATED GROUND TRACK MAP ─────────────────────────────────────────────
    st.subheader("🌍 Live Ground Track Map")
    import plotly.graph_objects as go
    orbital_elements = load_orbital_elements(modules, DB_PATH)
    TRACK_COLORS = [
        '#00d4ff', '#00ff88', '#ff6b35', '#ffd700', '#ff3355',
        '#a78bfa', '#34d399', '#f97316', '#60a5fa', '#fb7185'
    ]
    # ── Satellite selector ────────────────────────────────────────────────────
    all_tles_list = load_tles(modules, DB_PATH, limit=315)
    sat_options = [t.object_name for t in all_tles_list]
    selected_sats = st.multiselect(
        "🔍 Select satellites to track (up to 10):",
        options=sat_options,
        default=sat_options[:10],
        max_selections=10,
        key="dashboard_sat_select"
    )
    if not selected_sats:
        selected_sats = sat_options[:10]
    # Filter orbital_elements to selection
    selected_ids_set = set(selected_sats)
    selected_oe = {
        obj_id: (oe, name)
        for obj_id, (oe, name) in orbital_elements.items()
        if name in selected_ids_set
    }
    if not selected_oe:
        selected_oe = dict(list(orbital_elements.items())[:10])
    ANIM_STEPS = 120
    all_tracks = {}
    from orbit_mech_engine import SGP4Propagator, CoordinateTransform
    for idx, (obj_id, (oe, name)) in enumerate(list(selected_oe.items())[:10]):
        try:
            prop = SGP4Propagator(oe)
            times = [oe.epoch + timedelta(hours=i)
                     for i in np.linspace(0, 6, ANIM_STEPS)]
            lats, lons = [], []
            for t in times:
                state = prop.propagate(t)
                lat, lon, _ = CoordinateTransform.eci_to_geographic(state.r, t)
                lats.append(lat)
                lons.append(lon)
            color = TRACK_COLORS[idx % len(TRACK_COLORS)]
            all_tracks[obj_id] = (lats, lons, name, color)
        except Exception:
            continue
    ids = list(all_tracks.keys())
    n_tracks = len(all_tracks)
    if all_tracks:
        col_map, col_legend = st.columns([5, 1])
        fig_map = go.Figure()
        fig_map.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            height=460,
            geo=dict(
                projection_type='equirectangular',
                showland=True, landcolor='#0d1a2e',
                showocean=True, oceancolor='#070b14',
                showlakes=False,
                showcountries=True, countrycolor='#1e3a5f',
                showcoastlines=True, coastlinecolor='#1e3a5f',
                bgcolor='#070b14',
                lataxis=dict(gridcolor='#1e3a5f', showgrid=True),
                lonaxis=dict(gridcolor='#1e3a5f', showgrid=True),
            ),
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        # Static orbit lines
        for obj_id, (lats, lons, name, color) in all_tracks.items():
            fig_map.add_trace(go.Scattergeo(
                lat=lats, lon=lons,
                mode='lines',
                line=dict(color=color, width=1),
                opacity=0.55,
                hoverinfo='skip',
                showlegend=False,
            ))
        # Initial dot positions — emoji satellite icon
        for obj_id, (lats, lons, name, color) in all_tracks.items():
            fig_map.add_trace(go.Scattergeo(
                lat=[lats[0]], lon=[lons[0]],
                mode='text',
                text=['🛰️'],
                textfont=dict(size=16),
                textposition='middle center',
                showlegend=False,
                hovertemplate=f'<b>{name}</b><br>CURRENT POSITION<extra></extra>',
            ))
        # Animation frames
        frames = []
        for step in range(ANIM_STEPS):
            frame_data = []
            for obj_id, (lats, lons, name, color) in all_tracks.items():
                frame_data.append(go.Scattergeo(
                    lat=lats, lon=lons,
                    mode='lines',
                    line=dict(color=color, width=1),
                    opacity=0.55,
                    hoverinfo='skip',
                    showlegend=False,
                ))
            for obj_id, (lats, lons, name, color) in all_tracks.items():
                frame_data.append(go.Scattergeo(
                    lat=[lats[step]], lon=[lons[step]],
                    mode='text',
                    text=['🛰️'],
                    textfont=dict(size=16),
                    textposition='middle center',
                ))
            frames.append(go.Frame(data=frame_data, name=str(step)))
        fig_map.frames = frames
        fig_map.update_layout(
            updatemenus=[dict(
                type='buttons', showactive=False,
                y=0.02, x=0.01, xanchor='left', yanchor='bottom',
                buttons=[
                    dict(label='▶ PLAY', method='animate',
                         args=[None, dict(
                             frame=dict(duration=600, redraw=True),
                             fromcurrent=True,
                             transition=dict(duration=0),
                             mode='immediate', loop=True,
                         )]),
                    dict(label='⏹ STOP', method='animate',
                         args=[[None], dict(
                             frame=dict(duration=0, redraw=False),
                             mode='immediate',
                             transition=dict(duration=0),
                         )]),
                ]
            )],
            sliders=[dict(
                active=0,
                currentvalue=dict(prefix='T+', visible=True, xanchor='right'),
                pad=dict(t=10, b=5),
                steps=[dict(
                    args=[[str(k)], dict(frame=dict(duration=0, redraw=True),
                        mode='immediate', transition=dict(duration=0))],
                    label='', method='animate',
                ) for k in range(ANIM_STEPS)]
            )]
        )
        with col_map:
            st.plotly_chart(fig_map, use_container_width=True,
                            config={'displayModeBar': False})
        with col_legend:
            st.markdown("**Active Tracks**")
            for obj_id in ids:
                _, __, name, color = all_tracks[obj_id]
                st.markdown(
                    f'<div style="border-left:3px solid {color};'
                    f'padding-left:8px;margin-bottom:4px;'
                    f'font-size:0.7rem;color:#e8f4f8;">{name[:18]}</div>',
                    unsafe_allow_html=True
                )
    else:
        st.info("No orbital data available for ground track map.")
  # ── TELEMETRY ROW ─────────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("### 📡 TELEMETRY — LIVE ORBITAL DATA")
    if ids:
        cols = st.columns(len(ids))
        for idx, obj_id in enumerate(ids):
            lats, lons, name, color = all_tracks[obj_id]
            oe, _ = orbital_elements[obj_id]
            try:
                from Analytics import OrbitalAnalytics
                import math as _math
                apogee, perigee = OrbitalAnalytics.compute_apogee_perigee(oe)
                period = OrbitalAnalytics.orbital_period_hours(oe)
                inc = _math.degrees(oe.i)
                alt = (apogee + perigee) / 2
                with cols[idx]:
                    st.markdown(f"""
                    <div style="background:#0d1525;border:1px solid #1e3a5f;
                                border-top:2px solid {color};border-radius:3px;
                                padding:10px 12px;">
                        <div style="font-family:Share Tech Mono,monospace;color:{color};
                                    font-size:0.6rem;letter-spacing:0.1em;margin-bottom:8px;
                                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                            {name[:16]}
                        </div>
                        <div style="font-family:Share Tech Mono,monospace;
                                    font-size:0.58rem;line-height:2.0;">
                            <span style="color:#3a5a7a;">ALT </span>
                            <span style="color:#e8f4f8;">{alt:.0f} km</span><br>
                            <span style="color:#3a5a7a;">INC </span>
                            <span style="color:#e8f4f8;">{inc:.2f}°</span><br>
                            <span style="color:#3a5a7a;">PER </span>
                            <span style="color:#e8f4f8;">{period:.3f} hr</span><br>
                            <span style="color:#3a5a7a;">APO </span>
                            <span style="color:#e8f4f8;">{apogee:.0f} km</span><br>
                            <span style="color:#3a5a7a;">PEG </span>
                            <span style="color:#e8f4f8;">{perigee:.0f} km</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception:
                continue
    # ── BOTTOM ROW — catalog table + orbit distribution ───────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col_table, col_charts = st.columns([2, 1])
    with col_table:
        st.markdown("### 📋 CATALOG — RECENT OBJECTS")
        all_tles_bottom = load_tles(modules, DB_PATH, limit=15)
        df = pd.DataFrame([{
            'OBJECT': t.object_name,
            'NORAD ID': t.norad_cat_id,
            'EPOCH': t.epoch.strftime('%Y-%m-%d'),
            'INC (°)': f"{t.inclination:.2f}",
            'ECC': f"{t.eccentricity:.6f}",
            'MM (rev/day)': f"{t.mean_motion:.4f}",
        } for t in all_tles_bottom])
        st.dataframe(df, use_container_width=True, height=280, hide_index=True)
    with col_charts:
        st.markdown("### 📊 ORBIT DISTRIBUTION")
        import math as _math
        import plotly.graph_objects as go
        incs = [_math.degrees(oe.i) for oe, _ in orbital_elements.items()
                if hasattr(oe, 'i')]
        if not incs:
            incs = [_math.degrees(v[0].i) for v in orbital_elements.values()]
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=incs, nbinsx=15,
            marker_color='#00d4ff',
            marker_line_color='#070b14',
            marker_line_width=1,
            opacity=0.8,
        ))
        fig_dist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,21,37,0.8)',
            font=dict(color='#7a9bb5', size=11),
            height=280,
            title=dict(text='INCLINATION DIST.', font=dict(color='#e8f4f8', size=12)),
            showlegend=False,
            bargap=0.05,
            margin=dict(l=40, r=20, t=40, b=40),
            xaxis=dict(gridcolor='#1e3a5f', color='#7a9bb5'),
            yaxis=dict(gridcolor='#1e3a5f', color='#7a9bb5'),
        )
        st.plotly_chart(fig_dist, use_container_width=True,
                        config={'displayModeBar': False})
# ============================================================================
# PAGE: CONJUNCTION ANALYSIS
# ============================================================================
# ============================================================================
# PAGE: 3D GLOBE — CesiumJS
# ============================================================================
# ── CONTROLS ──────────────────────────────────────────────────────────────
    all_tles_3d = load_tles(modules, DB_PATH, limit=315)
    sat_opts_3d = [t.object_name for t in all_tles_3d]
    ctrl1, ctrl2, ctrl3 = st.columns([2, 5, 1])
    with ctrl1:
        prop_hours = st.slider("PROPAGATION (HRS)", 1, 72, 24, key="prop_hrs_3d")
    with ctrl2:
        selected = st.multiselect(
            "SELECT SATELLITES",
            options=sat_opts_3d,
            default=sat_opts_3d[:5],
            max_selections=10,
            key="sat_select_3d",
        )
    with ctrl3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.button("PLOT", type="primary", use_container_width=True, key="plot_3d")
    st.markdown("<hr style='border-color:#1e3a5f;margin:8px 0 12px 0;'>", unsafe_allow_html=True)
    TRACK_COLORS_3D = [
        '#00d4ff', '#00ff88', '#ff6b35', '#ffd700', '#ff3355',
        '#a78bfa', '#34d399', '#f97316', '#60a5fa', '#fb7185',
    ]
    selected_tles = [t for t in all_tles_3d if t.object_name in set(selected)]
    # ── COMPUTE ORBITS IN ECI FRAME (realistic inertial orbits) ───────────────
    import math
    orbit_data = []
    n_points = 120 # points per orbit track
    for idx, tle in enumerate(selected_tles):
        color_hex = TRACK_COLORS_3D[idx % len(TRACK_COLORS_3D)]
        try:
            from orbit_mech_engine import SGP4Propagator
            oe = tle.to_orbital_element()
            prop = SGP4Propagator(oe)
            # Span exactly one full orbital period for a closed ellipse
            period_hrs = (2 * math.pi / oe.n) / 3600 \
                if hasattr(oe, 'n') and oe.n > 0 \
                else prop_hours / len(selected_tles)
            span = min(period_hrs * 1.5, prop_hours)
            times = [oe.epoch + timedelta(hours=h)
                     for h in np.linspace(0, span, n_points)]
            # ECI positions (km) — inertial frame, no Earth rotation applied
            eci_positions = []
            geo_positions = []
            from orbit_mech_engine import CoordinateTransform
            for t in times:
                state = prop.propagate(t)
                rx, ry, rz = state.r[0], state.r[1], state.r[2]
                eci_positions.append([rx, ry, rz])
                # Also compute geodetic for telemetry
                jd = (t - datetime(2000, 1, 1, 12)).total_seconds()/86400 + 2451545.0
                gmst = math.fmod(
                    280.46061837 + 360.98564736629*(jd-2451545.0), 360.0
                ) * math.pi / 180.0
                ecef_x = rx*math.cos(-gmst) - ry*math.sin(-gmst)
                ecef_y = rx*math.sin(-gmst) + ry*math.cos(-gmst)
                ecef_z = rz
                r_norm = math.sqrt(ecef_x**2 + ecef_y**2 + ecef_z**2)
                geo_positions.append([
                    math.degrees(math.atan2(ecef_y, ecef_x)),
                    math.degrees(math.asin(ecef_z / r_norm)),
                    (r_norm - 6371.0) * 1000
                ])
            orbit_data.append({
                'name': tle.object_name,
                'color_hex': color_hex,
                'eci_positions': eci_positions, # for 3D globe
                'geo_positions': geo_positions, # for telemetry
            })
        except Exception as e:
            st.warning(f"Failed: {tle.object_name} — {e}")
            continue
    orbit_json = json.dumps(orbit_data)
   # ── THREE.JS HTML ──────────────────────────────────────────────────────────
    globe_html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    html, body {{ width:100%; height:100%; background:#070b14; overflow:hidden; }}
    canvas {{ display:block; }}
    #legend {{
      position:absolute; top:12px; right:12px;
      background:rgba(13,21,37,0.88);
      border:1px solid #1e3a5f; border-radius:4px;
      padding:10px 14px;
      font-family:'Share Tech Mono',monospace;
      font-size:11px; color:#7a9bb5; max-width:180px; z-index:999;
    }}
    #legend .title {{
      color:#00d4ff; letter-spacing:.15em;
      text-transform:uppercase; font-size:10px; margin-bottom:8px;
    }}
    .leg-item {{ display:flex; align-items:center; gap:8px; margin-bottom:5px; }}
    .leg-dot {{ width:10px; height:10px; border-radius:50%; flex-shrink:0; border:1px solid rgba(255,255,255,0.4); }}
    .leg-name {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:#e8f4f8; font-size:10px; }}
    #controls {{
      position:absolute; bottom:14px; left:50%; transform:translateX(-50%);
      display:flex; gap:8px; z-index:999;
    }}
    .ctrl-btn {{
      background:rgba(13,21,37,0.9); border:1px solid #00d4ff; color:#00d4ff;
      font-family:'Share Tech Mono',monospace; font-size:11px; letter-spacing:.1em;
      padding:6px 16px; cursor:pointer; border-radius:2px; text-transform:uppercase;
      transition: all .15s;
    }}
    .ctrl-btn:hover {{ background:#00d4ff; color:#070b14; }}
    #status {{
      position:absolute; top:12px; left:12px;
      background:rgba(13,21,37,0.88); border:1px solid #1e3a5f; border-radius:4px;
      padding:8px 12px; font-family:'Share Tech Mono',monospace;
      font-size:10px; color:#7a9bb5; z-index:999; line-height:1.8;
    }}
  </style>
</head>
<body>
<div id="status">
  <span style="color:#00d4ff;">SGP4</span> // THREE.JS GLOBE<br>
  <span style="color:#00ff88;" id="sat-count">0</span> SATELLITES TRACKED<br>
  <span style="color:#7a9bb5;" id="time-display"></span>
</div>
<div id="legend">
  <div class="title">ACTIVE TRACKS</div>
  <div id="leg-items"></div>
</div>
<div id="controls">
  <button class="ctrl-btn" onclick="resetCamera()">⌂ RESET</button>
  <button class="ctrl-btn" onclick="toggleRotate()" id="rotBtn">⟳ EARTH ROT ON</button>
  <button class="ctrl-btn" onclick="focusNext()">⊙ FOCUS SAT</button>
  <button class="ctrl-btn" onclick="toggleAnim()" id="animBtn">⏸ PAUSE</button>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const ORBIT_DATA = {orbit_json};
// ── Renderer ──────────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x020509);
const camera = new THREE.PerspectiveCamera(40, window.innerWidth/window.innerHeight, 10, 200000);
camera.position.set(0, 4000, 22000);
// ── Starfield ─────────────────────────────────────────────────────────────────
(function() {{
  const sg = new THREE.BufferGeometry();
  const sv = [];
  for (let i = 0; i < 6000; i++) {{
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const r = 90000 + Math.random() * 10000;
    sv.push(
      r * Math.sin(phi) * Math.cos(theta),
      r * Math.sin(phi) * Math.sin(theta),
      r * Math.cos(phi)
    );
  }}
  sg.setAttribute('position', new THREE.Float32BufferAttribute(sv, 3));
  scene.add(new THREE.Points(sg, new THREE.PointsMaterial({{
    color: 0xffffff, size: 60, sizeAttenuation: true,
    transparent: true, opacity: 0.8
  }})));
}})();
// ── Lighting ──────────────────────────────────────────────────────────────────
const sunLight = new THREE.DirectionalLight(0xfff5e0, 1.6);
sunLight.position.set(80000, 30000, 50000);
scene.add(sunLight);
const ambLight = new THREE.AmbientLight(0x111a2e, 1.0);
scene.add(ambLight);
// ── Earth texture via detailed canvas ────────────────────────────────────────
const TW = 4096, TH = 2048;
const landCanvas = document.createElement('canvas');
landCanvas.width = TW;
landCanvas.height = TH;
const ctx = landCanvas.getContext('2d');
function drawDetailedEarth(ctx, W, H) {{
  // Deep ocean gradient
  const oceanGrad = ctx.createLinearGradient(0, 0, 0, H);
  oceanGrad.addColorStop(0, '#0a1f35');
  oceanGrad.addColorStop(0.3, '#0d2d4a');
  oceanGrad.addColorStop(0.5, '#0e3358');
  oceanGrad.addColorStop(0.7, '#0d2d4a');
  oceanGrad.addColorStop(1, '#0a1f35');
  ctx.fillStyle = oceanGrad;
  ctx.fillRect(0, 0, W, H);
  // Shallow water tint near coasts (painted before land)
  ctx.fillStyle = 'rgba(15,80,120,0.3)';
  ctx.fillRect(0, 0, W, H);
  function ll2xy(lat, lon) {{
    return [(lon + 180) / 360 * W, (90 - lat) / 180 * H];
  }}
  function poly(pts, color, alpha) {{
    ctx.save();
    ctx.globalAlpha = alpha !== undefined ? alpha : 1.0;
    ctx.fillStyle = color;
    ctx.beginPath();
    const [x0, y0] = ll2xy(...pts[0]);
    ctx.moveTo(x0, y0);
    for (let i = 1; i < pts.length; i++) {{
      const [x, y] = ll2xy(...pts[i]);
      ctx.lineTo(x, y);
    }}
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }}
  // Land base color
  const landColor = '#2d5a27';
  const landHighlight= '#3a7a32';
  const desertColor = '#c4a862';
  const mountainColor= '#7a6b55';
  const snowColor = '#e8f0f8';
  const iceColor = '#ddeeff';
  // ── North America ──
  poly([[72,-140],[70,-130],[68,-120],[65,-110],[60,-95],[60,-85],[55,-83],[50,-57],[47,-53],[45,-60],[42,-66],[35,-76],[25,-80],[20,-87],[15,-83],[8,-77],[8,-83],[15,-90],[20,-105],[22,-110],[30,-117],[35,-120],[40,-124],[48,-124],[55,-130],[60,-138],[65,-140],[72,-140]], landColor);
  // Rocky mountains
  poly([[50,-120],[45,-118],[35,-118],[30,-110],[35,-107],[42,-108],[50,-114],[50,-120]], mountainColor);
  // Great plains / prairies
  poly([[52,-110],[45,-108],[38,-105],[35,-103],[45,-97],[52,-100],[52,-110]], '#8ba870');
  // Mexico / Central America
  poly([[22,-105],[18,-95],[15,-88],[8,-77],[8,-83],[15,-90],[20,-87],[22,-97],[25,-95],[28,-110],[22,-105]], landColor);
  // Greenland
  poly([[83,-45],[83,-18],[78,-15],[72,-22],[66,-37],[65,-52],[68,-55],[72,-54],[78,-50],[83,-45]], iceColor);
  // Alaska
  poly([[72,-168],[65,-168],[60,-165],[55,-163],[58,-152],[62,-150],[65,-162],[70,-160],[72,-168]], landColor);
  // ── South America ──
  poly([[12,-72],[10,-62],[5,-52],[0,-50],[-5,-35],[-10,-37],[-15,-39],[-23,-43],[-33,-53],[-40,-62],[-55,-68],[-55,-72],[-45,-75],[-35,-72],[-20,-70],[-5,-78],[5,-77],[10,-75],[12,-72]], landColor);
  // Amazon basin
  poly([[5,-60],[0,-50],[-5,-45],[-10,-42],[-15,-45],[-10,-60],[-5,-65],[0,-65],[5,-60]], '#1a5c18');
  // Andes
  poly([[-5,-78],[-15,-75],[-25,-70],[-35,-70],[-45,-73],[-40,-68],[-25,-65],[-15,-70],[-5,-75],[-5,-78]], mountainColor);
  // Patagonia
  poly([[-40,-65],[-45,-67],[-55,-68],[-55,-63],[-45,-63],[-40,-62],[-40,-65]], '#7a8060');
  // ── Europe ──
  poly([[71,28],[71,15],[65,14],[58,5],[51,2],[44,0],[36,5],[36,10],[38,16],[41,20],[38,24],[38,28],[42,35],[45,35],[48,38],[55,22],[60,25],[65,25],[71,28]], landColor);
  // Scandinavia detail
  poly([[71,15],[68,14],[65,14],[60,5],[58,5],[60,10],[62,14],[65,18],[68,18],[71,22],[71,15]], '#3a6b35');
  // Iberian Peninsula
  poly([[44,0],[36,0],[36,10],[44,3],[44,0]], landColor);
  // British Isles (approximate)
  poly([[58,-5],[51,-5],[50,2],[56,0],[58,-3],[58,-5]], '#3a7a32');
  poly([[58,-3],[55,-6],[58,-8],[60,-5],[58,-3]], '#3a7a32');
  // Alps
  poly([[48,8],[44,8],[44,14],[48,14],[48,8]], mountainColor);
  // ── Africa ──
  poly([[37,10],[37,38],[12,44],[11,42],[0,42],[-10,40],[-28,33],[-35,20],[-35,17],[-25,15],[-10,12],[0,8],[5,0],[0,-5],[5,-8],[10,-15],[15,-17],[20,-17],[25,-15],[30,-10],[35,10],[37,10]], landColor);
  // Sahara Desert
  poly([[37,10],[30,-5],[20,-15],[15,-17],[12,14],[10,22],[15,25],[20,30],[28,30],[35,25],[37,18],[37,10]], desertColor);
  // Congo rainforest
  poly([[5,15],[0,15],[-5,12],[-8,18],[-5,25],[0,28],[5,22],[8,18],[5,15]], '#1a5c18');
  // Kalahari
  poly([[-20,20],[-28,18],[-35,17],[-35,25],[-25,28],[-20,25],[-20,20]], '#c4a862', 0.7);
  // Madagascar
  poly([[-12,44],[-25,44],[-25,48],[-12,50],[-12,44]], landColor);
  // ── Asia ──
  poly([[72,30],[72,60],[72,100],[72,140],[72,180],[55,180],[45,150],[35,140],[22,120],[10,105],[5,100],[10,80],[20,70],[25,60],[28,50],[35,36],[45,40],[55,25],[65,25],[71,28],[72,30]], landColor);
  // Siberia tundra
  poly([[72,60],[65,60],[60,70],[60,100],[65,110],[72,110],[72,60]], '#4a6640');
  // Gobi / Central Asian deserts
  poly([[50,75],[40,55],[35,55],[40,80],[45,90],[50,95],[50,75]], desertColor);
  poly([[50,90],[42,80],[38,85],[40,100],[48,105],[50,95],[50,90]], desertColor, 0.7);
  // Himalayas
  poly([[30,72],[28,88],[28,98],[30,100],[35,80],[32,72],[30,72]], mountainColor);
  // Indian subcontinent
  poly([[28,68],[8,68],[8,80],[20,88],[22,92],[28,90],[28,68]], '#3a7030');
  // Southeast Asia peninsula
  poly([[22,98],[10,98],[5,103],[10,108],[15,100],[22,102],[22,98]], landColor);
  // Japan (approx)
  poly([[45,141],[38,141],[34,132],[35,130],[40,135],[45,141]], landColor);
  // Indonesia/Borneo (approx)
  poly([[8,115],[0,108],[-4,114],[-4,118],[4,118],[8,115]], landColor);
  poly([[4,108],[0,104],[-4,105],[-4,112],[2,114],[6,110],[4,108]], landColor);
  // Arabian Peninsula
  poly([[30,36],[12,44],[12,50],[22,60],[26,56],[30,48],[32,40],[30,36]], desertColor);
  // ── Australia ──
  poly([[-14,126],[-14,136],[-20,148],[-28,154],[-38,146],[-38,136],[-32,124],[-22,114],[-17,122],[-14,126]], landColor);
  // Outback desert
  poly([[-18,126],[-24,120],[-30,122],[-32,134],[-25,138],[-18,132],[-18,126]], desertColor);
  // New Zealand
  poly([[-34,172],[-46,168],[-46,170],[-34,174],[-34,172]], landColor);
  poly([[-40,175],[-46,168],[-44,166],[-38,175],[-40,175]], landColor);
  // ── Antarctica ──
  poly([[-70,-180],[-70,180],[-90,180],[-90,-180],[-70,-180]], iceColor);
  poly([[-65,-180],[-65,180],[-70,180],[-70,-180],[-65,-180]], snowColor, 0.6);
  // ── Arctic ──
  ctx.fillStyle = iceColor;
  ctx.fillRect(0, 0, W, H * 0.042);
  ctx.fillStyle = snowColor;
  ctx.globalAlpha = 0.5;
  ctx.fillRect(0, 0, W, H * 0.065);
  ctx.globalAlpha = 1.0;
  // ── Ocean grid ──
  ctx.strokeStyle = 'rgba(0,120,180,0.12)';
  ctx.lineWidth = 1.5;
  for (let lon = -180; lon <= 180; lon += 30) {{
    const x = (lon + 180) / 360 * W;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }}
  for (let lat = -90; lat <= 90; lat += 30) {{
    const y = (90 - lat) / 180 * H;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }}
  // ── Equator highlight ──
  ctx.strokeStyle = 'rgba(0,180,255,0.2)';
  ctx.lineWidth = 2;
  const eqY = H / 2;
  ctx.beginPath(); ctx.moveTo(0, eqY); ctx.lineTo(W, eqY); ctx.stroke();
}}
drawDetailedEarth(ctx, TW, TH);
const earthTex = new THREE.CanvasTexture(landCanvas);
earthTex.needsUpdate = true;
// ── Specular / bump canvas ────────────────────────────────────────────────────
const specCanvas = document.createElement('canvas');
specCanvas.width = TW;
specCanvas.height = TH;
const sctx = specCanvas.getContext('2d');
// Black = no specular (land), white = specular (ocean)
const imgData = ctx.getImageData(0, 0, TW, TH);
sctx.fillStyle = 'white';
sctx.fillRect(0, 0, TW, TH);
// Darken over land areas — rough approximation using green channel
const sd = sctx.getImageData(0, 0, TW, TH);
for (let i = 0; i < imgData.data.length; i += 4) {{
  const r = imgData.data[i], g = imgData.data[i+1], b = imgData.data[i+2];
  // Land is greenish/brownish; ocean is dark blue
  if (g > 40 && g > b * 0.8) {{ // land-like pixel
    sd.data[i] = sd.data[i+1] = sd.data[i+2] = 20;
  }} else {{
    sd.data[i] = sd.data[i+1] = sd.data[i+2] = 200;
  }}
  sd.data[i+3] = 255;
}}
sctx.putImageData(sd, 0, 0);
const specTex = new THREE.CanvasTexture(specCanvas);
// ── Earth mesh ────────────────────────────────────────────────────────────────
const R = 6371;
const earthGeo = new THREE.SphereGeometry(R, 128, 128);
const earthMat = new THREE.MeshPhongMaterial({{
  map: earthTex,
  specularMap: specTex,
  specular: new THREE.Color(0x2255aa),
  shininess: 25,
}});
const earthMesh = new THREE.Mesh(earthGeo, earthMat);
scene.add(earthMesh);
// ── Atmosphere layers ─────────────────────────────────────────────────────────
// Inner glow
const atm1 = new THREE.Mesh(
  new THREE.SphereGeometry(R * 1.012, 64, 64),
  new THREE.MeshPhongMaterial({{
    color: new THREE.Color(0x1166cc),
    transparent: true, opacity: 0.06,
    side: THREE.FrontSide, depthWrite: false,
  }})
);
scene.add(atm1);
// Outer halo (visible from distance)
const atm2 = new THREE.Mesh(
  new THREE.SphereGeometry(R * 1.04, 64, 64),
  new THREE.MeshPhongMaterial({{
    color: new THREE.Color(0x0044aa),
    transparent: true, opacity: 0.04,
    side: THREE.BackSide, depthWrite: false,
  }})
);
scene.add(atm2);
// ── Orbit lines + satellite dots ──────────────────────────────────────────────
document.getElementById('sat-count').textContent = ORBIT_DATA.length;
const legItems = document.getElementById('leg-items');
// ECI → Three.js coords (ECI X=right, Y=up, Z=toward viewer)
function eci2xyz(rx, ry, rz) {{
  return new THREE.Vector3(rx, rz, -ry);
}}
const satObjects = [];
ORBIT_DATA.forEach(sat => {{
  const hex = sat.color_hex;
  const color = new THREE.Color(hex);
  const d = document.createElement('div');
  d.className = 'leg-item';
  d.innerHTML = `<div class="leg-dot" style="background:${{hex}}"></div>
                 <div class="leg-name">${{sat.name.substring(0,20)}}</div>`;
  legItems.appendChild(d);
  // Use ECI positions — proper inertial ellipse
  const pts = sat.eci_positions.map(p => eci2xyz(p[0], p[1], p[2]));
  // Orbit line
  const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
  scene.add(new THREE.Line(lineGeo,
    new THREE.LineBasicMaterial({{ color, transparent: true, opacity: 0.5 }})));
 // Satellite icon — cross body + solar panel wings
  const satGroup = new THREE.Group();

  // Main body (small box)
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(120, 80, 80),
    new THREE.MeshBasicMaterial({{ color }})
  );
  satGroup.add(body);

  // Left solar panel
  const panelL = new THREE.Mesh(
    new THREE.BoxGeometry(200, 60, 8),
    new THREE.MeshBasicMaterial({{ color: 0x4488ff, transparent: true, opacity: 0.85 }})
  );
  panelL.position.set(-200, 0, 0);
  satGroup.add(panelL);

  // Right solar panel
  const panelR = new THREE.Mesh(
    new THREE.BoxGeometry(200, 60, 8),
    new THREE.MeshBasicMaterial({{ color: 0x4488ff, transparent: true, opacity: 0.85 }})
  );
  panelR.position.set(200, 0, 0);
  satGroup.add(panelR);

  // Antenna dish (small cone pointing outward)
  const antenna = new THREE.Mesh(
    new THREE.ConeGeometry(30, 80, 8),
    new THREE.MeshBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.7 }})
  );
  antenna.position.set(0, 100, 0);
  satGroup.add(antenna);

  // Glow halo
  const glow = new THREE.Mesh(
    new THREE.SphereGeometry(180, 12, 12),
    new THREE.MeshBasicMaterial({{
      color, transparent: true, opacity: 0.15
    }})
  );
  satGroup.add(glow);

  satGroup.position.copy(pts[0]);
  scene.add(satGroup);
  satObjects.push({{ mesh: satGroup, glow, positions: pts, idx: 0 }});
}});
// ── Mouse interaction ─────────────────────────────────────────────────────────
let isDragging = false, prevX = 0, prevY = 0;
let rotX = 0.3, rotY = 0.5;
let earthRotY = 0;
renderer.domElement.addEventListener('mousedown', e => {{
  isDragging = true; prevX = e.clientX; prevY = e.clientY;
}});
renderer.domElement.addEventListener('mousemove', e => {{
  if (!isDragging) return;
  rotY += (e.clientX - prevX) * 0.004;
  rotX += (e.clientY - prevY) * 0.004;
  rotX = Math.max(-1.4, Math.min(1.4, rotX));
  prevX = e.clientX; prevY = e.clientY;
}});
window.addEventListener('mouseup', () => isDragging = false);
window.addEventListener('mouseleave', () => isDragging = false);
renderer.domElement.addEventListener('wheel', e => {{
  const dist = camera.position.length();
  const newDist = Math.max(7500, Math.min(55000, dist + e.deltaY * 4));
  camera.position.normalize().multiplyScalar(newDist);
  e.preventDefault();
}}, {{ passive: false }});
// ── Controls ──────────────────────────────────────────────────────────────────
let animating = true;
let autoRotate = true; // Earth auto-rotates by default
let focusIdx = 0;
function resetCamera() {{
  rotX = 0.3; rotY = 0.5;
  camera.position.set(0, 4000, 22000);
}}
function toggleRotate() {{
  autoRotate = !autoRotate;
  document.getElementById('rotBtn').textContent =
    autoRotate ? '⟳ EARTH ROT ON' : '⟳ EARTH ROT OFF';
}}
function toggleAnim() {{
  animating = !animating;
  document.getElementById('animBtn').textContent =
    animating ? '⏸ PAUSE' : '▶ PLAY';
}}
function focusNext() {{
  if (!satObjects.length) return;
  const s = satObjects[focusIdx % satObjects.length];
  const pos = s.positions[s.idx];
  const len = pos.length();
  camera.position.copy(pos.clone().normalize().multiplyScalar(len * 2.2));
  focusIdx++;
}}
// ── Animation ─────────────────────────────────────────────────────────────────
let lastAnim = 0;
const ANIM_MS = 700; // 2× slower satellite movement
function animate(ts) {{
  requestAnimationFrame(animate);
  // Earth auto-rotation
  if (autoRotate) earthRotY += 0.0008;
  earthMesh.rotation.y = earthRotY;
  atm1.rotation.y = earthRotY;
  // Satellite animation
  if (animating && ts - lastAnim > ANIM_MS) {{
    lastAnim = ts;
    satObjects.forEach(s => {{
      s.idx = (s.idx + 1) % s.positions.length;
      const pos = s.positions[s.idx];
      s.mesh.position.copy(pos);
      // Orient satellite so panels face perpendicular to velocity
      if (s.idx + 1 < s.positions.length) {{
        const next = s.positions[(s.idx + 1) % s.positions.length];
        s.mesh.lookAt(next);
      }}
    }});
  }}
  // Camera orbit
  const dist = camera.position.length();
  camera.position.set(
    dist * Math.sin(rotY) * Math.cos(rotX),
    dist * Math.sin(rotX),
    dist * Math.cos(rotY) * Math.cos(rotX)
  );
  camera.lookAt(0, 0, 0);
  // Time display
  document.getElementById('time-display').textContent =
    new Date().toUTCString().slice(17, 25) + ' UTC';
  renderer.render(scene, camera);
}}
animate(0);
window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>
"""
    # ── RENDER ────────────────────────────────────────────────────────────────
    col_globe, col_tel = st.columns([4, 1])
    with col_globe:
        components.html(globe_html, height=640, scrolling=False)
    with col_tel:
        st.markdown("""
        <div style="height:12px"></div>
        <div style="font-family:Share Tech Mono,monospace;color:#7a9bb5;
                    font-size:0.6rem;letter-spacing:0.15em;
                    text-transform:uppercase;margin-bottom:12px;">
            TELEMETRY
        </div>
        """, unsafe_allow_html=True)
        for sat in orbit_data[:5]:
            color = sat['color_hex']
            tle_obj = next((t for t in selected_tles
                            if t.object_name == sat['name']), None)
            if not tle_obj:
                continue
            try:
                from Analytics import OrbitalAnalytics
                import math as _math
                oe = tle_obj.to_orbital_element()
                apogee, perigee = OrbitalAnalytics.compute_apogee_perigee(oe)
                period = OrbitalAnalytics.orbital_period_hours(oe)
                inc = _math.degrees(oe.i)
                alt = (apogee + perigee) / 2
                st.markdown(f"""
                <div style="background:#0d1525;border:1px solid #1e3a5f;
                            border-top:2px solid {color};border-radius:3px;
                            padding:8px 10px;margin-bottom:8px;">
                    <div style="font-family:Share Tech Mono,monospace;color:{color};
                                font-size:0.58rem;letter-spacing:0.1em;margin-bottom:6px;
                                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                        {sat['name'][:16]}
                    </div>
                    <div style="font-family:Share Tech Mono,monospace;
                                font-size:0.55rem;line-height:2.0;">
                        <span style="color:#3a5a7a;">ALT </span>
                        <span style="color:#e8f4f8;">{alt:.0f} km</span><br>
                        <span style="color:#3a5a7a;">INC </span>
                        <span style="color:#e8f4f8;">{inc:.2f}°</span><br>
                        <span style="color:#3a5a7a;">PER </span>
                        <span style="color:#e8f4f8;">{period:.3f} hr</span><br>
                        <span style="color:#3a5a7a;">APO </span>
                        <span style="color:#e8f4f8;">{apogee:.0f} km</span><br>
                        <span style="color:#3a5a7a;">PEG </span>
                        <span style="color:#e8f4f8;">{perigee:.0f} km</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                continue
elif page == "Conjunction Analysis":
    st.markdown("""
    <style>
    .conj-card {
        background: #0d1525; border: 1px solid #1e3a5f;
        border-radius: 6px; padding: 16px; margin-bottom: 10px;
    }
    .risk-CRITICAL { border-left: 4px solid #ff3355; }
    .risk-HIGH { border-left: 4px solid #ff6b35; }
    .risk-MEDIUM { border-left: 4px solid #ffd700; }
    .risk-LOW { border-left: 4px solid #00ff88; }
    .risk-NEGLIGIBLE { border-left: 4px solid #7a9bb5; }
    .kpi-box {
        background: #111c2e; border: 1px solid #1e3a5f;
        border-top: 3px solid; border-radius: 4px;
        padding: 14px; text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0d1525;border:1px solid #1e3a5f;border-radius:3px;
                padding:8px 16px;margin-bottom:16px;">
        <span style="font-family:Share Tech Mono,monospace;color:#00d4ff;
                     font-size:0.85rem;letter-spacing:0.15em;">
            ⚠️ CONJUNCTION & COLLISION RISK ANALYSIS
        </span>
    </div>
    """, unsafe_allow_html=True)
    modules = load_modules()
    if not modules:
        st.stop()
    db = get_db(modules)
    if db.count_tles() < 2:
        st.warning("Need at least 2 TLEs.")
        st.stop()
    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        time_horizon = st.slider("TIME HORIZON (days)", 1, 14, 3)
    with col2:
        search_step = st.selectbox("SEARCH STEP (hrs)", [1.0, 2.0, 4.0], index=1)
    # Inside "Conjunction Analysis" page, after loading modules and db
    total_objects = db.count_tles()
    with col3:
        cat_limit = st.slider(
            "CATALOG SIZE",
            min_value=2,
            max_value=max(80, total_objects), # at least 80, but up to whatever is in DB
            value=min(30, total_objects),
            step=5,
            help=f"Max available in database: {total_objects}"
        )
    with col4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run_search = st.button("⚡ RUN ANALYSIS", type="primary",
                               use_container_width=True)
    st.markdown("<hr style='border-color:#1e3a5f;'>", unsafe_allow_html=True)
    # ── Fast cached search ────────────────────────────────────────────────────
    @st.cache_data(ttl=120, show_spinner=False)
    def fast_conjunction_search(db_path, time_horizon, search_step, limit):
        from parser_pipeline import TLEDatabase
        from conjunction_detection import ConjunctionDetector, ConjunctionSearch
        from orbit_mech_engine import SGP4Propagator
        db = TLEDatabase(db_path=db_path)
        tles = db.get_all_tles(limit=limit)
        oes = {}
        for t in tles:
            try:
                oes[t.norad_cat_id] = (t.to_orbital_element(), t.object_name)
            except Exception:
                continue
        detector = ConjunctionDetector(
            time_horizon_days=time_horizon,
            search_step_hours=search_step
        )
        return ConjunctionSearch(detector).search_catalog(oes)
    if run_search:
        with st.spinner("Computing... (cached after first run)"):
            conjunctions = fast_conjunction_search(
                DB_PATH, time_horizon, search_step, cat_limit)
        st.session_state['conjunctions'] = conjunctions
        st.session_state['conj_params'] = (time_horizon, search_step, cat_limit)
    conjunctions = st.session_state.get('conjunctions', [])
    if conjunctions:
        critical = sum(1 for c in conjunctions if c.risk_level == "CRITICAL")
        high = sum(1 for c in conjunctions if c.risk_level == "HIGH")
        medium = sum(1 for c in conjunctions if c.risk_level == "MEDIUM")
        low = sum(1 for c in conjunctions if c.risk_level in ("LOW","NEGLIGIBLE"))
        # ── KPI Row ───────────────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        kpis = [
            (k1, "TOTAL EVENTS", len(conjunctions), "#00d4ff"),
            (k2, "🔴 CRITICAL", critical, "#ff3355"),
            (k3, "🟠 HIGH", high, "#ff6b35"),
            (k4, "🟡 MEDIUM", medium, "#ffd700"),
            (k5, "🟢 LOW / NEG", low, "#00ff88"),
        ]
        for col, label, val, color in kpis:
            with col:
                st.markdown(f"""
                <div class="kpi-box" style="border-top-color:{color};">
                    <div style="font-family:Share Tech Mono,monospace;
                                color:#7a9bb5;font-size:0.6rem;
                                letter-spacing:0.15em;">{label}</div>
                    <div style="font-family:Share Tech Mono,monospace;
                                color:{color};font-size:1.8rem;
                                margin-top:4px;">{val}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        # ── Charts row ────────────────────────────────────────────────────────
        import plotly.graph_objects as go
        cc1, cc2 = st.columns(2)
        with cc1:
            # Risk breakdown donut
            risk_counts = {'CRITICAL': critical, 'HIGH': high,
                           'MEDIUM': medium, 'LOW/NEG': low}
            colors_pie = ['#ff3355','#ff6b35','#ffd700','#00ff88']
            fig_pie = go.Figure(go.Pie(
                labels=list(risk_counts.keys()),
                values=list(risk_counts.values()),
                hole=0.6,
                marker_colors=colors_pie,
                textfont=dict(family='Share Tech Mono', color='#e8f4f8', size=11),
            ))
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Share Tech Mono', color='#7a9bb5'),
                showlegend=True,
                legend=dict(bgcolor='rgba(13,21,37,0.8)',
                            bordercolor='#1e3a5f', borderwidth=1,
                            font=dict(color='#e8f4f8')),
                margin=dict(l=10,r=10,t=30,b=10),
                height=260,
                title=dict(text='RISK BREAKDOWN',
                           font=dict(color='#e8f4f8', size=12,
                                     family='Share Tech Mono')),
                annotations=[dict(text='RISK', x=0.5, y=0.5,
                                  font=dict(color='#00d4ff', size=13,
                                            family='Share Tech Mono'),
                                  showarrow=False)]
            )
            st.plotly_chart(fig_pie, use_container_width=True,
                            config={'displayModeBar': False})
        with cc2:
            # Pc scatter
            fig_sc = go.Figure()
            risk_groups = {}
            for c in conjunctions:
                risk_groups.setdefault(c.risk_level, []).append(c)
            rc_map = {'CRITICAL':'#ff3355','HIGH':'#ff6b35',
                      'MEDIUM':'#ffd700','LOW':'#00ff88','NEGLIGIBLE':'#7a9bb5'}
            for risk, evts in risk_groups.items():
                fig_sc.add_trace(go.Scatter(
                    x=[e.min_distance for e in evts],
                    y=[e.probability_of_collision for e in evts],
                    mode='markers', name=risk,
                    marker=dict(color=rc_map.get(risk,'#7a9bb5'),
                                size=8, line=dict(color='#070b14', width=0.5)),
                    text=[f"{e.sat1_name[:10]} ↔ {e.sat2_name[:10]}" for e in evts],
                    hovertemplate='<b>%{text}</b><br>Dist:%{x:.1f}km Pc:%{y:.2e}<extra></extra>'
                ))
            fig_sc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,21,37,0.8)',
                font=dict(family='Share Tech Mono', color='#7a9bb5', size=10),
                height=260, margin=dict(l=40,r=10,t=30,b=30),
                xaxis=dict(title='MIN DIST (km)', gridcolor='#1e3a5f',
                           color='#7a9bb5'),
                yaxis=dict(title='Pc', type='log', gridcolor='#1e3a5f',
                           color='#7a9bb5'),
                legend=dict(bgcolor='rgba(13,21,37,0.8)',
                            bordercolor='#1e3a5f', borderwidth=1,
                            font=dict(color='#e8f4f8', size=9)),
                title=dict(text='Pc vs DISTANCE',
                           font=dict(color='#e8f4f8', size=12,
                                     family='Share Tech Mono')),
            )
            st.plotly_chart(fig_sc, use_container_width=True,
                            config={'displayModeBar': False})
        # ── Event table ───────────────────────────────────────────────────────
        st.markdown("""
        <div style="font-family:Share Tech Mono,monospace;color:#00d4ff;
                    font-size:0.75rem;letter-spacing:0.15em;
                    border-left:3px solid #00d4ff;padding-left:10px;
                    margin:8px 0 12px 0;">
            CONJUNCTION EVENT TABLE
        </div>
        """, unsafe_allow_html=True)
        rows = []
        for c in conjunctions[:50]:
            rows.append({
                'SAT 1': c.sat1_name[:20],
                'SAT 2': c.sat2_name[:20],
                'TCA': c.tca.strftime('%m-%d %H:%M'),
                'DIST (km)': f"{c.min_distance:.1f}",
                'Pc': f"{c.probability_of_collision:.2e}",
                'RISK': c.risk_level,
            })
        import pandas as pd
        def style_risk(val):
            return {'CRITICAL':'color:#ff3355;font-weight:bold',
                    'HIGH': 'color:#ff6b35;font-weight:bold',
                    'MEDIUM': 'color:#ffd700',
                    'LOW': 'color:#00ff88',
                    'NEGLIGIBLE':'color:#7a9bb5'}.get(val,'')
        df = pd.DataFrame(rows)
        styled = df.style\
            .map(style_risk, subset=['RISK'])\
            .set_table_styles([
                {'selector':'thead th',
                 'props':[('background','#0d1525'),
                          ('color','#00d4ff'),
                          ('font-family','Share Tech Mono,monospace'),
                          ('font-size','0.65rem'),
                          ('letter-spacing','0.1em'),
                          ('border-bottom','1px solid #1e3a5f')]},
                {'selector':'tbody td',
                 'props':[('background','#0a1220'),
                          ('color','#e8f4f8'),
                          ('font-family','Share Tech Mono,monospace'),
                          ('font-size','0.7rem'),
                          ('border-bottom','1px solid #111c2e')]},
                {'selector':'tbody tr:hover td',
                 'props':[('background','#111c2e')]},
            ])
        st.dataframe(styled, use_container_width=True,
                     height=340, hide_index=True)
        # ── Timeline ──────────────────────────────────────────────────────────
        fig_tl = go.Figure()
        for risk, evts in risk_groups.items():
            fig_tl.add_trace(go.Scatter(
                x=[e.tca for e in evts],
                y=[e.probability_of_collision for e in evts],
                mode='markers', name=risk,
                marker=dict(color=rc_map.get(risk,'#7a9bb5'), size=10,
                            line=dict(color='#070b14', width=1)),
                text=[f"{e.sat1_name[:10]} ↔ {e.sat2_name[:10]}" for e in evts],
                hovertemplate='<b>%{text}</b><br>TCA:%{x}<br>Pc:%{y:.2e}<extra></extra>'
            ))
        fig_tl.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,21,37,0.8)',
            font=dict(family='Share Tech Mono', color='#7a9bb5', size=10),
            height=220, margin=dict(l=40,r=10,t=30,b=30),
            xaxis=dict(title='TIME OF CLOSEST APPROACH',
                       gridcolor='#1e3a5f', color='#7a9bb5'),
            yaxis=dict(title='Pc', type='log',
                       gridcolor='#1e3a5f', color='#7a9bb5'),
            legend=dict(bgcolor='rgba(13,21,37,0.8)',
                        bordercolor='#1e3a5f', font=dict(color='#e8f4f8',size=9)),
            title=dict(text='EVENT TIMELINE',
                       font=dict(color='#e8f4f8',size=12,
                                 family='Share Tech Mono')),
        )
        st.plotly_chart(fig_tl, use_container_width=True,
                        config={'displayModeBar': False})
        # ── Export ────────────────────────────────────────────────────────────
        import json as _json
        ce1, ce2 = st.columns(2)
        with ce1:
            st.download_button("⬇ EXPORT CSV",
                pd.DataFrame(rows).to_csv(index=False),
                "conjunctions.csv", "text/csv",
                use_container_width=True)
        with ce2:
            st.download_button("⬇ EXPORT JSON",
                _json.dumps([c.to_dict() for c in conjunctions],
                            indent=2, default=str),
                "conjunctions.json", "application/json",
                use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px;color:#3a5a7a;
                    font-family:Share Tech Mono,monospace;
                    font-size:0.8rem;letter-spacing:0.15em;">
            NO ANALYSIS RUN — CONFIGURE PARAMETERS AND PRESS ⚡ RUN ANALYSIS
        </div>
        """, unsafe_allow_html=True)
# ============================================================================
# PAGE: ANALYTICS
# ============================================================================
elif page == "Analytics":
    st.markdown("""
    <div style="background:#0d1525;border:1px solid #1e3a5f;border-radius:3px;
                padding:8px 16px;margin-bottom:16px;">
        <span style="font-family:Share Tech Mono,monospace;color:#00d4ff;
                     font-size:0.85rem;letter-spacing:0.15em;">
            📊 CATALOG ANALYTICS — POWER BI STYLE
        </span>
    </div>
    """, unsafe_allow_html=True)
    modules = load_modules()
    if not modules:
        st.stop()
    db = get_db(modules)
    if db.count_tles() == 0:
        st.warning("No data available.")
        st.stop()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import math as _math
    import pandas as pd
    all_tles = load_tles(modules, DB_PATH, limit=315)
    orbital_elements = load_orbital_elements(modules, DB_PATH)
    from Analytics import OrbitalAnalytics
    stats = OrbitalAnalytics.catalog_statistics(orbital_elements)
    apogees = [OrbitalAnalytics.compute_apogee_perigee(oe)[0]
                for oe, _ in orbital_elements.values()]
    perigees = [OrbitalAnalytics.compute_apogee_perigee(oe)[1]
                for oe, _ in orbital_elements.values()]
    incs = [_math.degrees(oe.i) for oe, _ in orbital_elements.values()]
    eccs = [oe.e for oe, _ in orbital_elements.values()]
    periods = [OrbitalAnalytics.orbital_period_hours(oe)
                for oe, _ in orbital_elements.values()]
    names = [name for _, name in orbital_elements.values()]
    alts = [(a+p)/2 for a,p in zip(apogees, perigees)]
    # ── KPI Row ───────────────────────────────────────────────────────────────
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    kpi_data = [
        (k1, "OBJECTS", len(orbital_elements), "#00d4ff"),
        (k2, "AVG ALT (km)", f"{sum(alts)/len(alts):.0f}", "#00ff88"),
        (k3, "AVG INC (°)", f"{sum(incs)/len(incs):.1f}", "#ffd700"),
        (k4, "AVG ECC", f"{sum(eccs)/len(eccs):.4f}", "#ff6b35"),
        (k5, "AVG PERIOD(hr)", f"{sum(periods)/len(periods):.2f}", "#a78bfa"),
        (k6, "MAX ALT (km)", f"{max(apogees):.0f}", "#00d4ff"),
    ]
    for col, label, val, color in kpi_data:
        with col:
            st.markdown(f"""
            <div style="background:#111c2e;border:1px solid #1e3a5f;
                        border-top:3px solid {color};border-radius:4px;
                        padding:12px;text-align:center;margin-bottom:8px;">
                <div style="font-family:Share Tech Mono,monospace;
                            color:#7a9bb5;font-size:0.55rem;
                            letter-spacing:0.15em;">{label}</div>
                <div style="font-family:Share Tech Mono,monospace;
                            color:{color};font-size:1.3rem;
                            margin-top:4px;">{val}</div>
            </div>
            """, unsafe_allow_html=True)
    # ── Row 1: Altitude histogram + Inclination histogram ─────────────────────
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=apogees, name='APOGEE',
            marker_color='#00d4ff', opacity=0.75, nbinsx=25))
        fig.add_trace(go.Histogram(x=perigees, name='PERIGEE',
            marker_color='#00ff88', opacity=0.75, nbinsx=25))
        fig.update_layout(
            barmode='overlay',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,21,37,0.9)',
            font=dict(family='Share Tech Mono', color='#7a9bb5', size=10),
            height=240, margin=dict(l=40,r=10,t=35,b=30),
            title=dict(text='ALTITUDE DISTRIBUTION (km)',
                       font=dict(color='#e8f4f8',size=11,
                                 family='Share Tech Mono')),
            xaxis=dict(gridcolor='#1e3a5f', color='#7a9bb5'),
            yaxis=dict(gridcolor='#1e3a5f', color='#7a9bb5'),
            legend=dict(bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e8f4f8', size=9)),
            bargap=0.02,
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={'displayModeBar': False})
    with r1c2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=incs, nbinsx=25,
            marker_color='#ff6b35', opacity=0.85,
            marker_line_color='#070b14', marker_line_width=0.5))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,21,37,0.9)',
            font=dict(family='Share Tech Mono', color='#7a9bb5', size=10),
            height=240, margin=dict(l=40,r=10,t=35,b=30),
            title=dict(text='INCLINATION DISTRIBUTION (°)',
                       font=dict(color='#e8f4f8',size=11,
                                 family='Share Tech Mono')),
            xaxis=dict(gridcolor='#1e3a5f', color='#7a9bb5'),
            yaxis=dict(gridcolor='#1e3a5f', color='#7a9bb5'),
            bargap=0.02, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={'displayModeBar': False})
    # ── Row 2: Scatter + Eccentricity ─────────────────────────────────────────
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=incs, y=alts, mode='markers',
            marker=dict(
                color=eccs, colorscale='Viridis',
                size=6, opacity=0.7,
                colorbar=dict(
                    title=dict(text='ECC', font=dict(color='#7a9bb5',size=9)),
                    tickfont=dict(color='#7a9bb5', size=8),
                    thickness=10, len=0.8,
                ),
                line=dict(color='rgba(0,0,0,0)', width=0),
            ),
            text=names,
            hovertemplate='<b>%{text}</b><br>Inc:%{x:.1f}° Alt:%{y:.0f}km<extra></extra>'
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,21,37,0.9)',
            font=dict(family='Share Tech Mono', color='#7a9bb5', size=10),
            height=260, margin=dict(l=40,r=60,t=35,b=30),
            title=dict(text='ALTITUDE vs INCLINATION (color=ECC)',
                       font=dict(color='#e8f4f8',size=11,
                                 family='Share Tech Mono')),
            xaxis=dict(title='INC (°)', gridcolor='#1e3a5f', color='#7a9bb5'),
            yaxis=dict(title='AVG ALT (km)', gridcolor='#1e3a5f',
                       color='#7a9bb5'),
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={'displayModeBar': False})
    with r2c2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=eccs, nbinsx=25,
            marker_color='#a78bfa', opacity=0.85,
            marker_line_color='#070b14', marker_line_width=0.5
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,21,37,0.9)',
            font=dict(family='Share Tech Mono', color='#7a9bb5', size=10),
            height=260, margin=dict(l=40,r=10,t=35,b=30),
            title=dict(text='ECCENTRICITY DISTRIBUTION',
                       font=dict(color='#e8f4f8',size=11,
                                 family='Share Tech Mono')),
            xaxis=dict(gridcolor='#1e3a5f', color='#7a9bb5'),
            yaxis=dict(gridcolor='#1e3a5f', color='#7a9bb5'),
            bargap=0.02, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={'displayModeBar': False})
    # ── Row 3: Period distribution + Alt vs Period scatter ────────────────────
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=periods, nbinsx=25,
            marker_color='#00ff88', opacity=0.85,
            marker_line_color='#070b14', marker_line_width=0.5
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,21,37,0.9)',
            font=dict(family='Share Tech Mono', color='#7a9bb5', size=10),
            height=220, margin=dict(l=40,r=10,t=35,b=30),
            title=dict(text='ORBITAL PERIOD DISTRIBUTION (hrs)',
                       font=dict(color='#e8f4f8',size=11,
                                 family='Share Tech Mono')),
            xaxis=dict(gridcolor='#1e3a5f', color='#7a9bb5'),
            yaxis=dict(gridcolor='#1e3a5f', color='#7a9bb5'),
            bargap=0.02, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={'displayModeBar': False})
    with r3c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=periods, y=alts, mode='markers',
            marker=dict(color='#ffd700', size=5, opacity=0.65,
                        line=dict(color='rgba(0,0,0,0)')),
            text=names,
            hovertemplate='<b>%{text}</b><br>Period:%{x:.2f}hr Alt:%{y:.0f}km<extra></extra>'
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,21,37,0.9)',
            font=dict(family='Share Tech Mono', color='#7a9bb5', size=10),
            height=220, margin=dict(l=40,r=10,t=35,b=30),
            title=dict(text='ALTITUDE vs ORBITAL PERIOD',
                       font=dict(color='#e8f4f8',size=11,
                                 family='Share Tech Mono')),
            xaxis=dict(title='PERIOD (hrs)', gridcolor='#1e3a5f',
                       color='#7a9bb5'),
            yaxis=dict(title='AVG ALT (km)', gridcolor='#1e3a5f',
                       color='#7a9bb5'),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={'displayModeBar': False})
    # ── Row 4: Full catalog table ──────────────────────────────────────────────
    st.markdown("""
    <div style="font-family:Share Tech Mono,monospace;color:#00d4ff;
                font-size:0.75rem;letter-spacing:0.15em;
                border-left:3px solid #00d4ff;padding-left:10px;
                margin:8px 0 12px 0;">
        FULL CATALOG TABLE
    </div>
    """, unsafe_allow_html=True)
    df_cat = pd.DataFrame([{
        'OBJECT': name,
        'NORAD ID': obj_id,
        'ALT (km)': f"{(OrbitalAnalytics.compute_apogee_perigee(oe)[0]+OrbitalAnalytics.compute_apogee_perigee(oe)[1])/2:.0f}",
        'INC (°)': f"{_math.degrees(oe.i):.2f}",
        'ECC': f"{oe.e:.6f}",
        'PERIOD (hr)': f"{OrbitalAnalytics.orbital_period_hours(oe):.3f}",
        'APOGEE (km)': f"{OrbitalAnalytics.compute_apogee_perigee(oe)[0]:.0f}",
        'PERIGEE (km)':f"{OrbitalAnalytics.compute_apogee_perigee(oe)[1]:.0f}",
    } for obj_id, (oe, name) in orbital_elements.items()])
    st.dataframe(df_cat, use_container_width=True,
                 height=320, hide_index=True)
    with st.expander("📋 RAW STATISTICS JSON"):
        st.json(stats)
# ============================================================================
# PAGE: DATA MANAGEMENT
# ============================================================================
elif page == "Data Management":
    st.title("📥 Data Management")
   
    modules = load_modules()
    if not modules:
        st.stop()
   
    db = get_db(modules)
   
    st.subheader("Current Database Status")
   
    col1, col2, col3 = st.columns(3)
   
    with col1:
        count = db.count_tles()
        st.metric("Records", count)
   
    with col2:
        st.metric("Database", "SQLite")
   
    with col3:
        st.metric("Status", "🟢 Ready")
   
    st.markdown("---")
   
    st.subheader("Fetch Latest TLEs")
   
    col1, col2 = st.columns([3, 1])
   
    with col1:
        group = st.selectbox(
            "TLE Group",
            ["last-30-days", "active", "starlink", "oneweb", "iridium"],
            index=0
        )
   
    with col2:
        st.write("")
        st.write("")
        fetch_btn = st.button("Fetch", type="primary", use_container_width=True)
   
    if fetch_btn:
        with st.spinner(f"Fetching {group} TLEs from Celestrak..."):
            try:
                tles = modules['CelestrakClient'].fetch_group(group=group)
                db.insert_batch(tles, replace=True)
               
                st.success(f"✅ Loaded {len(tles)} TLE records")
                st.info(f"Database now contains {db.count_tles()} total records")
           
            except Exception as e:
                st.error(f"Error fetching TLEs: {e}")
   
    st.markdown("---")
    st.subheader("Browse Data")
   
    browse_limit = st.slider("Show first N records:", 5, 100, 20)
   
    all_tles = db.get_all_tles(limit=50)
   
    df = pd.DataFrame([
        {
            'Object': t.object_name,
            'NORAD ID': t.norad_cat_id,
            'Epoch': t.epoch.strftime('%Y-%m-%d'),
            'Inclination (°)': f"{t.inclination:.2f}",
            'Eccentricity': f"{t.eccentricity:.6f}",
            'Mean Motion': f"{t.mean_motion:.5f}"
        }
        for t in all_tles
    ])
   
    st.dataframe(df, use_container_width=True)
# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888; font-size: 12px;">
        Space Situational Awareness Dashboard | Powered by SGP4, Plotly, Streamlit
    </div>
    """, unsafe_allow_html=True)