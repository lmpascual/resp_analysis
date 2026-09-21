#!/usr/bin/env python3
"""
Upgrade build_modulation_presentation_html.py to support multiple trial-by-trial
transition detection methods:
1. Poisson Change-Point (MLE, PyTau-style)
2. Single-Trial SDF Dynamic Threshold (35%)
3. Significant Pause / Burst Core
4. First Spike / Last Spike (Legacy)
"""

from pathlib import Path

SCRIPT_PATH = Path(r"C:\Users\lpascu\analysis_src\scripts\build_modulation_presentation_html.py")
content = SCRIPT_PATH.read_text(encoding="utf-8")

# 1. Add compute_all_methods_for_unit before main()
methods_func = '''
def compute_all_methods_for_unit(spks, cell_type, t_on_arr, t_flow_arr, t_emg_arr, ti_flow_arr, n_trials):
    methods = {
        'poisson_mle': {
            'name': 'Poisson Change-Point (MLE, PyTau-style)',
            'label1': 'Activation / Onset (t₁)' if cell_type == 'insp' else 'Inactivation / Quench (t₁)',
            'label2': 'Cessation Cliff (t₂)' if cell_type == 'insp' else 'Rebound Disinhibition (t₂)',
            't1_on': [], 't2_on': [], 't1_off': [], 't2_off': [], 'ti': []
        },
        'sdf_thresh': {
            'name': 'Single-Trial SDF Dynamic Threshold (35%)',
            'label1': 'Activation / Onset (t₁)' if cell_type == 'insp' else 'Inactivation / Quench (t₁)',
            'label2': 'Cessation Cliff (t₂)' if cell_type == 'insp' else 'Rebound Disinhibition (t₂)',
            't1_on': [], 't2_on': [], 't1_off': [], 't2_off': [], 'ti': []
        },
        'quench_pause': {
            'name': 'Significant Pause / Core Burst',
            'label1': 'Burst Start (t₁)' if cell_type == 'insp' else 'Inactivation / Quench (t₁)',
            'label2': 'Burst Cliff (t₂)' if cell_type == 'insp' else 'Rebound Disinhibition (t₂)',
            't1_on': [], 't2_on': [], 't1_off': [], 't2_off': [], 'ti': []
        },
        'first_last': {
            'name': 'First Spike / Last Spike (Legacy)',
            'label1': 'First Spike (t_start)',
            'label2': 'Last Spike Cliff (t_cliff)',
            't1_on': [], 't2_on': [], 't1_off': [], 't2_off': [], 'ti': []
        }
    }
    
    dt_grid = 0.002
    kernel = signal.windows.gaussian(15, std=3)
    kernel /= np.sum(kernel)
    
    for tr in range(n_trials):
        t_on = t_on_arr[tr]
        t_flow = t_flow_arr[tr]
        t_emg = t_emg_arr[tr]
        ti = ti_flow_arr[tr]
        
        # 1. Baseline First/Last
        s_b = spks[(spks >= t_on - 0.04) & (spks <= t_emg + 0.06)]
        if len(s_b) >= 2:
            f_on = (s_b[0] - t_on) * 1000.0
            l_on = (s_b[-1] - t_on) * 1000.0
            f_off = (s_b[0] - t_flow) * 1000.0
            l_off = (s_b[-1] - t_flow) * 1000.0
            methods['first_last']['t1_on'].append(f_on)
            methods['first_last']['t2_on'].append(l_on)
            methods['first_last']['t1_off'].append(f_off)
            methods['first_last']['t2_off'].append(l_off)
            methods['first_last']['ti'].append(ti)
            
        # 2. Significant Pause / Burst Core
        s_long = spks[(spks >= t_on - 0.10) & (spks <= t_flow + 0.15)]
        if len(s_long) >= 2:
            isis = np.diff(s_long) * 1000.0
            if cell_type == 'exp':
                max_idx = np.argmax(isis)
                if isis[max_idx] > 35.0:
                    q_on = (s_long[max_idx] - t_on) * 1000.0
                    r_on = (s_long[max_idx + 1] - t_on) * 1000.0
                    q_off = (s_long[max_idx] - t_flow) * 1000.0
                    r_off = (s_long[max_idx + 1] - t_flow) * 1000.0
                    methods['quench_pause']['t1_on'].append(q_on)
                    methods['quench_pause']['t2_on'].append(r_on)
                    methods['quench_pause']['t1_off'].append(q_off)
                    methods['quench_pause']['t2_off'].append(r_off)
                    methods['quench_pause']['ti'].append(ti)
            else: # insp
                burst_spks = s_long[(s_long >= t_on - 0.02) & (s_long <= t_flow + 0.05)]
                if len(burst_spks) >= 2:
                    methods['quench_pause']['t1_on'].append((burst_spks[0] - t_on) * 1000.0)
                    methods['quench_pause']['t2_on'].append((burst_spks[-1] - t_on) * 1000.0)
                    methods['quench_pause']['t1_off'].append((burst_spks[0] - t_flow) * 1000.0)
                    methods['quench_pause']['t2_off'].append((burst_spks[-1] - t_flow) * 1000.0)
                    methods['quench_pause']['ti'].append(ti)
                    
        # 3. Single-trial SDF threshold (35%)
        w_start = t_on - 0.10
        w_end = t_flow + 0.15
        t_grid = np.arange(w_start, w_end, dt_grid)
        s_tr = spks[(spks >= w_start) & (spks <= w_end)]
        counts, _ = np.histogram(s_tr, bins=np.append(t_grid, t_grid[-1] + dt_grid))
        sdf = np.convolve(counts / dt_grid, kernel, mode='same')
        pk = np.max(sdf)
        mn = np.min(sdf)
        if pk - mn > 15.0:
            thresh = mn + 0.35 * (pk - mn)
            above = sdf >= thresh
            diffs = np.diff(above.astype(int))
            up_cross = np.where(diffs == 1)[0]
            down_cross = np.where(diffs == -1)[0]
            if cell_type == 'insp':
                valid_ups = [u for u in up_cross if (t_grid[u] >= t_on - 0.04) and (t_grid[u] <= t_flow)]
                valid_downs = [d for d in down_cross if (t_grid[d] >= t_on) and (t_grid[d] <= t_flow + 0.06)]
                if len(valid_ups) > 0 and len(valid_downs) > 0:
                    t1 = t_grid[valid_ups[0]]
                    t2 = t_grid[valid_downs[-1]]
                    methods['sdf_thresh']['t1_on'].append((t1 - t_on) * 1000.0)
                    methods['sdf_thresh']['t2_on'].append((t2 - t_on) * 1000.0)
                    methods['sdf_thresh']['t1_off'].append((t1 - t_flow) * 1000.0)
                    methods['sdf_thresh']['t2_off'].append((t2 - t_flow) * 1000.0)
                    methods['sdf_thresh']['ti'].append(ti)
            else: # exp
                valid_downs = [d for d in down_cross if (t_grid[d] >= t_on - 0.06) and (t_grid[d] <= t_on + 0.04)]
                valid_ups = [u for u in up_cross if (t_grid[u] >= t_flow - 0.03) and (t_grid[u] <= t_flow + 0.12)]
                if len(valid_downs) > 0 and len(valid_ups) > 0:
                    t1 = t_grid[valid_downs[0]]
                    t2 = t_grid[valid_ups[-1]]
                    methods['sdf_thresh']['t1_on'].append((t1 - t_on) * 1000.0)
                    methods['sdf_thresh']['t2_on'].append((t2 - t_on) * 1000.0)
                    methods['sdf_thresh']['t1_off'].append((t1 - t_flow) * 1000.0)
                    methods['sdf_thresh']['t2_off'].append((t2 - t_flow) * 1000.0)
                    methods['sdf_thresh']['ti'].append(ti)
                    
        # 4. Fast Poisson MLE Changepoint (2-changepoint grid)
        t1_cand = np.arange(t_on - 0.04, t_on + 0.04, 0.003)
        t2_cand = np.arange(t_flow - 0.03, t_flow + 0.06, 0.003)
        best_ll = -np.inf
        best_t1, best_t2 = np.nan, np.nan
        s_c = spks[(spks >= t_on - 0.08) & (spks <= t_flow + 0.10)]
        T_0 = t_on - 0.08
        T_end = t_flow + 0.10
        if len(s_c) >= 2:
            n_tot = len(s_c)
            for c1 in t1_cand:
                n1 = np.sum(s_c < c1)
                d1 = max(c1 - T_0, 1e-4)
                l1 = max(n1 / d1, 1e-4)
                ll1 = n1 * np.log(l1) - n1
                for c2 in t2_cand:
                    if c2 <= c1 + 0.01: continue
                    n2 = np.sum((s_c >= c1) & (s_c < c2))
                    d2 = c2 - c1
                    l2 = max(n2 / d2, 1e-4)
                    ll2 = n2 * np.log(l2) - n2
                    
                    n3 = n_tot - n1 - n2
                    d3 = max(T_end - c2, 1e-4)
                    l3 = max(n3 / d3, 1e-4)
                    ll3 = n3 * np.log(l3) - n3
                    
                    if cell_type == 'insp' and (l2 < l1 or l2 < l3): continue
                    if cell_type == 'exp' and (l2 > l1 or l2 > l3): continue
                    
                    tot = ll1 + ll2 + ll3
                    if tot > best_ll:
                        best_ll = tot
                        best_t1, best_t2 = c1, c2
                        
            if not np.isnan(best_t1):
                methods['poisson_mle']['t1_on'].append((best_t1 - t_on) * 1000.0)
                methods['poisson_mle']['t2_on'].append((best_t2 - t_on) * 1000.0)
                methods['poisson_mle']['t1_off'].append((best_t1 - t_flow) * 1000.0)
                methods['poisson_mle']['t2_off'].append((best_t2 - t_flow) * 1000.0)
                methods['poisson_mle']['ti'].append(ti)
                
    # Calculate linear regression and jitter metrics for each method
    for mname in methods:
        d = methods[mname]
        t1_on = np.array(d['t1_on'])
        t2_on = np.array(d['t2_on'])
        t1_off = np.array(d['t1_off'])
        t2_off = np.array(d['t2_off'])
        ti = np.array(d['ti'])
        
        if len(ti) > 10:
            s1, i1, r1, p1, _ = stats.linregress(ti, t1_on)
            s2, i2, r2, p2, _ = stats.linregress(ti, t2_on)
            s2_off, _, r2_off, _, _ = stats.linregress(ti, t2_off)
            d['metrics'] = {
                'mean_t1_on': round(float(t1_on.mean()), 1),
                'std_t1_on': round(float(t1_on.std()), 2),
                'slope_t1': round(float(s1), 3),
                'r2_t1': round(float(r1 ** 2), 3),
                'mean_t2_on': round(float(t2_on.mean()), 1),
                'std_t2_on': round(float(t2_on.std()), 2),
                'slope_t2': round(float(s2), 3),
                'r2_t2': round(float(r2 ** 2), 3),
                'mean_t2_off': round(float(t2_off.mean()), 1),
                'std_t2_off': round(float(t2_off.std()), 2),
                'slope_t2_off': round(float(s2_off), 3),
                'r2_t2_off': round(float(r2_off ** 2), 3)
            }
        else:
            d['metrics'] = {
                'mean_t1_on': 0.0, 'std_t1_on': 0.0, 'slope_t1': 0.0, 'r2_t1': 0.0,
                'mean_t2_on': 0.0, 'std_t2_on': 0.0, 'slope_t2': 0.0, 'r2_t2': 0.0,
                'mean_t2_off': 0.0, 'std_t2_off': 0.0, 'slope_t2_off': 0.0, 'r2_t2_off': 0.0
            }
        d['t1_on'] = np.round(t1_on, 1).tolist()
        d['t2_on'] = np.round(t2_on, 1).tolist()
        d['t1_off'] = np.round(t1_off, 1).tolist()
        d['t2_off'] = np.round(t2_off, 1).tolist()
        d['ti'] = np.round(ti, 1).tolist()
        
    return methods
'''

content = content.replace("def main():", methods_func + "\ndef main():")

# 2. Update archetypes_info with cell_type
old_archetypes = '''    archetypes_info = {
        255: {
            "name": "Unit 255",
            "archetype": "preBötC Inspiratory Ramping Follower",
            "desc": "Gradual accelerating ramp during inspiration, abruptly cut off at flow expiration onset. Demonstrates the decoupling between slow onset and sharp cliff.",
            "depth_um": 1262.5
        },
        70: {
            "name": "Unit 070",
            "archetype": "preBötC Vanguard Burst Follower",
            "desc": "Explosive, razor-sharp burst onset locked to I_on with sub-7ms jitter. Burst persists throughout inspiration with smeared offset.",
            "depth_um": 1287.5
        },
        94: {
            "name": "Unit 094",
            "archetype": "BötC Expiratory Suppression",
            "desc": "Active during expiration, intensely quenched during inspiration. Disinhibition fires immediately upon flow slope reversal.",
            "depth_um": 187.5
        },
        18: {
            "name": "Unit 018",
            "archetype": "BötC Post-I Rebound Pioneer",
            "desc": "Dense pre-inspiratory firing, quenched at I_on, followed by an immediate sharp rebound pioneer burst locked to the flow trough.",
            "depth_um": 187.5
        },
        163: {
            "name": "Unit 163",
            "archetype": "preBötC Candidate Pacemaker",
            "desc": "Rhythmogenic pioneer unit initiating firing ~25 ms prior to diaphragm EMG onset, driving network escape from the Inspiration-Off attractor.",
            "depth_um": 1537.5
        }
    }'''

new_archetypes = '''    archetypes_info = {
        255: {
            "name": "Unit 255",
            "archetype": "preBötC Inspiratory Ramping Follower",
            "desc": "Gradual accelerating ramp during inspiration, abruptly cut off at flow expiration onset. Demonstrates the decoupling between slow onset and sharp cliff.",
            "depth_um": 1262.5,
            "cell_type": "insp"
        },
        70: {
            "name": "Unit 070",
            "archetype": "preBötC Vanguard Burst Follower",
            "desc": "Explosive, razor-sharp burst onset locked to I_on with sub-7ms jitter. Burst persists throughout inspiration with smeared offset.",
            "depth_um": 1287.5,
            "cell_type": "insp"
        },
        94: {
            "name": "Unit 094",
            "archetype": "BötC Expiratory Suppression",
            "desc": "Active during expiration, intensely quenched during inspiration. Disinhibition fires immediately upon flow slope reversal.",
            "depth_um": 187.5,
            "cell_type": "exp"
        },
        18: {
            "name": "Unit 018",
            "archetype": "BötC Post-I Rebound Pioneer",
            "desc": "Dense pre-inspiratory firing, quenched at I_on, followed by an immediate sharp rebound pioneer burst locked to the flow trough.",
            "depth_um": 187.5,
            "cell_type": "exp"
        },
        163: {
            "name": "Unit 163",
            "archetype": "preBötC Candidate Pacemaker",
            "desc": "Rhythmogenic pioneer unit initiating firing ~25 ms prior to diaphragm EMG onset, driving network escape from the Inspiration-Off attractor.",
            "depth_um": 1537.5,
            "cell_type": "insp"
        }
    }'''

content = content.replace(old_archetypes, new_archetypes)

# 3. Call compute_all_methods_for_unit and add to exemplar_data
old_exemplar_block = '''        exemplar_data.append({
            "uid": uid,
            "name": archetypes_info[uid]["name"],
            "archetype": archetypes_info[uid]["archetype"],
            "desc": archetypes_info[uid]["desc"],
            "depth_um": archetypes_info[uid]["depth_um"],'''

new_exemplar_block = '''        cell_type = archetypes_info[uid]["cell_type"]
        methods_dict = compute_all_methods_for_unit(spks, cell_type, t_on_arr, t_flow_arr, t_off_emg_arr, ti_flow_arr, n_trials)

        exemplar_data.append({
            "uid": uid,
            "cell_type": cell_type,
            "methods": methods_dict,
            "name": archetypes_info[uid]["name"],
            "archetype": archetypes_info[uid]["archetype"],
            "desc": archetypes_info[uid]["desc"],
            "depth_um": archetypes_info[uid]["depth_um"],'''

content = content.replace(old_exemplar_block, new_exemplar_block)

# 4. Add Transition Detector selector and dynamic label spans in HTML controls
old_controls = '''      <div style="display: flex; gap: 12px; align-items: center;">
        <span style="font-size: 12px; font-weight: 700; color: var(--muted); text-transform: uppercase;">Archetype:</span>
        <select id="exSelect">
          <option value="255">Unit #255 — Inspiratory Ramping Follower (1262 µm)</option>
          <option value="70">Unit #070 — Vanguard Burst Follower (1287 µm)</option>
          <option value="94">Unit #094 — BötC Expiratory Quenched (187 µm)</option>
          <option value="18">Unit #018 — BötC Post-I Rebound Pioneer (187 µm)</option>
          <option value="163">Unit #163 — preBötC Early Pacemaker (1537 µm)</option>
        </select>
        <span style="font-size: 12px; color: var(--cyan); font-weight: 600;" id="exArchetypeTag"></span>
      </div>

      <div class="check-group">
        <label><input type="checkbox" id="chkFirstSpk" checked> First Spike ($t_{start}$)</label>
        <label><input type="checkbox" id="chkLastSpk" checked> Last Spike Cliff ($t_{cliff}$)</label>'''

new_controls = '''      <div style="display: flex; gap: 14px; align-items: center; flex-wrap: wrap;">
        <div style="display: flex; gap: 8px; align-items: center;">
          <span style="font-size: 12px; font-weight: 700; color: var(--muted); text-transform: uppercase;">Archetype:</span>
          <select id="exSelect" style="background: #060910; color: #fff; border: 1px solid var(--border); padding: 5px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer;">
            <option value="255">Unit #255 — Inspiratory Ramping Follower (1262 µm)</option>
            <option value="70">Unit #070 — Vanguard Burst Follower (1287 µm)</option>
            <option value="94">Unit #094 — BötC Expiratory Quenched (187 µm)</option>
            <option value="18">Unit #018 — BötC Post-I Rebound Pioneer (187 µm)</option>
            <option value="163">Unit #163 — preBötC Early Pacemaker (1537 µm)</option>
          </select>
          <span style="font-size: 12px; color: var(--cyan); font-weight: 600;" id="exArchetypeTag"></span>
        </div>

        <div style="display: flex; gap: 8px; align-items: center; border-left: 1px solid var(--border); padding-left: 14px;">
          <span style="font-size: 12px; font-weight: 700; color: #a855f7; text-transform: uppercase;">Transition Detector:</span>
          <select id="selMethod" style="background: #060910; color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 5px 10px; border-radius: 6px; font-size: 12px; font-weight: 700; cursor: pointer;">
            <option value="poisson_mle" selected>Poisson Change-Point (MLE, PyTau-style)</option>
            <option value="sdf_thresh">Single-Trial SDF Dynamic Threshold (35%)</option>
            <option value="quench_pause">Significant Pause / Core Burst</option>
            <option value="first_last">First Spike / Last Spike (Legacy)</option>
          </select>
        </div>
      </div>

      <div class="check-group">
        <label><input type="checkbox" id="chkFirstSpk" checked> <span id="lblFirstSpk" style="color: #38bdf8; font-weight: 600;">Transition 1 (t₁)</span></label>
        <label><input type="checkbox" id="chkLastSpk" checked> <span id="lblLastSpk" style="color: #fb923c; font-weight: 600;">Transition 2 (t₂)</span></label>'''

content = content.replace(old_controls, new_controls)

# 5. Update Workbench JS event listeners and functions
old_wb_init = '''    const exSelect = document.getElementById('exSelect');
    exSelect.onchange = (e) => {
      currentExUid = parseInt(e.target.value);
      renderWorkbench();
    };

    ['chkFirstSpk', 'chkLastSpk', 'chkFlowTrough', 'chkEmgOffset', 'chkBraking'].forEach(id => {
      document.getElementById(id).onchange = renderWorkbench;
    });

    function getEx() { return DATA.exemplars.find(x => x.uid === currentExUid); }

    function renderWorkbench() {
      const ex = getEx();
      if (!ex) return;
      document.getElementById('exArchetypeTag').innerText = ex.archetype;

      drawWbRaster('cvWbRasterOn', 'onset', ex);
      drawWbPeth('cvWbPethOn', 'onset', ex);

      drawWbRaster('cvWbRasterOff', 'offset', ex);
      drawWbScatter('cvWbScatter', ex);
    }'''

new_wb_init = '''    const exSelect = document.getElementById('exSelect');
    const selMethod = document.getElementById('selMethod');

    exSelect.onchange = (e) => {
      currentExUid = parseInt(e.target.value);
      updateLabels();
      renderWorkbench();
    };

    selMethod.onchange = () => {
      updateLabels();
      renderWorkbench();
    };

    ['chkFirstSpk', 'chkLastSpk', 'chkFlowTrough', 'chkEmgOffset', 'chkBraking'].forEach(id => {
      document.getElementById(id).onchange = renderWorkbench;
    });

    function getEx() { return DATA.exemplars.find(x => x.uid === currentExUid); }

    function updateLabels() {
      const ex = getEx();
      if (!ex || !ex.methods) return;
      const mKey = selMethod.value;
      const mData = ex.methods[mKey] || ex.methods['first_last'];
      document.getElementById('lblFirstSpk').innerText = mData.label1;
      document.getElementById('lblLastSpk').innerText = mData.label2;
    }

    function renderWorkbench() {
      const ex = getEx();
      if (!ex) return;
      document.getElementById('exArchetypeTag').innerText = ex.archetype;
      updateLabels();

      drawWbRaster('cvWbRasterOn', 'onset', ex);
      drawWbPeth('cvWbPethOn', 'onset', ex);

      drawWbRaster('cvWbRasterOff', 'offset', ex);
      drawWbScatter('cvWbScatter', ex);
    }'''

content = content.replace(old_wb_init, new_wb_init)

# 6. Update drawWbRaster overlay section to pull from selected method
old_raster_overlays = '''      // 4. Overlays for First Spike & Last Spike Cliff
      const firstArr = mode === 'onset' ? ex.scatter_first_on : ex.scatter_first_off;
      const lastArr = mode === 'onset' ? ex.scatter_last_on : ex.scatter_last_off;

      if (document.getElementById('chkFirstSpk').checked && firstArr) {
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        for (let tr = 0; tr < firstArr.length; tr++) {
          const x = toX(firstArr[tr]);
          const y = toY(tr + 0.5);
          if (tr === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      if (document.getElementById('chkLastSpk').checked && lastArr) {
        ctx.strokeStyle = '#fb923c';
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        for (let tr = 0; tr < lastArr.length; tr++) {
          const x = toX(lastArr[tr]);
          const y = toY(tr + 0.5);
          if (tr === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }'''

new_raster_overlays = '''      // 4. Overlays for Transition 1 & Transition 2
      const mKey = selMethod ? selMethod.value : 'poisson_mle';
      const mData = (ex.methods && ex.methods[mKey]) ? ex.methods[mKey] : (ex.methods ? ex.methods['first_last'] : null);
      const firstArr = mData ? (mode === 'onset' ? mData.t1_on : mData.t1_off) : (mode === 'onset' ? ex.scatter_first_on : ex.scatter_first_off);
      const lastArr = mData ? (mode === 'onset' ? mData.t2_on : mData.t2_off) : (mode === 'onset' ? ex.scatter_last_on : ex.scatter_last_off);

      if (document.getElementById('chkFirstSpk').checked && firstArr) {
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        for (let tr = 0; tr < firstArr.length; tr++) {
          const x = toX(firstArr[tr]);
          const y = toY(tr + 0.5);
          if (tr === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      if (document.getElementById('chkLastSpk').checked && lastArr) {
        ctx.strokeStyle = '#fb923c';
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        for (let tr = 0; tr < lastArr.length; tr++) {
          const x = toX(lastArr[tr]);
          const y = toY(tr + 0.5);
          if (tr === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }'''

content = content.replace(old_raster_overlays, new_raster_overlays)

# 7. Update drawWbScatter to use mData scatter and regression metrics
old_scatter_body = '''    function drawWbScatter(cvId, ex) {
      const cv = document.getElementById(cvId);
      const ctx = cv.getContext('2d');
      const w = cv.width, h = cv.height;
      ctx.clearRect(0, 0, w, h);

      const padL = 55, padR = 20, padT = 15, padB = 25;
      const plotW = w - padL - padR;
      const plotH = h - padT - padB;

      // X: T_I duration (40 - 170 ms), Y: Latency from I_on (-50 to +200 ms)
      const xMin = 40, xMax = 170;
      const yMin = -40, yMax = 180;

      function toX(t) { return padL + (t - xMin) / (xMax - xMin) * plotW; }
      function toY(l) { return padT + plotH - (l - yMin) / (yMax - yMin) * plotH; }

      // Identity slope line (y = x)
      ctx.strokeStyle = 'rgba(255,255,255,0.15)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(toX(40), toY(40)); ctx.lineTo(toX(170), toY(170)); ctx.stroke();
      ctx.setLineDash([]);

      // Scatter: First Spikes (Blue)
      ctx.fillStyle = '#38bdf8';
      for (let i = 0; i < ex.scatter_ti.length; i++) {
        ctx.beginPath();
        ctx.arc(toX(ex.scatter_ti[i]), toY(ex.scatter_first_on[i]), 2.2, 0, Math.PI * 2);
        ctx.fill();
      }

      // Scatter: Last Spike Cliff (Orange)
      ctx.fillStyle = '#fb923c';
      for (let i = 0; i < ex.scatter_ti.length; i++) {
        ctx.beginPath();
        ctx.arc(toX(ex.scatter_ti[i]), toY(ex.scatter_last_on[i]), 2.2, 0, Math.PI * 2);
        ctx.fill();
      }

      // Regression lines
      // First spike regression (Blue)
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      const yF1 = ex.mean_first_on + ex.slope_first * (40 - 68);
      const yF2 = ex.mean_first_on + ex.slope_first * (170 - 68);
      ctx.moveTo(toX(40), toY(yF1)); ctx.lineTo(toX(170), toY(yF2)); ctx.stroke();

      // Last spike regression (Orange)
      ctx.strokeStyle = '#fb923c';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      const yL1 = ex.mean_last_on + ex.slope_last * (40 - 68);
      const yL2 = ex.mean_last_on + ex.slope_last * (170 - 68);
      ctx.moveTo(toX(40), toY(yL1)); ctx.lineTo(toX(170), toY(yL2)); ctx.stroke();

      // Labels & HUD
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.strokeRect(padL, padT, plotW, plotH);
      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      for (let ti = 50; ti <= 150; ti += 50) {
        ctx.fillText(`${ti} ms`, toX(ti), padT + plotH + 15);
      }
      ctx.fillText('Inspiratory Duration T_I (ms)', padL + plotW / 2, padT + plotH + 22);

      // Regression readouts
      ctx.textAlign = 'left';
      ctx.fillStyle = '#38bdf8';
      ctx.fillText(`First Spk β = ${ex.slope_first.toFixed(2)} (R²=${ex.r2_first})`, padL + 10, padT + 18);
      ctx.fillStyle = '#fb923c';
      ctx.fillText(`Last Spk β = ${ex.slope_last.toFixed(2)} (R²=${ex.r2_last})`, padL + 10, padT + 32);
    }'''

new_scatter_body = '''    function drawWbScatter(cvId, ex) {
      const cv = document.getElementById(cvId);
      const ctx = cv.getContext('2d');
      const w = cv.width, h = cv.height;
      ctx.clearRect(0, 0, w, h);

      const padL = 55, padR = 20, padT = 15, padB = 25;
      const plotW = w - padL - padR;
      const plotH = h - padT - padB;

      const mKey = selMethod ? selMethod.value : 'poisson_mle';
      const mData = (ex.methods && ex.methods[mKey]) ? ex.methods[mKey] : (ex.methods ? ex.methods['first_last'] : null);
      const tiArr = mData ? mData.ti : ex.scatter_ti;
      const t1Arr = mData ? mData.t1_on : ex.scatter_first_on;
      const t2Arr = mData ? mData.t2_on : ex.scatter_last_on;
      const m = mData ? mData.metrics : {
        slope_t1: ex.slope_first, r2_t1: ex.r2_first, std_t1_on: ex.sigma_first_on, mean_t1_on: ex.mean_first_on,
        slope_t2: ex.slope_last, r2_t2: ex.r2_last, std_t2_off: ex.sigma_last_off, mean_t2_on: ex.mean_last_on
      };

      // X: T_I duration (40 - 170 ms), Y: Latency from I_on (-60 to +200 ms)
      const xMin = 40, xMax = 170;
      const yMin = -60, yMax = 200;

      function toX(t) { return padL + (t - xMin) / (xMax - xMin) * plotW; }
      function toY(l) { return padT + plotH - (l - yMin) / (yMax - yMin) * plotH; }

      // Identity slope line (y = x)
      ctx.strokeStyle = 'rgba(255,255,255,0.15)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(toX(40), toY(40)); ctx.lineTo(toX(170), toY(170)); ctx.stroke();
      ctx.setLineDash([]);

      // Scatter: Transition 1 (Cyan)
      ctx.fillStyle = '#38bdf8';
      for (let i = 0; i < tiArr.length; i++) {
        ctx.beginPath();
        ctx.arc(toX(tiArr[i]), toY(t1Arr[i]), 2.2, 0, Math.PI * 2);
        ctx.fill();
      }

      // Scatter: Transition 2 (Orange)
      ctx.fillStyle = '#fb923c';
      for (let i = 0; i < tiArr.length; i++) {
        ctx.beginPath();
        ctx.arc(toX(tiArr[i]), toY(t2Arr[i]), 2.2, 0, Math.PI * 2);
        ctx.fill();
      }

      // Regression lines
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      const yF1 = m.mean_t1_on + m.slope_t1 * (40 - 68);
      const yF2 = m.mean_t1_on + m.slope_t1 * (170 - 68);
      ctx.moveTo(toX(40), toY(yF1)); ctx.lineTo(toX(170), toY(yF2)); ctx.stroke();

      ctx.strokeStyle = '#fb923c';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      const yL1 = m.mean_t2_on + m.slope_t2 * (40 - 68);
      const yL2 = m.mean_t2_on + m.slope_t2 * (170 - 68);
      ctx.moveTo(toX(40), toY(yL1)); ctx.lineTo(toX(170), toY(yL2)); ctx.stroke();

      // Labels & HUD
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.strokeRect(padL, padT, plotW, plotH);
      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      for (let ti = 50; ti <= 150; ti += 50) {
        ctx.fillText(`${ti} ms`, toX(ti), padT + plotH + 15);
      }
      ctx.fillText('Inspiratory Duration T_I (ms)', padL + plotW / 2, padT + plotH + 22);

      // Regression readouts
      const lbl1 = mData ? mData.label1 : 'First Spk';
      const lbl2 = mData ? mData.label2 : 'Last Spk';
      ctx.textAlign = 'left';
      ctx.fillStyle = '#38bdf8';
      ctx.fillText(`${lbl1} β = ${m.slope_t1.toFixed(2)} (R²=${m.r2_t1.toFixed(2)}, σ=${m.std_t1_on.toFixed(1)}ms)`, padL + 10, padT + 18);
      ctx.fillStyle = '#fb923c';
      ctx.fillText(`${lbl2} β = ${m.slope_t2.toFixed(2)} (R²=${m.r2_t2.toFixed(2)}, σ=${m.std_t2_off.toFixed(1)}ms)`, padL + 10, padT + 32);
    }'''

content = content.replace(old_scatter_body, new_scatter_body)

SCRIPT_PATH.write_text(content, encoding="utf-8")
print("[SUCCESS] Successfully upgraded build_modulation_presentation_html.py!")
