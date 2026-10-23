from pathlib import Path
REPO_ROOT = Path(r'c:\Users\lpascu\resp_analysis')
windows = ['win1_baseline', 'win2_anox1', 'win3_anox2', 'win4_gasp_early', 'win5_gasp_late', 'win6_flow_shift', 'win7_recovery']
for w in windows:
    for s in ['kilosort4', 'lupin']:
        p = REPO_ROOT / 'outputs' / 'e097' / 'subwindows_sorting' / w / s / 'alf' / 'spike_times.npy'
        if not p.exists():
            print(f'MISSING: {w}/{s}')
        else:
            print(f'OK: {w}/{s}')
print('All checked!')
