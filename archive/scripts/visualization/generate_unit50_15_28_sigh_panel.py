import matplotlib.pyplot as plt
from PIL import Image
import os

os.makedirs('presentation_assets', exist_ok=True)

# 1. Composite Panel: Unit 50 & Unit 15 Fixed-Latency Disinhibition & Pioneer Spikes
p_u50 = r'c:\Users\lpascu\analysis_src\output_individual_neuron_rasters_onset_sorted_insp_duration_LP21_60s\unit_050_depth_465um_raster_peth.png'
p_u15 = r'c:\Users\lpascu\analysis_src\output_individual_neuron_rasters_onset_sorted_insp_duration_LP21_60s\unit_015_depth_480um_raster_peth.png'

fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=200)
fig.patch.set_facecolor('#ffffff')

if os.path.exists(p_u50):
    im50 = Image.open(p_u50)
    axes[0].imshow(im50)
    axes[0].axis('off')
    axes[0].set_title("Unit #50 (Depth: 465 µm | BötC)\nSharp Early-I Quenching & Fixed-Latency Turn-On (~110 ms post-I_on)", 
                      fontsize=12, fontweight='bold', color='#0f172a', pad=10)

if os.path.exists(p_u15):
    im15 = Image.open(p_u15)
    axes[1].imshow(im15)
    axes[1].axis('off')
    axes[1].set_title("Unit #15 (Depth: 480 µm | BötC)\nSharp I_on Inhibition & Precisely Timed Pioneer Spike (~115 ms post-I_on)", 
                      fontsize=12, fontweight='bold', color='#0f172a', pad=10)

plt.tight_layout()
out_disinh = 'presentation_assets/unit50_unit15_disinhibition_panel.png'
plt.savefig(out_disinh, dpi=200, bbox_inches='tight')
plt.savefig('presentation_assets/unit50_unit15_disinhibition_panel.svg', format='svg', bbox_inches='tight')
plt.close()
print("Saved", out_disinh, "and .svg")

# 2. Sigh & Unit 28 Trial 120 Panel
p_u28 = r'c:\Users\lpascu\analysis_src\output_individual_neuron_rasters_onset_sorted_insp_duration_LP21_60s\unit_028_depth_900um_raster_peth.png'
fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=200)
fig.patch.set_facecolor('#ffffff')

if os.path.exists(p_u28):
    im28 = Image.open(p_u28)
    axes[0].imshow(im28)
    axes[0].axis('off')
    axes[0].set_title("Unit #28 (Depth: 900 µm | preBötC Inspiratory Unit)\nNote Trial ~120: Extended Burst with Highly Regular ISIs (Putative Sigh)", 
                      fontsize=12, fontweight='bold', color='#0f172a', pad=10)

# Right: Conceptual Diagram of Sigh Trajectory vs Eupneic Trajectory in State Space
ax2 = axes[1]
ax2.set_facecolor('#0f172a')
import numpy as np

# Draw normal limit cycle
t = np.linspace(0, 2*np.pi, 200)
x_eupnea = 1.2 * np.cos(t)
y_eupnea = 0.9 * np.sin(t)
ax2.plot(x_eupnea, y_eupnea, color='#38bdf8', lw=3, label='Normal Eupneic Limit Cycle')

# Draw expanded sigh trajectory
t_sigh = np.linspace(0, 2*np.pi, 300)
x_sigh = 2.4 * np.cos(t_sigh)
y_sigh = 1.9 * np.sin(t_sigh)
ax2.plot(x_sigh, y_sigh, color='#f43f5e', lw=3.5, linestyle='--', label='Sigh (Augmented Breath) Trajectory')

# Fixed point / attractor well
ax2.scatter([1.2], [0.0], color='#fbbf24', s=180, zorder=5, label='Inspiration-Off Attractor')
ax2.annotate('Attractor Basin\n(Convergence point)', xy=(1.2, 0.0), xytext=(0.5, -0.6),
             arrowprops=dict(facecolor='#fbbf24', shrink=0.08, width=1.5, headwidth=6),
             color='#fbbf24', fontweight='bold', fontsize=10)

ax2.set_title("Dynamical Manifold: Eupnea vs. Sigh Excursion (Bush et al., 2024)", 
              fontsize=13, fontweight='bold', color='#f8fafc', pad=12)
ax2.set_xlabel("Latent Mode 1 (PC1)", color='#94a3b8', fontsize=11, fontweight='semibold')
ax2.set_ylabel("Latent Mode 2 (PC2)", color='#94a3b8', fontsize=11, fontweight='semibold')
ax2.tick_params(colors='#94a3b8')
ax2.grid(True, linestyle=':', alpha=0.3, color='#64748b')
ax2.legend(loc='upper left', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=10)

plt.tight_layout()
out_sigh = 'presentation_assets/sigh_trial120_dynamics_panel.png'
plt.savefig(out_sigh, dpi=200, bbox_inches='tight')
plt.savefig('presentation_assets/sigh_trial120_dynamics_panel.svg', format='svg', bbox_inches='tight')
plt.close()
print("Saved", out_sigh, "and .svg")
