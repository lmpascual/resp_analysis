import matplotlib.pyplot as plt
from PIL import Image
import os

os.makedirs('presentation_assets', exist_ok=True)

# Unit 004 onset & offset
p_u4_on = r'c:\Users\lpascu\analysis_src\output_individual_neuron_rasters_onset_sorted_insp_duration_LP21_60s\unit_004_depth_1155um_raster_peth.png'
p_u4_off = r'c:\Users\lpascu\analysis_src\output_individual_neuron_rasters_offset_sorted_insp_duration_LP21_60s\unit_004_depth_1155um_aligned_offset_raster_peth.png'

# Unit 016 onset & offset
p_u16_on = r'c:\Users\lpascu\analysis_src\output_individual_neuron_rasters_onset_sorted_insp_duration_LP21_60s\unit_016_depth_1005um_raster_peth.png'
p_u16_off = r'c:\Users\lpascu\analysis_src\output_individual_neuron_rasters_offset_sorted_insp_duration_LP21_60s\unit_016_depth_1005um_aligned_offset_raster_peth.png'

fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=200)
fig.patch.set_facecolor('#ffffff')

# U4 onset
im1 = Image.open(p_u4_on)
axes[0, 0].imshow(im1)
axes[0, 0].axis('off')
axes[0, 0].set_title("Unit 004 (Depth 1,155 µm) - Aligned to Inspiration Onset (I_on)\nSharp Onset, Dispersed Offset across T_I", 
                     fontsize=12, fontweight='bold', color='#0f172a', pad=10)

# U4 offset
im2 = Image.open(p_u4_off)
axes[0, 1].imshow(im2)
axes[0, 1].axis('off')
axes[0, 1].set_title("Unit 004 (Depth 1,155 µm) - Aligned to Inspiration Offset (I_off)\nDispersed Onset, Fixed Offset Alignment", 
                     fontsize=12, fontweight='bold', color='#0f172a', pad=10)

# U16 onset
im3 = Image.open(p_u16_on)
axes[1, 0].imshow(im3)
axes[1, 0].axis('off')
axes[1, 0].set_title("Unit 016 (Depth 1,005 µm) - Aligned to Inspiration Onset (I_on)\nRamping Pre-inspiratory Onset, Sharp Offset", 
                     fontsize=12, fontweight='bold', color='#0f172a', pad=10)

# U16 offset
im4 = Image.open(p_u16_off)
axes[1, 1].imshow(im4)
axes[1, 1].axis('off')
axes[1, 1].set_title("Unit 016 (Depth 1,005 µm) - Aligned to Inspiration Offset (I_off)\nAbrupt Burst Quenching at Inspiration-Off", 
                     fontsize=12, fontweight='bold', color='#0f172a', pad=10)

plt.tight_layout()
out_path = 'presentation_assets/comparison_onset_vs_offset_units.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.savefig('presentation_assets/comparison_onset_vs_offset_units.svg', format='svg', bbox_inches='tight')
plt.close()
print(f"Saved {out_path} and .svg")
