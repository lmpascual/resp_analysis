import base64
from PIL import Image
from pathlib import Path

def export_subsets():
    base_dir = Path(r"c:\Users\lpascu\analysis_src")
    img_path = base_dir / "output_lebedev_rigor_checks_60s" / "lebedev_2019_rotation_dynamics_rigor_checks.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Missing {img_path}")

    img = Image.open(img_path)
    out_dir = base_dir / "presentation_assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Panels A, B, C, D (Sequential Wave Cascade)
    crop_abcd = img.crop((350, 220, 3500, 925))
    p1_png = out_dir / "lebedev_panel_a_b_c_d_sequential_wave.png"
    p1_svg = out_dir / "lebedev_panel_a_b_c_d_sequential_wave.svg"
    crop_abcd.save(p1_png, dpi=(300, 300))

    # 2. Panels E, F, I1, I2 (Direct 3D Trajectory & PCA Invariance)
    crop_efi = img.crop((350, 940, 1920, 2400))
    p2_png = out_dir / "lebedev_panel_e_f_i_subpop_trajectories.png"
    p2_svg = out_dir / "lebedev_panel_e_f_i_subpop_trajectories.svg"
    crop_efi.save(p2_png, dpi=(300, 300))

    # 3. Panels G, H & Summary (Geometric Fit & Rigor Summary)
    crop_gh = img.crop((1960, 940, 3500, 2400))
    p3_png = out_dir / "lebedev_panel_g_h_summary.png"
    p3_svg = out_dir / "lebedev_panel_g_h_summary.svg"
    crop_gh.save(p3_png, dpi=(300, 300))

    # Vector SVGs
    for png, svg in [(p1_png, p1_svg), (p2_png, p2_svg), (p3_png, p3_svg)]:
        with open(png, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        im = Image.open(png)
        w, h = im.size
        svg_txt = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n'
            f'  <image width="{w}" height="{h}" href="data:image/png;base64,{b64}" />\n'
            f'</svg>'
        )
        with open(svg, "w") as f:
            f.write(svg_txt)

    print(f"Successfully generated Lebedev subset panels:")
    print(f"  - {p1_png.name} ({crop_abcd.size[0]}x{crop_abcd.size[1]})")
    print(f"  - {p2_png.name} ({crop_efi.size[0]}x{crop_efi.size[1]})")
    print(f"  - {p3_png.name} ({crop_gh.size[0]}x{crop_gh.size[1]})")

if __name__ == "__main__":
    export_subsets()
