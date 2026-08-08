import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2

from datetime import datetime

def get_label(path, all_paths=None):
    basename = os.path.basename(path).replace('.npy', '')
    if '_to_' in basename:
        if all_paths is not None:
            parts = basename.split('_to_')
            if len(parts) == 2:
                base0 = parts[0]
                base1_and_interp = parts[1]
                if '_interp' in base1_and_interp:
                    base1, interp_part = base1_and_interp.split('_interp')
                    try:
                        i = int(interp_part)
                        
                        total_interps = 0
                        for p in all_paths:
                            p_base = os.path.basename(p).replace('.npy', '')
                            if p_base.startswith(f"{base0}_to_{base1}_interp"):
                                total_interps += 1
                                
                        if total_interps == 0:
                            total_interps = i + 1
                            
                        # Extract the timestamp part
                        t0_str = base0.split('_')[-1]
                        t1_str = base1.split('_')[-1]
                        
                        # Try to parse timestamps
                        # We expect format like YYYY-MM-DDTHH-MM or YYYY-MM-DDTHH-MM-SS
                        if len(t0_str.split('-')) == 4:
                            fmt = "%Y-%m-%dT%H-%M"
                        else:
                            fmt = "%Y-%m-%dT%H-%M-%S"
                            
                        t0 = datetime.strptime(t0_str, fmt)
                        t1 = datetime.strptime(t1_str, fmt)
                        
                        dt = (t1 - t0) / (total_interps + 1)
                        t_interp = t0 + dt * (i + 1)
                        
                        # Return formatted string with (AI)
                        if fmt == "%Y-%m-%dT%H-%M":
                            return f"{t_interp.strftime('%Y-%m-%d %H:%M')} (AI)"
                        else:
                            return f"{t_interp.strftime('%Y-%m-%d %H:%M:%S')} (AI)"
                    except (ValueError, IndexError):
                        pass
        return "Interpolated (AI)"
    else:
        parts = basename.split('T')
        if len(parts) == 2:
            time_part = parts[1].replace('-', ':')
            return f"{parts[0]} {time_part}"
        return basename

def load_frame(path):
    arr = np.load(path)
    return arr.squeeze()

def main(args):
    frames = []
    for f in args.frames:
        if os.path.exists(f):
            frames.append(load_frame(f))
        else:
            print(f"Warning: File not found {f}")
    
    if len(frames) == 0:
        print("No valid frames provided.")
        return

    # Find the smallest dimensions to center-crop all frames to match
    min_h = min(f.shape[0] for f in frames)
    min_w = min(f.shape[1] for f in frames)
    
    cropped_frames = []
    for f in frames:
        h, w = f.shape
        cy, cx = h // 2, w // 2
        dy, dx = min_h // 2, min_w // 2
        cropped_frames.append(f[cy-dy:cy+dy, cx-dx:cx+dx])
    
    frames = cropped_frames

    # Normalize all frames globally so the contrast is consistent across the sequence
    global_min = min(np.min(f) for f in frames)
    global_max = max(np.max(f) for f in frames)

    norm_frames = []
    for i, f in enumerate(frames):
        if global_max > global_min:
            n = (f - global_min) / (global_max - global_min)
        else:
            n = f - global_min
        
        # Apply colormap
        colored = plt.get_cmap(args.cmap)(n)
        # Convert to uint8 RGB for saving as GIF
        colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
        
        # Add label
        label = get_label(args.frames[i], args.frames)
        cv2.putText(colored_uint8, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        norm_frames.append(colored_uint8)

    # 1. Create GIF (Animated Visualization)
    if args.gif:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
        gif_path = f"{args.out_prefix}.gif"
        # We can add a ping-pong loop for better visualization in hackathon demos
        if args.ping_pong:
            anim_frames = norm_frames + norm_frames[-2:0:-1]
        else:
            anim_frames = norm_frames
            
        imageio.mimsave(gif_path, anim_frames, fps=args.fps, loop=0)
        print(f"Saved GIF animation to {gif_path}")

    # 2. Create Grid (Static comparison)
    if args.grid:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
        n_frames = len(frames)
        fig, axes = plt.subplots(1, n_frames, figsize=(4 * n_frames, 4))
        if n_frames == 1:
            axes = [axes]
        
        for i, (ax, frame) in enumerate(zip(axes, frames)):
            # We use the raw frame and apply cmap with vmin/vmax so matplotlib handles normalization
            ax.imshow(frame, cmap=args.cmap, vmin=global_min, vmax=global_max)
            ax.axis('off')
            label = get_label(args.frames[i], args.frames)
            if i == 0 or i == n_frames - 1:
                ax.set_title(label, fontsize=14, weight='bold')
            else:
                ax.set_title(label, fontsize=14, color='blue')
        
        plt.tight_layout()
        grid_path = f"{args.out_prefix}_grid.png"
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        print(f"Saved grid comparison to {grid_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize satellite frames (static grid & GIF)")
    parser.add_argument("--frames", nargs='+', required=True, help="List of .npy files in chronological order (e.g. frame0 interp1 interp2 frame1)")
    parser.add_argument("--out-prefix", default="results/visualization", help="Prefix for output files")
    parser.add_argument("--cmap", default="gray", help="Matplotlib colormap (e.g., viridis, gray, plasma, magma)")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second for the GIF")
    parser.add_argument("--ping-pong", action="store_true", help="Make the GIF loop back and forth (ping-pong)")
    parser.add_argument("--no-gif", dest="gif", action="store_false", help="Disable GIF generation")
    parser.add_argument("--no-grid", dest="grid", action="store_false", help="Disable Grid generation")
    args = parser.parse_args()
    main(args)
