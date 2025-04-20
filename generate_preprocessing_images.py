import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
print("--- Script Started ---")

# Assuming wrappers.py is in the same directory
try:
    from wrappers import wrap_mario, WarpFrame, FrameStack # Import specific wrappers if needed
    print("--- Imported wrappers successfully ---")
except ImportError as e:
    print(f"!!! ERROR: Failed to import from wrappers.py: {e}")
    print("!!! Make sure wrappers.py is in the same directory.")
    exit()

# --- Configuration ---
OUTPUT_DIR = "preprocessing_visuals"
FRAME_SKIP_VALUE = 4 # Make sure this matches your actual wrappers.py!
# --- End Configuration ---

# Create output directory
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- Ensured output directory exists: {OUTPUT_DIR} ---")
except Exception as e:
    print(f"!!! ERROR: Could not create output directory: {e}")
    exit()

# --- Stage 1: Get Raw Frame ---
try:
    print("Stage 1: Generating raw frame...")
    env_raw = gym_super_mario_bros.make('SuperMarioBros-v0')
    env_raw = JoypadSpace(env_raw, COMPLEX_MOVEMENT)
    print("  Raw environment created.")
    raw_obs = env_raw.reset()
    print("  Raw environment reset.")
    # Take a few steps
    for i in range(50):
        raw_obs, _, done, _ = env_raw.step(env_raw.action_space.sample())
        if done:
            raw_obs = env_raw.reset()
    print("  Took initial steps.")
    env_raw.close()
    print("  Raw environment closed.")

    plt.figure(figsize=(4, 3.75)) # Approx NES aspect ratio
    plt.imshow(raw_obs)
    plt.title("1. Raw Input Frame")
    plt.axis('off')
    save_path = os.path.join(OUTPUT_DIR, "1_raw_frame.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved raw frame to {save_path}")
except Exception as e:
    print(f"!!! ERROR in Stage 1 (Raw Frame): {e}")
    exit()

# --- Stage 2 & 3: Apply WarpFrame (Grayscale + Resize) ---
try:
    print("Stage 2/3: Generating warped frame...")
    gray_frame = cv2.cvtColor(raw_obs, cv2.COLOR_RGB2GRAY)
    print("  Converted to grayscale.")
    resized_frame = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
    print("  Resized frame.")

    plt.figure(figsize=(3, 3))
    plt.imshow(resized_frame, cmap='gray')
    plt.title("2. Grayscale & Resized (84x84)")
    plt.axis('off')
    save_path = os.path.join(OUTPUT_DIR, "2_warped_frame.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved warped frame to {save_path}")
except Exception as e:
    print(f"!!! ERROR in Stage 2/3 (Warp Frame): {e}")
    exit()

# --- Stage 4: Get Stacked Frames ---
try:
    print("Stage 4: Generating stacked frames...")
    env_wrapped = gym_super_mario_bros.make('SuperMarioBros-v0')
    env_wrapped = JoypadSpace(env_wrapped, COMPLEX_MOVEMENT)
    env_wrapped = wrap_mario(env_wrapped) # Apply ALL your wrappers
    print("  Wrapped environment created.")

    state = env_wrapped.reset()
    print("  Wrapped environment reset.")
    # Take a few steps
    for _ in range(10): # Need at least k=4 steps
        action = env_wrapped.action_space.sample()
        state, _, done, _ = env_wrapped.step(action)
        if done:
            state = env_wrapped.reset()
    print("  Took initial steps in wrapped env.")
    env_wrapped.close()
    print("  Wrapped environment closed.")

    stacked_frames_array = np.array(state)
    print(f"  Converted LazyFrames to numpy array, shape: {stacked_frames_array.shape}")

    is_scaled = np.max(stacked_frames_array) <= 1.0
    print(f"  Frame seems scaled (max val <= 1.0): {is_scaled}")

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    fig.suptitle("3. Example Stacked Input Frames (k=4)", fontsize=14)

    num_channels = stacked_frames_array.shape[-1] if len(stacked_frames_array.shape) == 3 else 1

    if len(stacked_frames_array.shape) == 3 and num_channels == 4:
        for i in range(4):
            frame_to_show = stacked_frames_array[:, :, i]
            img = frame_to_show * 255.0 if is_scaled else frame_to_show
            axes[i].imshow(img.astype(np.uint8), cmap='gray', vmin=0, vmax=255)
            axes[i].set_title(f"Frame t-{3-i}")
            axes[i].axis('off')
    elif len(stacked_frames_array.shape) == 2:
         axes[0].imshow(stacked_frames_array, cmap='gray')
         axes[0].set_title("Single Frame")
         axes[0].axis('off')
         print("!!! Warning: State shape suggests FrameStack might not have applied correctly.")
    else:
         print(f"!!! Error: Unexpected state shape {stacked_frames_array.shape}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_path = os.path.join(OUTPUT_DIR, "3_stacked_frames.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved stacked frames plot to {save_path}")
except Exception as e:
    print(f"!!! ERROR in Stage 4 (Stacked Frames): {e}")
    exit()

print(f"\n--- Final State Tensor Shape Confirmed: {stacked_frames_array.shape} ---") # ADD THIS LINE
print(f"\n--- Script Finished --- Images should be in directory: {OUTPUT_DIR}")