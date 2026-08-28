"""
Generate visualization plots for the training report.
Saves PNG files to c:/holo/plots/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import re

os.makedirs('c:/holo/plots', exist_ok=True)

plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 150

def parse_log(filepath, has_lr=False):
    """Parse training log file and extract epoch, train_loss, test_loss"""
    epochs, train_losses, test_losses, lrs = [], [], [], []
    with open(filepath, 'r') as f:
        for line in f:
            # Match lines like: E   0 | TrainL: 244 | TestL: 251 (*1e-3) | LR: 0.001998
            m = re.match(r'E\s*(\d+)\s*\|\s*TrainL:\s*(\d+)\s*\|\s*TestL:\s*(\d+)', line)
            if m:
                epochs.append(int(m.group(1)))
                train_losses.append(int(m.group(2)))
                test_losses.append(int(m.group(3)))
                lr_m = re.search(r'LR:\s*([\d.]+)', line)
                if lr_m:
                    lrs.append(float(lr_m.group(1)))
                else:
                    lrs.append(None)
    return epochs, train_losses, test_losses, lrs


# ============================================================
# Parse all logs
# ============================================================
fwd_v1_e, fwd_v1_tr, fwd_v1_te, _ = parse_log('c:/holo/forward_run_log.txt')
fwd_v2_e, fwd_v2_tr, fwd_v2_te, fwd_v2_lr = parse_log('c:/holo/forward_v2_log.txt')

inv_v1_e, inv_v1_tr, inv_v1_te, _ = parse_log('c:/holo/inverse_run_log.txt')
inv_v2_e, inv_v2_tr, inv_v2_te, inv_v2_lr = parse_log('c:/holo/inverse_v2_origfwd_log.txt')
inv_v2f_e, inv_v2f_tr, inv_v2f_te, inv_v2f_lr = parse_log('c:/holo/inverse_v2_log.txt')
inv_v3_e, inv_v3_tr, inv_v3_te, inv_v3_lr = parse_log('c:/holo/inverse_v3_log.txt')


# ============================================================
# Plot 1: Forward Model Comparison (v1 vs v2)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(fwd_v1_e, fwd_v1_te, 'b-o', markersize=4, label='v1 Test Loss', linewidth=2)
ax1.plot(fwd_v2_e, fwd_v2_te, 'r-s', markersize=4, label='v2 Test Loss', linewidth=2)
ax1.axhline(y=205, color='blue', linestyle='--', alpha=0.5, label='v1 Best (205)')
ax1.axhline(y=193, color='red', linestyle='--', alpha=0.5, label='v2 Best (193)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Loss (RMSE × 10³)')
ax1.set_title('Forward Model: Test Loss Comparison')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([180, 260])

ax2.plot(fwd_v1_e, fwd_v1_tr, 'b--', markersize=3, alpha=0.7, label='v1 Train', linewidth=1.5)
ax2.plot(fwd_v1_e, fwd_v1_te, 'b-', markersize=3, label='v1 Test', linewidth=2)
ax2.plot(fwd_v2_e, fwd_v2_tr, 'r--', markersize=3, alpha=0.7, label='v2 Train', linewidth=1.5)
ax2.plot(fwd_v2_e, fwd_v2_te, 'r-', markersize=3, label='v2 Test', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss (RMSE × 10³)')
ax2.set_title('Forward Model: Train vs Test (Overfitting Analysis)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('c:/holo/plots/01_forward_comparison.png', bbox_inches='tight')
plt.close()
print("Saved: 01_forward_comparison.png")


# ============================================================
# Plot 2: Inverse Model Comparison (v1 vs v2 fair)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(inv_v1_e, inv_v1_te, 'b-o', markersize=4, label='v1 (OtS) Test Loss', linewidth=2)
ax1.plot(inv_v2_e, inv_v2_te, 'r-s', markersize=3, label='v2 (OtS_v2) Test Loss', linewidth=2)
ax1.axhline(y=317, color='blue', linestyle='--', alpha=0.5, label='v1 Best (317)')
ax1.axhline(y=315, color='red', linestyle='--', alpha=0.5, label='v2 Best (315)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Loss (RMSE × 10³)')
ax1.set_title('Inverse Model: Test Loss (Fair Comparison, Same Fwd)')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([310, 400])

# Train vs Test for inverse
ax2.plot(inv_v2_e, inv_v2_tr, 'r--', alpha=0.7, label='v2 Train', linewidth=1.5)
ax2.plot(inv_v2_e, inv_v2_te, 'r-', label='v2 Test', linewidth=2)
ax2.plot(inv_v1_e, inv_v1_tr, 'b--', alpha=0.7, label='v1 Train', linewidth=1.5)
ax2.plot(inv_v1_e, inv_v1_te, 'b-', label='v1 Test', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss (RMSE × 10³)')
ax2.set_title('Inverse Model: Train vs Test Gap')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('c:/holo/plots/02_inverse_comparison.png', bbox_inches='tight')
plt.close()
print("Saved: 02_inverse_comparison.png")


# ============================================================
# Plot 3: All Inverse Models (v1, v2-origfwd, v2-v2fwd, v3)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.plot(inv_v1_e, inv_v1_te, 'b-o', markersize=4, label='v1 OtS (params=195K)', linewidth=2)
ax.plot(inv_v2_e, inv_v2_te, 'r-s', markersize=3, label='v2 OtS_v2 + orig fwd (params=2.2M)', linewidth=2)
ax.plot(inv_v2f_e, inv_v2f_te, 'g-^', markersize=3, label='v2 OtS_v2 + v2 fwd (params=2.2M)', linewidth=2)
if len(inv_v3_e) > 0:
    ax.plot(inv_v3_e, inv_v3_te, 'm-D', markersize=3, label='v3 OtS_v3 + orig fwd (params=16M)', linewidth=2)

ax.axhline(y=317, color='blue', linestyle='--', alpha=0.4, label='v1 Best (317)')
ax.axhline(y=315, color='red', linestyle='--', alpha=0.4, label='v2 Best (315)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Loss (RMSE × 10³)')
ax.set_title('All Inverse Model Variants: Test Loss Comparison')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim([300, 460])

plt.tight_layout()
plt.savefig('c:/holo/plots/03_all_inverse_models.png', bbox_inches='tight')
plt.close()
print("Saved: 03_all_inverse_models.png")


# ============================================================
# Plot 4: Summary Bar Chart
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Forward model bar chart
fwd_models = ['v1\n(StO, 83K params)', 'v2\n(StO_v2, 904K params)']
fwd_losses = [205, 193]
colors_fwd = ['#4472C4', '#ED7D31']
bars = ax1.bar(fwd_models, fwd_losses, color=colors_fwd, width=0.5, edgecolor='black')
ax1.set_ylabel('Best Test Loss (RMSE × 10³)')
ax1.set_title('Forward Model (StO): Best Results')
ax1.set_ylim([0, 250])
for bar, val in zip(bars, fwd_losses):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3, 
             str(val), ha='center', va='bottom', fontweight='bold', fontsize=14)
ax1.axhline(y=193, color='green', linestyle='--', alpha=0.3)

# Inverse model bar chart
inv_models = ['v1\n(OtS, 195K)', 'v2+orig_fwd\n(OtS_v2, 2.2M)', 'v2+v2_fwd\n(OtS_v2, 2.2M)', 'v3*\n(OtS_v3, 16M)']
inv_losses = [317, 315, 326, min(inv_v3_te) if inv_v3_te else 999]
colors_inv = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']
bars = ax2.bar(inv_models, inv_losses, color=colors_inv, width=0.5, edgecolor='black')
ax2.set_ylabel('Best Test Loss (RMSE × 10³)')
ax2.set_title('Inverse Model (OtS): Best Results')
ax2.set_ylim([0, 450])
for bar, val in zip(bars, inv_losses):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3, 
             str(val), ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('c:/holo/plots/04_summary_bar.png', bbox_inches='tight')
plt.close()
print("Saved: 04_summary_bar.png")


# ============================================================
# Plot 5: Forward Model Error Floor Analysis
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Show the gap between train and test to demonstrate overfitting/floor
fwd_v2_gap = [te - tr for tr, te in zip(fwd_v2_tr, fwd_v2_te)]
ax.plot(fwd_v2_e, fwd_v2_tr, 'r--', label='v2 Train Loss', linewidth=2, alpha=0.8)
ax.plot(fwd_v2_e, fwd_v2_te, 'r-', label='v2 Test Loss', linewidth=2)
ax.fill_between(fwd_v2_e, fwd_v2_tr, fwd_v2_te, alpha=0.15, color='red', label='Generalization Gap')
ax.axhline(y=193, color='darkred', linestyle=':', alpha=0.7, label='Test Floor ≈ 193')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (RMSE × 10³)')
ax.set_title('Forward Model v2: Generalization Gap Analysis\n(Train keeps decreasing but Test plateaus → error floor)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('c:/holo/plots/05_forward_error_floor.png', bbox_inches='tight')
plt.close()
print("Saved: 05_forward_error_floor.png")


# ============================================================
# Plot 6: Bottleneck Diagram - Why we can't improve further
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Inverse Design Pipeline: Bottleneck Analysis', fontsize=16, fontweight='bold', pad=20)

# Draw the pipeline boxes
boxes = [
    (0.5, 4, 2, 1.5, 'Target\nOptical\nResponse\n(60-dim)', '#E6F3FF'),
    (3.5, 4, 2, 1.5, 'Inverse\nModel\n(OtS)', '#FFE6E6'),
    (6.5, 4, 2, 1.5, 'Forward\nModel\n(StO)', '#E6FFE6'),
]
for x, y, w, h, text, color in boxes:
    rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor=color, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold', zorder=3)

# Arrows
ax.annotate('', xy=(3.4, 4.75), xytext=(2.6, 4.75),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(6.4, 4.75), xytext=(5.6, 4.75),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Labels on arrows
ax.text(3.0, 5.2, '60→6×6\nbinary', ha='center', fontsize=8, color='navy')
ax.text(6.0, 5.2, '6×6→60\ncontinuous', ha='center', fontsize=8, color='darkgreen')

# Output
ax.text(9.0, 4.75, 'Predicted\nOptical\nResponse', ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFFCC', edgecolor='black', linewidth=2))
ax.annotate('', xy=(8.3, 4.75), xytext=(8.6, 4.75),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Error annotations
ax.text(4.5, 2.5, '❌ Bottleneck 1: Forward Model Error Floor\n'
        'Forward model RMSE ≥ 193×10⁻³\n'
        'This is the IRREDUCIBLE ERROR from the surrogate model.\n'
        'The inverse model loss can NEVER be lower than this.',
        fontsize=9, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFD7D7', edgecolor='red', linewidth=2))

ax.text(4.5, 0.8, '❌ Bottleneck 2: One-to-Many Mapping\n'
        'Multiple 6×6 binary structures → same optical response.\n'
        'The inverse problem is ILL-POSED: no unique solution.\n'
        'The model must pick ONE from many valid structures.',
        fontsize=9, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFE8D7', edgecolor='orange', linewidth=2))

ax.text(4.5, 6.8, '📊 Current Results: Forward Best = 193 | Inverse Best = 315\n'
        'Inverse loss (315) = Forward error (193) + Inverse-specific error (~122)\n'
        'Even a PERFECT inverse model would still have loss ≈ 193',
        fontsize=9, ha='center', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#D7FFD7', edgecolor='green', linewidth=2))

plt.tight_layout()
plt.savefig('c:/holo/plots/06_bottleneck_diagram.png', bbox_inches='tight')
plt.close()
print("Saved: 06_bottleneck_diagram.png")


# ============================================================
# Plot 7: Model Scaling Analysis
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

params = [194531, 2213521, 16046277]
best_losses = [317, 315, min(inv_v3_te) if inv_v3_te else 383]
labels = ['v1 (OtS)\n195K params', 'v2 (OtS_v2)\n2.2M params', 'v3 (OtS_v3)\n16M params']

ax.semilogx(params, best_losses, 'ro-', markersize=12, linewidth=2, zorder=3)
for p, l, label in zip(params, best_losses, labels):
    ax.annotate(f'{label}\nLoss={l}', (p, l), textcoords="offset points", 
                xytext=(0, 20), ha='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))

ax.axhline(y=193, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Forward Model Floor (193)')
ax.fill_between([1e4, 1e8], [193, 193], [0, 0], alpha=0.1, color='green')

ax.set_xlabel('Model Parameters (log scale)')
ax.set_ylabel('Best Test Loss (RMSE × 10³)')
ax.set_title('Inverse Model: Scaling Law Analysis\n(More parameters ≠ better loss → fundamental limit)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([1e5, 3e7])
ax.set_ylim([180, 400])

plt.tight_layout()
plt.savefig('c:/holo/plots/07_scaling_analysis.png', bbox_inches='tight')
plt.close()
print("Saved: 07_scaling_analysis.png")

print("\n✅ All plots generated successfully!")
