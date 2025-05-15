import json
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt

# the fields you may change:
# 1) csv_filename   2) jsonl_files   3) output_file   4) extract_initial_question logic

def extract_initial_question(raw_text):
    """
    Given a raw question string, this function:
      1. Strips off any text after "\n\nPrevious Answer:".
      2. Strips off any text after " Choices: " if present.
      3. Replaces any remaining newlines with spaces.
      4. Returns the cleaned question on one line.
    """
    marker_prev_answer = "\n\nPrevious Answer:"
    if marker_prev_answer in raw_text:
        raw_text = raw_text.split(marker_prev_answer)[0]
    marker_choices = " Choices: "
    if marker_choices in raw_text:
        raw_text = raw_text.split(marker_choices)[0]
    return raw_text.strip().replace("\n", " ")

# CSV cache and JSONL input
csv_filename = "/scratch/dkhasha1/bzhang90/Self-InPerfect/scout_new_feedback_5d.csv"
jsonl_files = ["/scratch/dkhasha1/bzhang90/Self-InPerfect/scout_5d_4.1_feedback.jsonl"]
output_file = "5d_newfeed_scout_all.txt"

# Load or build DataFrame
df = None
if os.path.exists(csv_filename):
    print(f"Reading data from existing CSV: {csv_filename}")
    df = pd.read_csv(csv_filename, engine="python")
else:
    results = {}
    for path in jsonl_files:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    q = data.get("original_question", "")
                    it = int(data.get("iteration", 0))
                    lp = data.get("response_probs")
                    if lp is None:
                        continue
                    prob = float(np.exp(lp))
                    corr = int(data.get("is_correct", 0))
                    results.setdefault(q, {})[it] = (prob, corr)
                except json.JSONDecodeError:
                    continue
    rows = []
    for q, its in results.items():
        for it, (prob, corr) in its.items():
            rows.append({"Question": q, "Iteration": it, "Probability": prob, "Is Correct": corr})
    df = pd.DataFrame(rows)
    df.to_csv(csv_filename, index=False)
    print(f"Saved extracted data to {csv_filename}")

# Analysis and bucketing
with open(output_file, 'w') as f:
    # ensure types
    df['Probability'] = pd.to_numeric(df['Probability'], errors='coerce')
    df['Iteration']   = pd.to_numeric(df['Iteration'], errors='coerce')
    df['Is Correct']  = df['Is Correct'].astype(int)

    # iteration-0 subset
    df0 = df[df['Iteration'] == 0].copy()
    minp, maxp = df0['Probability'].min(), df0['Probability'].max()

    # choose nice tick interval (e.g., 0.01)
    step = 0.01
    nice_min = math.floor(minp/step) * step
    nice_max = math.ceil(maxp/step) * step
    # define all bucket centers at these nice values
    bucket_centers = np.arange(nice_min, nice_max + step, step)

    # assign each prob to the nearest bucket center
    df0['bucket_mid'] = df0['Probability'].apply(
        lambda p: float(bucket_centers[np.abs(bucket_centers - p).argmin()])
    )

    # initial stats per bucket_mid
    bucket_stats = df0.groupby('bucket_mid').agg(
        num_questions=('Question', 'count'),
        accuracy=('Is Correct', 'mean')
    ).reset_index()

    # final-correct: ever-correct over all iterations
    final_corr = (
        df.groupby('Question')['Is Correct']
          .max()
          .reset_index()
          .rename(columns={'Is Correct': 'FinalCorrect'})
    )
    # map back iteration-0 probability
    final_corr['iter0_probability'] = final_corr['Question'].map(
        df0.set_index('Question')['Probability']
    )
    final_corr = final_corr.dropna(subset=['iter0_probability'])
    # assign to nearest bucket center
    final_corr['bucket_mid'] = final_corr['iter0_probability'].apply(
        lambda p: float(bucket_centers[np.abs(bucket_centers - p).argmin()])
    )

    # final stats per bucket_mid
    final_stats = final_corr.groupby('bucket_mid').agg(
        final_accuracy=('FinalCorrect', 'mean')
    ).reset_index()

    # merge initial and final
    bucket_stats = bucket_stats.merge(
        final_stats, on='bucket_mid', how='left'
    ).fillna({'final_accuracy': 0})

    # compute improvement statistics
    bucket_stats['num_incorrect']     = bucket_stats['num_questions'] * (1 - bucket_stats['accuracy'])
    bucket_stats['num_correct']       = bucket_stats['num_questions'] * bucket_stats['accuracy']
    bucket_stats['num_final_correct'] = bucket_stats['num_questions'] * bucket_stats['final_accuracy']
    bucket_stats['mid_acc'] = (
        (bucket_stats['num_final_correct'] - bucket_stats['num_correct'])
        / bucket_stats['num_incorrect'].replace({0: np.nan})
    ).fillna(0)

    # write summary
    f.write("Bucket Analysis Results (Nice-Center Buckets):\n")
    for _, r in bucket_stats.iterrows():
        f.write(
            f"Center {r['bucket_mid']:.2f}: Count={int(r['num_questions'])}, "
            f"InitAcc={r['accuracy']:.3f}, Imp={r['mid_acc']:.3f}, "
            f"FinalAcc={r['final_accuracy']:.3f}\n"
        )
    f.write("\n")

    # compute standard deviation and filter
    bucket_stats['Std'] = np.sqrt(
        (bucket_stats['final_accuracy'] * (1 - bucket_stats['final_accuracy']))
        / bucket_stats['num_questions']
    )
    filtered = bucket_stats[bucket_stats['num_questions'] > 10]

    # plotting
# ===== STYLING UPGRADES =====
    # define colors, markers, and styles
    color_final   = '#1f77b4'
    color_delta   = '#ff7f0e'
    color_initial = '#2ca02c'
    marker_final   = 'o'
    marker_delta   = 'D'
    marker_initial = 's'
    lw = 2.5  # line width
    ms = 7    # marker size
    mew = 0.8 # marker edge width

    plt.figure(figsize=(8, 5))

    # Final Accuracy
    plt.errorbar(
        filtered['bucket_mid'],
        filtered['final_accuracy'],
        yerr=filtered['Std'],
        fmt=marker_final + '-',
        color=color_final,
        linewidth=lw, markersize=ms,
        markeredgecolor='black', markeredgewidth=mew,
        capsize=5,
        label='Final Accuracy'
    )

    # Delta Accuracy
    plt.errorbar(
        filtered['bucket_mid'],
        filtered['final_accuracy'] - filtered['accuracy'],
        yerr=filtered['Std'],
        fmt=marker_delta + '--',
        color=color_delta,
        linewidth=lw, markersize=ms,
        markeredgecolor='black', markeredgewidth=mew,
        capsize=5,
        label='$\Delta$ Accuracy'
    )

    # Initial Accuracy
    plt.plot(
        filtered['bucket_mid'],
        filtered['accuracy'],
        marker_initial + '-.',
        color=color_initial,
        linewidth=lw, markersize=ms,
        markeredgecolor='black', markeredgewidth=mew,
        label='Initial Accuracy'
    )

    # Labels and grid
    plt.xlabel('Initial Confidence', fontsize=13, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')

    # X-ticks at exact bucket centers
    ticks = filtered['bucket_mid'].values
    plt.yticks(fontsize=14)
    plt.xticks(ticks, [f"{t:.2f}" for t in ticks], fontsize=14)
    plt.xlim(filtered['bucket_mid'].min(), filtered['bucket_mid'].max())

    # Legend above plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fontsize=11, frameon=False)

    plt.tight_layout()
    plt.savefig('5d_newfeed_scout_all.pdf')
    plt.close()

print(f"Results saved to {output_file}")