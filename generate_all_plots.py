import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# the field needed to be change: 
# 1) csv_filename: output a csv 
# 2) jsonl_files: the log file as input 
# 3) output_file: file to store bucket statistics
# 4) possibly need to change the ways of extracting question as key -> extract_initial_question(raw_text)

def extract_initial_question(raw_text):
    """
    Given a raw question string, this function:
    1. Strips off any text after "\n\nPrevious Answer:".
    2. Strips off any text after " Choices: " if present.
    3. Replaces any remaining newlines with spaces.
    4. Returns the cleaned question on one line.
    """
    # truncate at "\n\nPrevious Answer:" if that marker exists.
    marker_prev_answer = "\n\nPrevious Answer:"
    if marker_prev_answer in raw_text:
        question_part = raw_text.split(marker_prev_answer)[0]
    else:
        question_part = raw_text

    # truncate at " Choices: " if that marker exists.
    marker_choices = " Choices: "
    if marker_choices in question_part:
        question_part = question_part.split(marker_choices)[0]

    # strip leading/trailing whitespace and replace internal newlines
    question_part = question_part.strip().replace("\n", " ")

    return question_part

# create a csv file for checking the process
csv_filename = "/scratch/dkhasha1/bzhang90/Self-InPerfect/extracted_log_probs_5d_manual_newfeedback.csv"

if os.path.exists(csv_filename):
    print(f"Reading data from existing CSV: {csv_filename}")
    df = pd.read_csv(
        csv_filename,
        engine="python", 
        sep=",",
        quotechar='"'
    )
else:
    # enter the log file that needed to be extracted
    jsonl_files = [
        "/scratch/dkhasha1/bzhang90/Self-InPerfect/arith_logs/new_5_digits_new_mult_feedback_answer_manual_newpart1.jsonl",
        "/scratch/dkhasha1/bzhang90/Self-InPerfect/new_arith_logs/new_5_digits_new_mult_feedback_answer_manual_newpart2.jsonl",
        "/scratch/dkhasha1/bzhang90/Self-InPerfect/new_arith_logs/new_5_digits_new_mult_feedback_answer_manual_oldpart1.jsonl",
        "/scratch/dkhasha1/bzhang90/Self-InPerfect/new_arith_logs/new_5_digits_new_mult_feedback_answer_manual_oldpart2.jsonl"
    ]
 
    # use the cleaned (initial) question text as the key.
    results = {}  # {question: {iteration: (probability, is_correct)}}
    
    for jsonl_file in jsonl_files:
        if not os.path.exists(jsonl_file):
            print(f"File not found: {jsonl_file}")
            continue
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    raw_question = data.get("question", "Unknown Question")
                    question = extract_initial_question(raw_question)
                    # ensure format is correct
                    marker_c = " Choices: "
                    if marker_c in question:
                        question = question.split(marker_c)[0]
                    question = question.strip()
                    iteration = data.get("iteration", 0)
                    log_prob = data.get("response_probs", None) 
                    
                    probability = np.exp(log_prob).item()

                    is_correct = data.get("is_correct", None)
                    
                    if question not in results:
                        results[question] = {}
                    results[question][iteration] = (probability, is_correct)
                except json.JSONDecodeError:
                    print(f"skipping invalid JSON in {jsonl_file}: {line}")
    
    rows = []
    for question, iterations in results.items():
        for iteration, (probability, is_correct) in iterations.items():
            rows.append({
                "Question": question,
                "Iteration": iteration,
                "Probability": probability,
                "Is Correct": is_correct
            })
    
    df = pd.DataFrame(rows)
    # save the extracted data for future runs.
    df.to_csv(csv_filename, index=False)
    print(f"saved to {csv_filename}")

output_file = "paper_submit.txt" # the file for check detail information of each bucket

with open(output_file, "w") as f:

    # ensure proper data types.
    df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")
    df["Iteration"] = pd.to_numeric(df["Iteration"], errors="coerce")
    df["Is Correct"] = df["Is Correct"].astype(int)


    # define fixed-range buckets
    df_iter0 = df[df["Iteration"] == 0].copy()

    min_prob = df_iter0["Probability"].min()
    max_prob = df_iter0["Probability"].max()
    num_bins = 15
    bins = np.linspace(min_prob, max_prob, num_bins + 1) # create an ordered list

    # Assign each iteration 0 record to a fixed-range bucket
    # this will assign to each bin listed above
    df_iter0["bucket"] = pd.cut(df_iter0["Probability"], bins=bins, include_lowest=True)

    # calculate stats
    # find the final iteration record for each question method checked
    final_indices = df.groupby("Question")["Iteration"].idxmax()
    final_df = df.loc[final_indices].copy()

    # map each question to its iteration 0 probability.
    question_to_prob = df_iter0.set_index("Question")["Probability"].to_dict()
    final_df["iter0_probability"] = final_df["Question"].map(question_to_prob)

    # drop any questions that do not have iteration 0 data.
    final_df = final_df.dropna(subset=["iter0_probability"])

    # assign final iteration records to the same fixed-range bucket.
    final_df["bucket"] = pd.cut(final_df["iter0_probability"], bins=bins, include_lowest=True)


    # merge iteration 0 and final iteration statistics
    bucket_stats = df_iter0.groupby("bucket").agg(
        num_questions=("Question", "count"),
        accuracy=("Is Correct", "mean")
    ).reset_index()

    final_stats = final_df.groupby("bucket").agg(
        final_accuracy=("Is Correct", "mean")
    ).reset_index()

    # merge the data
    bucket_stats = bucket_stats.merge(final_stats, on="bucket")

    # compute midpoints for plot
    bucket_stats["midpoint"] = bucket_stats["bucket"].apply(lambda interval: (interval.left + interval.right) / 2)

    # compute middle accuracy (Mid Acc)
    bucket_stats["num_incorrect"] = bucket_stats["num_questions"] * (1 - bucket_stats["accuracy"])
    bucket_stats["num_correct"] = bucket_stats["num_questions"] * bucket_stats["accuracy"]
    bucket_stats["num_final_correct"] = bucket_stats["num_questions"] * bucket_stats["final_accuracy"]

    # avoid division by zero
    bucket_stats["mid_acc"] = (bucket_stats["num_final_correct"] - bucket_stats["num_correct"]) / bucket_stats["num_incorrect"]
    bucket_stats["mid_acc"] = bucket_stats["mid_acc"].replace([np.inf, -np.inf], np.nan).fillna(0)  # Handle div-by-zero cases


    # print Results
    f.write("Bucket Analysis Results (Fixed-Range Buckets with Middle Accuracy Calculation):\n")
    for _, row in bucket_stats.iterrows():
        f.write(f"Bucket {row['bucket']}:\n")
        f.write(f"  - Midpoint: {row['midpoint']:.3f}\n")
        f.write(f"  - Number of Questions: {row['num_questions']}\n")
        f.write(f"  - Initial Accuracy: {row['accuracy']:.3f}\n")
        f.write(f"  - Final Accuracy (Mid Acc): {row['mid_acc']:.3f}\n")
        f.write(f"  - Total Final Accuracy: {row['final_accuracy']:.3f}\n")

    f.write("\n")
    # calculate standard deviation for each bucket
    bucket_stats["Std"] = np.sqrt((bucket_stats["final_accuracy"] * (1 - bucket_stats["final_accuracy"])) / bucket_stats["num_questions"])

    # find buckets with more than 50 questions
    filtered_bucket_stats = bucket_stats[bucket_stats["num_questions"] > 50]

    # plot midpoint vs. total final accuracy with error bars
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        filtered_bucket_stats["midpoint"], filtered_bucket_stats["final_accuracy"], yerr=filtered_bucket_stats["Std"],
        fmt='o-', capsize=5, label="Final Accuracy with Std Dev"
    )
    
    plt.plot(
    filtered_bucket_stats["midpoint"], filtered_bucket_stats["accuracy"],
    's-', label="Initial Accuracy"
    )


    plt.xlabel("Initial Confidence")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig("paper_submit.png")
    

print(f"results saved to {output_file}")
