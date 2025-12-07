from inspect_ai.analysis import evals_df, samples_df, messages_df

from inspect_ai.log import list_eval_logs

log_path = list_eval_logs()[0]


evals = evals_df(log_path)

print("📊 Results Summary:")
print(f"Model: {evals['model'][0]}")
print(f"Accuracy: {evals['score_match_accuracy'][0]}")
print(f"Samples: {evals['total_samples'][0]}")

samples = samples_df(log_path)

print(f"\nTotal samples: {len(samples)}")
print(f"Correct: {samples['score_match'].apply(lambda x: 0 if x == "I" else 1).sum()}")

messages = messages_df(log_path)

print(f"Sample Transcript:")
print(f"Input: {messages['content'][0]}")
print(f"Output: {messages['content'][1]}")