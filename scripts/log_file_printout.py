from inspect_ai.log import list_eval_logs, read_eval_log

log_path = list_eval_logs()[0]
log = read_eval_log(log_path)

print(f"📊 Results Summary:")
print(f"Samples: {log.eval.task_args.get('limit', 'all')}")
print(f"Accuracy: {log.results.scores[0].metrics["accuracy"].value}")
print(f"Model: {log.eval.model}")

print("\n📝 Example transcript:")
print(f"  Input: {log.samples[0].input}...")
print(f"  Transcript: {log.samples[0].scores["match"].answer}")