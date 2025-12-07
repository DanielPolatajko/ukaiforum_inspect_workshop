@task
def simple_decryption_task():
  return Task(
      dataset=dataset,
      solver=generate(),
      scorer=match(),
  )