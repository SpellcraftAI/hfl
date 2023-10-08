from transformers import pipeline

class Metadata:
  def __init__(self) -> None:
    pass

  def run(self):
    if self.pipeline is False:
      self.run_transformers()
    else:
      self.run_pipeline()

  # basic pipeline routine
  def run_pipeline(self):
    generator = pipeline(self.task, self.checkpoint)
    raw_out = generator(self.sequence)
    print(raw_out) # ensure --log
    return raw_out

  # in this check, a separate tokenizer/model/head
  # is specified. the definition is as follows:
  # - custom tokenizer is built or default used
  # - custom model is built from checkpoint or HF default
  #   used (distilbert-base-uncased-finetuned-sst-2-english)
  # - customizable pytorch-only preprocessing
  def run_transformers(self):
    pass