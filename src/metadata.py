from transformers import pipeline, AutoTokenizer
from utils import dict_log

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
    dict_log(raw_out) # ensure --log
    return raw_out

  # in this check, a separate tokenizer/model/head
  # is specified. the definition is as follows:
  # - custom tokenizer is built or default used
  # - custom model is built from checkpoint or HF default
  #   used (distilbert-base-uncased-finetuned-sst-2-english)
  # - customizable pytorch-only preprocessing
  def run_transformers(self):
    if self.tokenizer is not None and self.tokenizer is not False:
      self.build_tokenizer()

  def build_tokenizer(self):
    print("Initializing tokenizer...")
    feature_checkp = self.tokenizer["checkpoint"] if "checkpoint" in self.tokenizer else self.checkpoint
    extra_opts = dict(self.tokenizer)
    if "checkpoint" in extra_opts:
      del extra_opts["checkpoint"]
    print(feature_checkp, extra_opts)
    pass
