from transformers import pipeline, AutoTokenizer
from utils import dict_log

class SyntheticTransformer:
  tokenizer_fn: None
  tokenizer_inputs: None
  model: None

class Metadata:
  tx = SyntheticTransformer()

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

  # - custom tokenizer is built or default used
  # - custom model is built from checkpoint or HF default
  #   used (distilbert-base-uncased-finetuned-sst-2-english)
  # - customizable pytorch-only preprocessing
  def run_transformers(self):
    if self.tokenizer is not None and self.tokenizer is not False:
      self.build_tokenizer()

  # @todo: batch inputs (splat inference.sequence)
  def build_tokenizer(self):
    print("Initializing tokenizer...")
    feature_checkp = self.tokenizer["checkpoint"] if "checkpoint" in self.tokenizer else self.checkpoint
    extra_opts = dict(self.tokenizer)
    if "checkpoint" in extra_opts:
      del extra_opts["checkpoint"]
    if "return_tensors" not in extra_opts:
      extra_opts["return_tensors"] = "pt" # enforce torch
    tokenizer = AutoTokenizer.from_pretrained(feature_checkp)
    inputs = tokenizer(self.sequence, **extra_opts)
    self.tx.tokenizer_fn = tokenizer
    self.tx.tokenizer_inputs = inputs
    print(feature_checkp, extra_opts, self.tx.tokenizer_inputs)
