import json
from transformers import pipeline
from pygments import highlight
from pygments.formatters.terminal256 import Terminal256Formatter
from pygments.lexers.web import JsonLexer

def dict_log(dict):
  raw_json = json.dumps(dict, indent=4)
  colorful = highlight(
      raw_json,
      lexer=JsonLexer(),
      formatter=Terminal256Formatter(),
  )
  print(colorful)

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
    pass