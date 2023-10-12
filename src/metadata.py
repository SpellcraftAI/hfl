import torch
from enum import Enum
from datasets import Dataset, load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForCausalLM, AutoModel, DataCollatorWithPadding, TrainingArguments, Trainer
from utils import dict_dump, trace_dump, rule_dump, join_truthy_keys

SEQ_CLX_BINARY_LABEL_MAPPING = {0: "negative", 1: "positive"}

class HFLMode(Enum):
  TRAIN = "train"
  EVAL  = "eval"
  INF   = "inference"

# The head is initialized to the task for relevance
# and enabling transfer learning, where necessary.
class Task(Enum):
    SEQUENCE_CLASSIFICATION = "sequence-classification"
    MASKED_LANGUAGE_MODEL = "masked-language-model"
    TEXT_GENERATION = "text-generation"

def serialize_batch_encoding(batch_enc):
  return {
    'input_ids': batch_enc.input_ids.tolist(),
    'attention_mask': batch_enc.attention_mask.tolist(),
  }

class SyntheticTransformer:
  tokenizer_fn: None
  tokenizer_inputs: None
  model: None
  outputs: None
  dataset: None

  def pipe_tokens_to_model(self):
    self.outputs = self.model(**self.tokenizer_inputs)

  def decode_causal_model_output(self):
    pass

  def softmax_seqclx_model_output(self):
    pass

  def argmax_seqclx_model_output(self):
    with torch.no_grad():
      outputs = self.model(self.tokenizer_inputs["input_ids"])

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_label = SEQ_CLX_BINARY_LABEL_MAPPING[predicted_class]
    dict_dump({"predicted_class":predicted_class, "predicted_label":predicted_label})

class Metadata:
  tx = SyntheticTransformer()

  def __init__(self) -> None:
    pass

  def run(self):
    if self.pipeline is False:
      rule_dump("[bold blue]hfl synthetic")
      self.run_synthetic()
    else:
      rule_dump("[bold blue]hfl pipeline")
      self.run_pipeline()

  # basic pipeline routine
  def run_pipeline(self):
    generator = pipeline(self.task, self.checkpoint)
    raw_out = generator(self.sequence)
    dict_dump(raw_out) # ensure --log
    return raw_out

  # - custom tokenizer is built or default used
  # - custom model is built from checkpoint or HF default
  #   used (distilbert-base-uncased-finetuned-sst-2-english)
  # - customizable pytorch-only preprocessing
  def run_synthetic(self):
    if self.tokenizer is not None and self.tokenizer is not False:
      self.build_tokenizer()
      self.build_model()

  # @todo: batch inputs (splat inference.sequence)
  def build_tokenizer(self):
    trace_dump("Initializing tokenizer...")
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
    trace_dump("Using [bold]"+feature_checkp+"[/bold] (CFG="+join_truthy_keys(extra_opts)+")")
    dict_dump(serialize_batch_encoding(self.tx.tokenizer_inputs), indent=None)

  def build_model(self):
    trace_dump("Initializing model...")

    model_checkp = self.model["checkpoint"] if isinstance(self.model, dict) and "checkpoint" in self.model else self.checkpoint
    model_extra_opts = dict(self.model)
    if "checkpoint" in model_extra_opts:
      del model_extra_opts["checkpoint"]
    task = Task(self.task)
    self.model_task = task

    if task == Task.SEQUENCE_CLASSIFICATION:
        self.tx.model = AutoModelForSequenceClassification.from_pretrained(model_checkp, **model_extra_opts)
    elif task == Task.MASKED_LANGUAGE_MODEL:
        self.tx.model = AutoModelForMaskedLM.from_pretrained(model_checkp, **model_extra_opts)
    elif task == Task.TEXT_GENERATION:
        self.tx.model = AutoModelForCausalLM.from_pretrained(model_checkp, **model_extra_opts)
    else:
        raise ValueError("Invalid task identifier specified.")

    trace_dump("Using [bold]"+model_checkp+"[/bold]")

    model_config = self.tx.model.config
    input_count = model_config.num_attention_heads
    token_count = model_config.max_position_embeddings
    model_dimensions = model_config.hidden_size
    model_seq_ctx = {
      "input_count": input_count,
      "token_count": token_count,
      "model_dimensions": model_dimensions
    }

    trace_dump("Listed sequence context for model:")
    dict_dump(model_seq_ctx)

    # Apply HFL modes to trainer and synthetic
    hflmode = HFLMode(self.mode)
    if hflmode == HFLMode.TRAIN or hflmode == HFLMode.EVAL:
      self.dsload()
    elif hflmode == HFLMode.INF: # resume model task
      trace_dump("Piping tokenizer inputs to model...")
      self.tx.pipe_tokens_to_model()
      self.postprocess()
    else:
      pass

  def dsload(self):
    if isinstance(self.dataset, dict):
        name, ds_set, split = self.dataset["name"], self.dataset["set"] if "set" in self.dataset else None, self.dataset["split"] if "split" in self.dataset else None
        trace_dump("Loading dataset: [bold blue]" + name)
        if ds_set is not None:
          self.tx.dataset = load_dataset(name, ds_set)
        else:
          self.tx.dataset = load_dataset(name)
        trace_dump("✓ Dataset loaded & cached...")
        self.train_or_eval()

  def train_or_eval(self):
    hflmode = HFLMode(self.mode)
    cfg_training_args = self.trainer["args"] if "args" in self.trainer else dict()
    trace_dump("With mode: [bold magenta]" + str(hflmode))
    training_args = TrainingArguments(**cfg_training_args)
    trainer = Trainer(
      model=self.tx.model,
      args=training_args,
      train_dataset=self.tx.dataset["train"],
      eval_dataset=self.tx.dataset["validation"],
      data_collator=DataCollatorWithPadding(self.tx.tokenizer_fn),
    )

    if hflmode == HFLMode.TRAIN:
      trace_dump("Training model on dataset...")
      # trainer.train()
      # @fix requires multi-sequence tokenization
      # train_data = Dataset.from_dict({
      #   'input_ids': train_input_ids,
      #   'label': train_labels,
      # })

      # eval_data = Dataset.from_dict({
      #   'input_ids': eval_input_ids,
      #   'label': eval_labels,
      # })
      # trainer.train(train_dataset=train_data, eval_dataset=eval_data)
      trace_dump("✓ Done training. Switch to `eval` for benchmarking/cross-validation.")
      pass
    elif hflmode == HFLMode.EVAL:
      trace_dump("Evaluating model on eval set...")
      results = trainer.evaluate()
      trace_dump("✓ Done evaluation, dumping metrics...")
      print(results)

  def postprocess(self):
    trace_dump("Unpacking model outputs (TASK="+self.model_task.value+")")
    if self.model_task == Task.SEQUENCE_CLASSIFICATION:
        # self.tx.softmax_seqclx_model_output()
        self.tx.argmax_seqclx_model_output()
    elif self.model_task == Task.TEXT_GENERATION:
        self.tx.decode_causal_model_output()
