## hfl

![Static Badge](https://img.shields.io/badge/experimental-v0)
![Static Badge](https://img.shields.io/badge/download-git-blue)

**hfl** is a single-file static configuration pipeline tool for HF.

**hfl** only works with common NLP tasks & is configured with PyTorch.

_This is currently experimental (WIP)._

### Motivation

_This section is incomplete._

### Feature Matrix

- [x] Basic pipelines.
- [x] Inherited checkpoints (`model` & `tokenizer`).
- [ ] Multi-sequence tokenization.
- [ ] Training on train datasets (local & AWS SageMaker).
- [ ] Deploying fine-tuned models (`push_to_hub`).
- [ ] Localized metrics from test sets.
- [ ] Specifying custom preprocessor scripts.
- [ ] Integration with custom training loops.

### Installation

Target a global pip installation to this repository.

### Build from Source

#### Windows Compat

You can either `make create_env_win_cmd` or if you require to change the execution policy then `make create_env_win_pow` followed by the relevant load target.

### Usage

_This section is incomplete._
