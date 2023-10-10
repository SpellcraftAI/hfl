import json
from rich.console import Console
from rich.syntax import Syntax

console = Console()

def join_truthy_keys(input):
  return ", ".join([key for key, value in input.items() if value is True])

def rule_dump(tagline):
  console.rule(tagline)

def trace_dump(msg):
  console.log(msg)

def dict_dump(dict, indent=2):
  raw_json = json.dumps(dict, indent=indent)
  syntax = Syntax(raw_json, "json", theme="lightbulb", line_numbers=True, word_wrap=True, background_color="#000000")
  console.print(syntax)
