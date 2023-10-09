import json
from pygments import highlight
from pygments.formatters.terminal256 import Terminal256Formatter
from pygments.lexers.web import JsonLexer

def dict_log(dict):
  raw_json = json.dumps(dict, indent=4)
  highlighted = highlight(
      raw_json,
      lexer=JsonLexer(),
      formatter=Terminal256Formatter(),
  )
  print(highlighted)
