import metadata

class Loader:
  def __init__(self, spec) -> None:
    self.metadata = metadata.Metadata()
    self.spec = spec
    self.write_meta()
    self.metadata.run()
  
  def write_meta(self):
    if "task" in self.spec:
      self.metadata.task = self.spec["task"]
    if "checkpoint" in self.spec:
      self.metadata.checkpoint = self.spec["checkpoint"]
    if "pipeline" in self.spec:
      self.metadata.pipeline = self.spec["pipeline"]
    else:
      self.metadata.pipeline = False
    if "inference" in self.spec:
      if "sequence" in self.spec["inference"]:
        self.metadata.sequence = self.spec["inference"]["sequence"]
