import yaml
import loader

HFLSPEC_PATH = "hflspec.yml"

if __name__ == "__main__":
  try:
    with open(HFLSPEC_PATH, "r") as yaml_file:
      data = yaml.load(yaml_file, Loader=yaml.FullLoader)
      l = loader.Loader(spec=data)
      print(l.spec)
  except FileNotFoundError:
      print(f"The file '{HFLSPEC_PATH}' was not found.")
  except Exception as e:
      print(f"An error occurred: {e}")
