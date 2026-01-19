import csv, json, os

class CSVLogger:
    def __init__(self, path):
        self.path = path

    def log(self, data):
        write_header = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(data)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def write_description(path, text):
    with open(path, "w") as f:
        f.write(text)