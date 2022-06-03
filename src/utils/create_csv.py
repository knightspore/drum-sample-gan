import os
import pandas as pd


class CreateCSV:
    def __init__(self, path):
        self.classes = {
            "kick": [],
            "snare": [],
            "hat": [],
            "clap": [],
            "ride": [],
            "crash": [],
            "tom": [],
            "perc": [],
            "cowbell": [],
            "clave": [],
        }
        self.classID = {
            "kick": 0,
            "snare": 1,
            "hat": 2,
            "clap": 3,
            "ride": 4,
            "crash": 5,
            "tom": 6,
            "perc": 7,
            "cowbell": 8,
            "clave": 9,
        }
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".wav") and "LOOP" not in file.upper():
                    for k in list(
                        self.classes.keys()
                    ):  # This search function can be improved
                        if k.upper() in file.upper():
                            self.classes[k].append(
                                {"path": os.path.join(root, file), "filename": file,}
                            )

    def __repr__(self):  # Print Summary
        list = ""
        for _class in self.classes:
            list += f"{_class}: {len(self.classes[_class])}\n"
        return list

    def csv(self, path):
        df = pd.DataFrame()

        for _class in self.classes:
            i = 0
            for sample in self.classes[_class]:
                if i < 10:
                    df = df.append(
                        {
                            "sample_file_name": sample["filename"],
                            "path": sample["path"].replace("\\", "/"),
                            "classID": _class,
                            "class": self.classID[_class],
                        },
                        ignore_index=True,
                    )
                    i += 1
                else:
                    break
        df = df.sample(frac=1).reset_index(drop=True)
        df.to_csv(path, index=False)
