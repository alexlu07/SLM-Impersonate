import os
import re
import pandas as pd
from torch.utils.data import DataLoader

class DataManager:
    def __init__(self, target, path="raw_data"):
        self.target = target
        self.path = path
        self.files = [os.path.splitext(f)[0] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        self.data_types = {
            "reply": self.get_reply_dataset
        }

    def get(self, data_type, subset, **kwargs):
        assert data_type in self.data_types, f"Data type needs to be one of {list(self.data_types.keys())}"

        if type(subset) != list:
            if subset.lower() == "all":
                subset = ".*"

            subset = [subset]

        subset = [pd.read_csv(os.path.join(self.path, f + ".csv")) for f in self.files if any(re.match(p, f) for p in subset)]
        assert len(subset) > 0, "Subset is empty"

        return self.data_types[data_type](subset, **kwargs)

    def get_reply_dataset(self, subset, timeout=None, window=1, **kwargs):
        df = pd.concat(subset, keys=range(len(subset)))
        df = df.reset_index(level=0, names="Source")

        df["Content"] = df["Content"].fillna("")
        df["Formatted"] = df.apply(lambda row: [{"role": row["Author"], "content": row["Content"]}], axis=1)

        df["Time"] = pd.to_datetime(df["Date"], utc=True)
        diff_time = df["Time"].diff().gt(pd.Timedelta(minutes=timeout)) if timeout else False
        diff_source = df["Source"] != df["Source"].shift()

        df["Group"] = (diff_time | diff_source).cumsum()
        
        df["chat"] = df.groupby("Group", group_keys=False)["Formatted"].apply(lambda group: pd.Series(group.rolling(window=window, min_periods=1), index=group.index).apply(lambda x: x.sum()))
        df["reply"] = (df["Author"].shift(-1) == self.target) & (df["Group"] == df["Group"].shift(-1))

        return df[["chat", "reply"]]