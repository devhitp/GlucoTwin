import pandas as pd
from datetime import datetime, timezone

dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
ts = pd.Timestamp(dt)
s = {ts}
print("Match?", dt in s)
