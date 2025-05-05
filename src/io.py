"""
Low-level I/O helpers.

Current slice: only a generator that streams (x,y,t,polarity) tensors
from an AEDAT4 recording.  Later we’ll add live-camera support here.
"""
from pathlib import Path
from typing import Iterator, Tuple, Union

import dv_processing as dv
import torch


def stream_aedat(
    path: Union[str, Path],
    *,
    batch_us: int = 10_000,
) -> Iterator[Tuple[torch.Tensor, Tuple[int, int]]]:
    """
    Yields (events, (H, W)) until the file ends.

    events: int/bool tensor of shape (N, 4) – columns = (x, y, t[µs], pol)
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)

    reader = dv.io.MonoCameraRecording(str(path))

    if not reader.isEventStreamAvailable():
        raise RuntimeError("Recording has no event stream")

    width, height = reader.getEventResolution()
    res_hw = (height, width)

    # Consume the file chunk by chunk.
    while reader.isRunning():
        evs = reader.getNextEventBatch()
        if evs is None or evs.size() == 0:
            continue  # keep trying

        np_events = evs.numpy()

        xs = torch.from_numpy(np_events["x"]).int()
        ys = torch.from_numpy(np_events["y"]).int()
        ts = torch.from_numpy(np_events["timestamp"]).long()  # µs
        ps = torch.from_numpy(np_events["polarity"]).bool()

        events_tensor = torch.cat(
            (
                xs.unsqueeze(1),
                ys.unsqueeze(1),
                ts.unsqueeze(1),
                ps.unsqueeze(1).to(torch.uint8),  # keep 0/1 but no up-cast
            ),
            dim=1,
        )
        yield events_tensor, res_hw
