#!/usr/bin/env python
"""Entry point for the final all-channel grand-average report.

This wrapper keeps the analysis implementation in G04. It also supplies the small
array-grand-average helper used by G04's ERP section. Keeping this helper here fixes
the missing-name error without changing any TFR or ratio calculations.
"""
import numpy as np
import G04_complete_grand_average_report as pipeline


def grand_arrays(items):
    """Mean subject-level single-channel MNE objects; return waveform and times."""
    if not items:
        raise ValueError("grand_arrays requires at least one subject-level object")
    times = np.asarray(items[0].times).copy()
    arrays = []
    for item in items:
        if len(item.times) != len(times) or not np.allclose(item.times, times):
            raise ValueError("Subject-level objects have inconsistent time axes")
        data = np.asarray(item.data)
        arrays.append(data[0] if data.ndim == 2 and data.shape[0] == 1 else np.squeeze(data))
    return np.mean(arrays, axis=0), times


pipeline.grand_arrays = grand_arrays

if __name__ == "__main__":
    pipeline.main()
