import warnings
import numpy as np

class XCEventMapping:
    """
    Event-level expansion mapper for HuggingFace datasets.

    This class converts file-level annotations into event-level rows.
    Each detected event becomes an individual dataset entry.

    Intended Usage
    --------------
    Used with HuggingFace `Dataset.map(..., batched=True)`.

    Important:
    - Does NOT store audio arrays.
    - Only keeps filepaths.
    - Expands events per file into separate rows.

    Parameters
    ----------
    biggest_cluster : bool, optional
        If True, only events belonging to the largest event cluster
        per file are retained. Useful for reducing noise or handling
        overlapping detections.
    no_call : bool, optional
        Whether to include no-call events.
        Currently not supported (warning is raised).
    n_time_random_sample_per_file : int, optional
        If > 0, duplicates file entries this many times without
        associated events. Useful for additional random sampling.

    Input Batch Requirements
    ------------------------
    Batch must contain:
        - "filepath"
        - "detected_events"
        - "event_cluster"

    Returns
    -------
    dict
        Expanded batch where each detected event becomes
        its own row.
    """

    def  __init__(self, biggest_cluster: bool = True, no_call: bool = False, n_time_random_sample_per_file: int = 0):

        if no_call:
            warnings.warn(
                f"no_call is not working, skipping including no_calls from bird recordings"
            )
        self.biggest_cluster = biggest_cluster
        self.no_call = no_call
        self.n_time_random_sample_per_file = n_time_random_sample_per_file

    def __call__(self, batch):
        # create new batch to fill: dict with name and then fill with list
        new_batch = {key: [] for key in batch.keys()}

        for b_idx in range(len(batch.get("filepath", []))):


            detected_events = np.array(batch["detected_events"][b_idx])
            detected_cluster = np.array(batch["event_cluster"][b_idx])

            if (
                not (len(detected_cluster) == 1 and detected_cluster[0] == -1)
                or len(detected_cluster) > 1
            ):
                mask = detected_cluster != -1
                detected_events = detected_events[mask]
                detected_cluster = detected_cluster[mask]

            if self.n_time_random_sample_per_file:
                for key in new_batch.keys():
                    for _ in range(self.n_time_random_sample_per_file):
                        if key == "audio":
                            new_batch[key].append(batch["filepath"][b_idx])
                        elif key == "detected_events":
                            new_batch[key].append([])
                        else:
                            v = batch[key][b_idx]
                            new_batch[key].append(
                                v if v != [] else None
                            )

            # check if an event was found
            if len(detected_events) >= 1:
                if self.biggest_cluster:
                    values, count = np.unique(
                        detected_cluster, return_counts=True
                    )  # count clusters!
                    detected_events = detected_events[
                        detected_cluster == values[count.argmax()]
                    ]
                    detected_cluster = detected_cluster[
                        detected_cluster == values[count.argmax()]
                    ]

                detected_events = detected_events.tolist()
                detected_cluster = detected_cluster.tolist()

                for i in range(len(detected_events)):
                    for key in new_batch.keys():
                        if key == "audio":
                            new_batch[key].append(batch["filepath"][b_idx])
                        elif key == "detected_events":
                            new_batch[key].append(detected_events[i])
                        elif key == "event_cluster":
                            new_batch[key].append([detected_cluster[i]])
                        else:
                            new_batch[key].append(batch[key][b_idx])

            else:
                for key in new_batch.keys():
                    if key == "audio":
                        new_batch[key].append(batch["filepath"][b_idx])
                    elif key == "detected_events":
                        new_batch[key].append([0, 5])
                    elif key == "event_cluster":
                        new_batch[key].append(list(detected_cluster))
                    else:
                        v = batch[key][b_idx]
                        new_batch[key].append(
                            v if v != [] else None
                        )

        return new_batch