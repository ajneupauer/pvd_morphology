from datetime import datetime
from pprint import pprint

import h5py
import numpy as np


class ImarisReader:
    """
    Initialize ImarisReader class by supplying a file path to read.
    Loads the corresponding file and its metadata.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file = None
        self.metadata = {}
        self._open_file()
        self._read_metadata()

    """
    Open the ims file at the filepath of the ImarisReader.
    """
    def _open_file(self):
        self.file = h5py.File(self.file_path, "r")
    
    """
    Returns attribute based on its type.
    """
    def _decode_attr(self, attr):
        # If binary encoded, read the text it encodes as utf-8
        if isinstance(attr, bytes):
            return attr.decode("utf-8")
        # If an array of dtype "S", return as a string
        elif isinstance(attr, np.ndarray) and attr.dtype.kind == "S":
            return attr.astype(str)
        # Otherwise, return as is
        return attr

    """
    Given a group within the raw image metadata, return its key value pairs in a dict.
    """
    def _read_attrs(self, group):
        attrs = {}

        for key, value in group.attrs.items():
            decoded_value = self._decode_attr(value)
            # If decoded value is an array
            if isinstance(decoded_value, np.ndarray):
                # If array is dtype "U" or "S", join values into a single string
                if decoded_value.dtype.kind in ["U", "S"]:
                    attrs[key] = "".join(decoded_value)
                # If size 1, just get value
                elif decoded_value.size == 1:
                    attrs[key] = decoded_value.item()
                # If larger size than 1, make into a list
                else:
                    attrs[key] = decoded_value.tolist()
            # Otherwise, just add key value pair to attrs
            else:
                attrs[key] = decoded_value

            # convert to numeric type if possible
            if isinstance(attrs[key], str):
                try:
                    if "." in attrs[key]:
                        attrs[key] = float(attrs[key])
                    else:
                        attrs[key] = int(attrs[key])
                except ValueError:
                    pass  # keep as string if conversion fails

        return attrs

    """
    Get number of image channels, which is not explicitly stated in raw metadata.
    """
    def _count_channels(self):
        dataset_info = self.file["DataSetInfo"]
        # Look for keys with "Channel" in their name and increase channel count for each one
        return sum(
            1 for key in dataset_info.keys() if key.startswith("Channel ")
        )

    """
    Given a dict on timepoints, return them in standard format in a list.
    """
    def _parse_timepoints(self, time_info_attrs):
        timepoints = []
        i = 1
        while f"TimePoint{i}" in time_info_attrs:
            timestamp_str = time_info_attrs[f"TimePoint{i}"]
            try:
                timestamp = datetime.strptime(
                    timestamp_str, "%Y-%m-%d %H:%M:%S.%f"
                )
                timepoints.append(timestamp)
            except ValueError:
                print(
                    f"Warning: unable to parse timestamp for TimePoint{i}: {timestamp_str}"
                )
            i += 1
        return timepoints
    
    """
    Given a list of timepoints, find the average interval between frames in seconds.
    """
    def _calculate_intervals(self, timepoints):
        # Need at least 2 timepoints to compute invervals
        if len(timepoints) < 2:
            return None
        # Differences between consecutive points, in seconds
        intervals = [
            (timepoints[i] - timepoints[i - 1]).total_seconds()
            for i in range(1, len(timepoints))
        ]
        avg_interval = sum(intervals) / len(intervals)
        avg_clipped = "{:.3f}".format(avg_interval)
        return float(avg_clipped)

    """
    Take metadata from the image file and store it in self.metadata.
    Format of self.metadata:
        {
        'image': {'size_x': , 'size_y': , 'size_z': , 'unit': , 'description': , 'recording_date': ,
                  'num_channels': , 'image_origin': , 'image_extent': , 'pixel_sizes': },
        'channels': [one dict per channel of the form 
            {'name': , 'description': , 'emission_wavelength': , 'excitation_wavelength': } ],
        'time_info': {'num_timepoints': , 'timepoints': , 'interval': }
        }
    """
    def _read_metadata(self):
        # Read DataSetInfo metadata
        dataset_info = self.file["DataSetInfo"]

        # Image metadata
        image = dataset_info["Image"]
        image_attrs = self._read_attrs(image)
        num_channels = self._count_channels()

        # Calculate pixel sizes in um
        origin_x = image_attrs["ExtMin0"]
        origin_y = image_attrs["ExtMin1"]
        origin_z = image_attrs["ExtMin2"]
        end_x = image_attrs["ExtMax0"]
        end_y = image_attrs["ExtMax1"]
        end_z = image_attrs["ExtMax2"]
        dx = (end_x - origin_x) / image_attrs["X"]
        dy = (end_y - origin_y) / image_attrs["Y"]
        dz = (end_z - origin_z) / image_attrs["Z"]
        dx = float("{:.4f}".format(dx))
        dy = float("{:.4f}".format(dy))
        dz = float("{:.4f}".format(dz))
        
        # metadata "image" key points to a dict of metadata of the whole image
        self.metadata["image"] = {
            "size_x": image_attrs["X"],
            "size_y": image_attrs["Y"],
            "size_z": image_attrs["Z"],
            "unit": image_attrs["Unit"],
            "description": image_attrs["Description"],
            "recording_date": image_attrs["RecordingDate"],
            "num_channels": num_channels,
            "image_origin": (origin_x, origin_y, origin_z),
            "image_extent": (end_x, end_y, end_z),
            "pixel_sizes": (dx, dy, dz),
        }

        # Channel metadata (info specific to each channel)
        self.metadata["channels"] = []
        for i in range(self.metadata["image"]["num_channels"]):
            channel = dataset_info[f"Channel {i}"]
            channel_attrs = self._read_attrs(channel)
            self.metadata["channels"].append({
                    "name": channel_attrs["Name"],
                    "description": channel_attrs["Description"],
                    #"color": [float(x) for x in channel_attrs["color"].split()],
                    "emission_wavelength": channel_attrs.get("LSMEmissionWavelength"),
                    "excitation_wavelength": channel_attrs.get("LSMExcitationWavelength")
                })

        # Time info
        time_info = dataset_info["TimeInfo"]
        time_info_attrs = self._read_attrs(time_info)
        num_timepoints = time_info_attrs["DatasetTimePoints"]
        timepoints = self._parse_timepoints(time_info_attrs)

        self.metadata["time_info"] = {
            "num_timepoints": time_info_attrs["DatasetTimePoints"],
            "timepoints": timepoints,
            "interval": self._calculate_intervals(timepoints),
        }

    """Returns metadata."""
    def get_metadata(self):
        return self.metadata

    """Get available resolution levels of the image."""
    def get_resolution_levels(self):
        # Each key in dataset is a resolution level
        return list(self.file["DataSet"].keys())

    """
    ***
    Lookup the data corresponding to a given channel at a given timepoint and 
    given resolution level. Optionally return the image array.
    """
    def get_image_data(self, resolution_level=0, time_point=0, channel=0, return_array=False):
        dataset_path = (
            f"/DataSet/ResolutionLevel {resolution_level}/"
            f"TimePoint {time_point}/Channel {channel}/Data"
        )
        dataset = self.file[dataset_path]

        if return_array:
            return np.array(dataset)

        return dataset
    
    """
    Get a 'histogram' of data corresponding to a given channel at a given timepoint 
    and given resolution level.
    """
    def get_histogram(self, resolution_level=0, time_point=0, channel=0):
        histogram_path = (
            f"/DataSet/ResolutionLevel {resolution_level}/"
            f"TimePoint {time_point}/Channel {channel}/Histogram"
        )
        return np.array(self.file[histogram_path])

    """
    Iterator that yields 3D chunks of the image data for all channels.

    :param time_point: The time point to read from
    :param chunk_size: The size of each chunk (z, y, x)
    :param overlap: The overlap between chunks (z, y, x)
    :yield: A tuple containing the chunk coordinates and the chunk data for all channels
    """
    def get_chunk_iterator(
        self, time_point=0, chunk_size=(64, 512, 512), overlap=(0, 0, 0)
    ):
        
        # Get the number of channels and image dimensions
        num_channels = self.metadata["image"]["num_channels"]
        image_size = (
            self.metadata["image"]["size_z"],
            self.metadata["image"]["size_y"],
            self.metadata["image"]["size_x"],
        )

        # Calculate the step size for each dimension
        step_size = tuple(max(1, c - o) for c, o in zip(chunk_size, overlap))

        # Iterate over the image in chunks
        for z in range(0, image_size[0], step_size[0]):
            for y in range(0, image_size[1], step_size[1]):
                for x in range(0, image_size[2], step_size[2]):
                    # Calculate the chunk boundaries
                    z_start, z_end = z, min(z + chunk_size[0], image_size[0])
                    y_start, y_end = y, min(y + chunk_size[1], image_size[1])
                    x_start, x_end = x, min(x + chunk_size[2], image_size[2])

                    # Prepare a list to store chunk data for all channels
                    chunk_data = []

                    # Read chunk data for each channel
                    for channel in range(num_channels):
                        dataset_path = (
                            f"/DataSet/ResolutionLevel 0/TimePoint {time_point}"
                            "/Channel {channel}/Data"
                        )
                        dataset = self.file[dataset_path]
                        channel_chunk = dataset[
                            z_start:z_end, y_start:y_end, x_start:x_end
                        ]
                        chunk_data.append(channel_chunk)

                    # Combine all channel data into a single 4D array (channel, z, y, x)
                    combined_chunk = np.stack(chunk_data)

                    # Yield the chunk coordinates and the combined chunk data
                    yield (
                        ((z_start, z_end), (y_start, y_end), (x_start, x_end)),
                        combined_chunk,
                    )

    """Print gathered image metadata."""
    def info(self):
        pprint(self.metadata, width=80, compact=True)

    """Close the ims file."""
    def close(self):
        if self.file:
            self.file.close()

    def __enter__(self):
        return self
    
    """Wraps close function."""
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
