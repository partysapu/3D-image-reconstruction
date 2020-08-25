import numpy as np

from raynet.utils.training_utils import dirac_distribution,\
    gaussian_distribution


def get_target_distribution_factory(
    depth_distribution_type,
    stddev_factor=1.0,
    std_is_distance=False
):
    if depth_distribution_type == "dirac":
        return dirac_distribution
    elif depth_distribution_type == "gaussian":
        return gaussian_distribution(stddev_factor, std_is_distance)
    else:
        raise NotImplementedError()


def get_sampling_type(name):
    if "bbox" in name:
        return "sample_points_in_bbox"
    elif "range" in name:
        return "sample_points_in_range"
    elif "disparity" in name:
        return "sample_points_in_disparity"
    elif "voxel_space" in name:
        return "sample_points_in_voxel_space"


class GenerationParameters(object):
    """This class is designed to hold all generation parameters
    """
    def __init__(
        self,
        depth_planes=32,
        neighbors=4,
        patch_shape=(11, 11, 3),
        grid_shape=np.array([64, 64, 32], dtype=np.int32),
        max_number_of_marched_voxels=400,
        expand_patch=True,
        target_distribution_factory=None,
        depth_range=None,
        step_depth=None,
        padding=None,
        sampling_type=None,
        gamma_mrf=None
    ):
        self.neighbors = neighbors
        self.patch_shape = patch_shape
        self.expand_patch = expand_patch
        self.depth_planes = depth_planes
        self.grid_shape = grid_shape
        self.depth_range = depth_range
        self.step_depth = step_depth
        self.padding = padding
        self.sampling_type = sampling_type

        self.target_distribution_factory = target_distribution_factory
        self.max_number_of_marched_voxels = max_number_of_marched_voxels
        self.gamma_mrf = gamma_mrf

    @classmethod
    def from_options(cls, argument_parser):
        # Make Namespace to dictionary to be able to use it
        args = vars(argument_parser)
        # Check if argument_parser contains the grid_shape argument
        grid_shape = args["grid_shape"] if "grid_shape" in args else None
        # Check if argument_parser contains the depth_range argument
        depth_range = args["depth_range"] if "depth_range" in args else None
        # Check if argument_parser contains the step_depth argument
        step_depth = args["step_depth"] if "step_depth" in args else None

        # Check if argument_parser contains the max_number_of_marched_voxels
        # argument
        mnofmv = args["maximum_number_of_marched_voxels"]
        max_number_of_marched_voxels =\
            mnofmv if "maximum_number_of_marched_voxels" in args else None

        # Get the padding value
        patch_shape =\
            args["patch_shape"] if "patch_shape" in args else (None,)*3
        padding =\
            args["padding"] if "padding" in args and args["padding"] is not None else patch_shape[0]

        neighbors = args["neighbors"] if "neighbors" in args else None
        depth_planes = args["depth_planes"] if "depth_planes" in args else None
        if "target_distribution_factory" in args:
            tdf = get_target_distribution_factory(
                argument_parser.target_distribution_factory,
                argument_parser.stddev_factor,
                argument_parser.std_is_distance
            )
        else:
            tdf = None

        # Check if argument_parser contains the sampling_policy argument
        try:
            sampling_type = get_sampling_type(argument_parser.sampling_policy)
        except AttributeError:
            sampling_type = None

        gamma_mrf =\
            args["initial_gamma_prior"] if "initial_gamma_prior" in args else None

        return cls(
            patch_shape=patch_shape,
            depth_planes=depth_planes,
            neighbors=neighbors,
            target_distribution_factory=tdf,
            grid_shape=grid_shape,
            max_number_of_marched_voxels=max_number_of_marched_voxels,
            depth_range=depth_range,
            step_depth=step_depth,
            padding=padding,
            sampling_type=sampling_type,
            gamma_mrf=gamma_mrf
        )

    def to_list2(self):
        l = []
        # print(i.keys())
        # l.append("neighbors")
        l.append(self.neighbors)
        # l.append("neighbors")
        l.append(self.patch_shape)
        # l.append("neighbors")
        l.append(self.expand_patch)
        # l.append("neighbors")
        l.append(self.depth_planes)
        # l.append("neighbors")
        l.append(self.grid_shape)
        # l.append("neighbors")
        l.append(self.depth_range)
        # l.append("neighbors")
        l.append(self.step_depth)
        # l.append("neighbors")
        l.append(self.padding)
        # l.append("neighbors")
        l.append(self.sampling_type)
        # l.append("neighbors")
        l.append(self.target_distribution_factory)
        # l.append("neighbors")
        l.append(self.max_number_of_marched_voxels)
        # l.append("neighbors")
        l.append(self.gamma_mrf)
        # l.append(self.neighbors)
        # print(self.patch_shape)
        # print(self.expand_patch)
        # print(self.depth_planes)
        # print(self.grid_shape)
        # print(self.depth_range)
        # print(self.step_depth)
        # print(self.padding)
        # print(self.sampling_type)
        # print(self.target_distribution_factory)
        # print(self.max_number_of_marched_voxels)
        # print(self.gamma_mrf))
        # for i in self:
        #     print(i)
        #     l.append(i)
        #     l.append('\n')
            # l.append("keys %s : value %f" %(i.keys(),i.values()))
        return l

        # key_idxs = {"loss": 0, "acc": 1, "mde": 2, "mae": 3}
        # for k in self.patch_shape.keys():
        #     # Get the values for that key
        #     d = self.patch_shape[k]
        #     if len(d) > 0:
        #         l[k] = "%.6f" % np.mean(d[:n_samples]) + " - " + "%.6f" % np.mean(d[-n_samples:])

        # for k in self.depth_planes.keys():
        #     # Get the values for that key
        #     d = self.depth_planes[k]
        #     if len(d) > 0:
        #         l[k] = "%.6f" % np.mean(d[:n_samples]) + " - " + "%.6f" % np.mean(d[-n_samples:])

    #     key_idxs = {"val_loss": 4, "val_acc": 5, "val_mde": 6, "val_mae": 7}
    #     for k in self.validation_data.keys():
    #         # Get the values for that key
    #         d = self.validation_data[k]
    #         if len(d) > 0:
    #             l[key_idxs[k]] = "%.6f" % d[0] + " - " + "%.6f" % d[-1]

        # return [x for x in l if x is not None]
