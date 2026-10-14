"""Two native Qwen visual resolutions without resizing or changing model inputs.

The fine map is the final frozen visual-block output before patch merging.
The coarse map is the trainable merger output in restored spatial order.
This is an explicit native-VLM adaptation, not the paper's Swin pyramid.
"""
import torch


def restore_native_maps(fine_window_order, coarse_spatial_order, grid_thw, window_index, merge_size=2):
    if tuple(grid_thw.shape) != (1, 3):
        raise ValueError('Exactly one native image is required')
    temporal, height, width = map(int, grid_thw[0].tolist())
    if temporal != 1 or height % merge_size or width % merge_size:
        raise ValueError('Unsupported temporal or spatial native grid')
    coarse_height, coarse_width = height // merge_size, width // merge_size
    groups = coarse_height * coarse_width
    if fine_window_order.ndim != 2 or fine_window_order.shape[0] != height * width:
        raise ValueError('Fine features do not match the native patch grid')
    if coarse_spatial_order.ndim != 2 or coarse_spatial_order.shape[0] != groups:
        raise ValueError('Coarse features do not match the native merger grid')
    if tuple(window_index.shape) != (groups,) or not torch.equal(
            window_index.sort().values.cpu(), torch.arange(groups)):
        raise ValueError('Window index is not a complete spatial permutation')
    inverse = window_index.argsort().to(fine_window_order.device)
    fine = fine_window_order.reshape(groups, merge_size * merge_size, -1)[inverse]
    fine = fine.reshape(coarse_height, coarse_width, merge_size, merge_size, -1)
    fine = fine.permute(4, 0, 2, 1, 3).reshape(-1, height, width)
    coarse = coarse_spatial_order.T.reshape(-1, coarse_height, coarse_width)
    return fine, coarse


class NativeFeatureCapture:
    """Observe the existing forward path; never replace or detach its tensors."""
    def __init__(self, model):
        self.visual = model.model.visual
        if type(self.visual).__name__ != 'Qwen2_5_VisionTransformerPretrainedModel':
            raise ValueError('Unexpected native visual architecture')
        if self.visual.spatial_merge_size != 2:
            raise ValueError('Unexpected native merge size')
        self.grid = self.fine = self.coarse = None
        self.handles = [self.visual.merger.register_forward_pre_hook(self._fine),
                        self.visual.register_forward_hook(self._coarse)]

    def _fine(self, module, arguments):
        if self.grid is not None:
            self.fine = arguments[0]

    def _coarse(self, module, arguments, output):
        if self.grid is not None:
            self.coarse = output

    def begin(self, grid):
        self.grid, self.fine, self.coarse = grid, None, None

    def maps(self):
        if self.grid is None or self.fine is None or self.coarse is None:
            raise ValueError('A complete native vision forward must precede localization')
        index, _ = self.visual.get_window_index(self.grid)
        return restore_native_maps(self.fine, self.coarse, self.grid, index, self.visual.spatial_merge_size)

    def clear(self):
        self.grid = self.fine = self.coarse = None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.clear()


def native_fusion_forward(model, batch, cell_ranges, capture):
    """Run native generation plus annotation-free localization in one forward.

    cell_ranges may come from the teacher-forced target or from generated OTSL.
    Geometry, logical spans, and relation labels never enter this function.
    All 28 decoder-layer outputs are used, excluding the input embeddings;
    the native model returns its final layer after the final RMS normalization.
    """
    capture.begin(batch['image_grid_thw'])
    output = model(**{k:v for k,v in batch.items() if k != 'labels'},
                   use_cache=False, output_hidden_states=True)
    states = output.hidden_states[1:]
    if len(states) != model.tdatr_localizer.decoder_layers:
        raise ValueError('Native decoder layer count differs')
    if any(state.shape[0] != 1 for state in states):
        raise ValueError('Native fusion expects batch size one')
    # FP32 localization master weights and sampling avoid unsupported CPU/BF16
    # grid sampling and give stable regression losses under BF16 model forwards.
    with torch.autocast(device_type=states[0].device.type, enabled=False):
        prediction = model.tdatr_localizer([state[0].float() for state in states],
            cell_ranges, [feature.float() for feature in capture.maps()])
    return output, prediction
