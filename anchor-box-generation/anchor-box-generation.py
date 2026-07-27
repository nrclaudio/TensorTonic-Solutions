import numpy as np
def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    scales = np.asarray(scales)
    aspect_ratios = np.asarray(aspect_ratios)
    stride = image_size / feature_size
    n_boxes = (feature_size*2) * scales * aspect_ratios
    bboxes = []
    for i in range(feature_size):
        for j in range(feature_size):
            center_x = (j + 0.5)*stride
            center_y = (i + 0.5)*stride
            for scale in scales:
                for ratio in aspect_ratios:
                    width = scale * np.sqrt(ratio)
                    height = scale / (np.sqrt(ratio))
                    bbox = [center_x - width/2, center_y - height/2, center_x + width/2, center_y + height/2]
                    bboxes.append(bbox)

    return bboxes   