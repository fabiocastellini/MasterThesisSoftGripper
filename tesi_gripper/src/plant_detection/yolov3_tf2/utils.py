from absl import logging
import numpy as np
import tensorflow as tf
import cv2

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]


def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def draw_outputs(img, outputs, class_names):

    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]

    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


def draw_labels(x, y, class_names):
    img = x.numpy()
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, class_names[classes[i]],
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          1, (0, 0, 255), 2)
    return img


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)




#My functions:
"""
extract bounding box coordinates from raw model outputs 
parameters:
    modelOuput: tensor, of shape (None, conv H, conv W, num anchors * (num class + 5))
    anchors: tensor, of shape (num anchors, 2)
    num class: int, number of classes 
returns:
    boxXY: tensor, of shape (None, conv H, conv W, num anchors, 2). The coordinates of box centers
    boxWH: tensor, of shape (None, conv H, conv W, num anchors, 2). The width and height of boxes
    objScore: tensor, of shae (None, conv H, conv W, num anchors, 1). The obj score of each anchor box 
    classProb: tensor, of shape (None, conv H, conv W, num anchors, num class). The class probabilities 
"""


def extractInfo(modelOutput, anchors, numClass):
    featureDim = modelOutput.shape
    numAnchor = 7 #anchors.shape[0]  # get the number of anchors, 5 for the Pascal dataset
    modelOutput = tf.reshape(modelOutput, shape=(-1, featureDim[1], featureDim[2], numAnchor, numClass + 5))
    """
    Now modelOutput has shape (-1, grid num W, grid num H, anchor num, 5+class num)
    For bounding box k in grid (i, j) in picture n, it is in stored in modelOutput[n, i, j, k, :]
    """
    imageShape = featureDim[1:3]  # get the width and height of output feature map

    """
    step 1: pass tx, ty through sigmoid and offset by grid coordinates so that we get bx, by
    Let's assume the raw boxXY = (0,0) and it's in grid (0,1)
    """
    boxXY = tf.nn.sigmoid(modelOutput[..., :2])  # boxXY now w.r.t top left corner of its grid(on grid scale)
    """
    now boxXY = (0.5, 0.5). It means it's 0.5 (grid weight) and 0.5(grid height) away from the top left corner of grid (0,1)
    the red dot in figure 3 shows its location
    """
    idx = getOffset(imageShape)  # convert box center to grid scale
    idx = tf.cast(idx, modelOutput.dtype)
    anchors = tf.cast(tf.reshape(anchors, (1, 1, 1, numAnchor, 2)), idx.dtype)
    boxXY = (boxXY + idx)
    """
    idx essentially converts the boxXY coordinates from w.r.t its own grid(0,1) to the black top left corner of gird (0,0)
    now boxXY = (0.5,0.5) + (0,1) = (0.5, 1.5), meaning it's 0.5 grid height and 1.5 grid width from tht black dot 
    for two boxes k1 and k2, their local overlapps iff boxXY 1 == boxXY 2 & they are in the same grid
    """

    """
    step 2: convert box width and hight 
    let's assume boxWH = (0.4, 0.4) and its anchor box has size (0.75, 0.5), which is shown as the blue box
    """
    boxWH = tf.math.exp(modelOutput[..., 2:4])
    """
    Now boxWH = (1.5, 1.5), meaning it's 1.5 times wider and taller than the anchor box
    """
    boxWH = boxWH * anchors
    """
    Now boxWH = (1.13, 0.75), meaning its width is 0.75 unit of grid width and height is 1.13 units of grid height, shown as
    the red box in figure 3. As you can see, for different anchors, boxWH = (1.13, 0.75) means different sizes
    """
    objScore = tf.nn.sigmoid(modelOutput[..., 4:5])  # objectiveness score; must be between 0 and 1
    classProb = tf.nn.softmax(
        modelOutput[..., 5:])  # probability of classes; pass through a softmax gate to obtain prob.

    return boxXY, boxWH, objScore, classProb


"""
generate grid offset for a given shape 
parameters:
    shape: tuple of length 2, [conv height, conv weidth]
returns:
    offset: tensor of shape [1. conv H, conv W, 1, 2]. 
"""


def getOffset(shape):
    hIndex = tf.reshape(tf.range(start=0, limit=shape[0]), (shape[0], 1))
    hIndex = tf.tile(hIndex, [1, shape[1]])  # expand in the height direction
    wIndex = tf.reshape(tf.range(start=0, limit=shape[1]), (1, shape[1]))
    wIndex = tf.tile(wIndex, [shape[0], 1])  # expand in the width direction
    idx = tf.stack([wIndex, hIndex], axis=-1)
    idx = tf.reshape(idx, shape=(1, *shape, 1, 2))  # reshape the offset so that it can add to boxXY directly
    return idx




"""
convert the box center and weight to top left and bottom right corners 
parameters: 
    boxXY: tensor, of shape (None, conv H, conv W, num anchors, 2). The coordinates of box centers w.r.t the top left grid center
    boxWH: tensor, of shape (None, conv H, conv W, num anchors, 2). The width and height of boxes on grid scale
returns:
    boxLoc: tensor, of shape (None, conv H, conv W, num anchors, 4). The top left and bottom right coordinates on grid scale
"""

def getBoxLoc(boxXY, boxWH):
    topLeft = boxXY - boxWH / 2  # top left
    bottomRight = boxXY + boxWH / 2  # bottom right
    # the last dimension is (x1, y1, x2, y2)
    # top left means it is closer to (0,0) in the image, which is the top-left corner
    # if displayed by matplotlib
    return tf.concat([topLeft, bottomRight], axis=-1)

"""
scale the boxes to full image scale 
parameters: 
    boxLoc: tensor, of shape (None, conv H, conv W, num anchors, 4). The top left and bottom right coordinates 
    scale: tuple, value = (32, 32). This is because conv dim = img dim // 32 
returns:
    boxLoc: tensor, of shape (None, conv H, conv W, num anchors, 4). scaled version of boxLoc
"""

def scaleBox(boxLoc, scale=(32, 32)):
    height, width = scale[0], scale[1]
    shape = tf.stack([height, width, height, width])
    shape = tf.reshape(shape, [1, 4])
    shape = tf.cast(shape, boxLoc.dtype)
    return boxLoc * shape


"""
filter out boxes with low object score 
parameters:
    boxLoc: tensor, of shape (None, conv H, conv W, num anchors, 4). The top left and bottom right coordinates 
    objScore: tensor, of shae (None, conv H, conv W, num anchors, 1). The obj score of each anchor box 
    classProb: tensor, of shape (None, conv H, conv W, num anchors, num class). The class probabilities 
    scoreThresh: filter threshold
returns 
    boxes: list of boxes, [box1, box2, ...]
    objScore: list of scores, [score1, score2, ...]
    classProb: list of class probability, [prob1, prob2, ...]
"""


def filterBox(boxLoc, objScore, classProb, scoreThresh=0.5):
    boxScore = objScore * classProb  # (None, B1, B2, S, NCLASS)
    boxClass = tf.argmax(boxScore, axis=-1)  # shape = (None, S, S, B)
    boxScore = tf.math.reduce_max(boxScore, axis=-1)  # shape = (None, S, S, B)
    mask = boxScore >= scoreThresh
    # filter out low-confidence boxes
    boxes = tf.boolean_mask(boxLoc, mask)
    scores = tf.boolean_mask(boxScore, mask)
    classes = tf.boolean_mask(boxClass, mask)
    """
    please note that tf.boolean_maks returns a list rather than maintaining the original shapes. In
    other words, boxes is a list of boxes with no specific ordering. box[0] could be the coordinates of a box in grid 1,1 or 0,0
    """
    return boxes, scores, classes


"""
filter out boxes that are significantly overlapped. Only preserve the box with the highest score.
This function should be called after filterBox to reduce workload
parameters:
    boxLoc: list of tensor, [box1, box2, ...]
    objScore: lit of tensor, [score1, score 2]
    iouThresh: float, filter threshold 
returns:
    boxLoc: list of tensor, [box1, box2, ...] boxes left 
    objScore: lit of tensor, [score1, score 2] their scores 
"""


def nonMaxSuppress(boxLoc, score, classPredict, maxBox=20, iouThresh=0.5):
    idx = tf.image.non_max_suppression(boxLoc, score, maxBox, iou_threshold=iouThresh)
    boxLoc = tf.gather(boxLoc, idx)
    score = tf.gather(score, idx)
    classPredict = tf.gather(classPredict, idx)
    return boxLoc, score, classPredict


"""
convert raw yolo output to a list of boxes
parameters:
    featureMap: tensor, (None, conv H, conv W, num anchors * (num class + 5))
    anchors:  tensor, (num anchors, 2) [[w, h],[w, h],...]
    numClass: int, the number of classes 
    maxBox: int, the maximum number of boxes returned 
    scoreThresh: float, used to filter out low confidence boxes 
    iouThresh: float, used to filter out boxes that are significantly overlapped 
returns:
    generator of values: 
        boxes: list of boxes for one image [[left top right bottom], [], []]
        scores: list of scores for those boxes [score1, score2, ...]
        classes: class prediction [pred1, pred2, ...]

"""


def raw2Box(featureMap, anchors, numClass, maxBox=20, scoreThresh=0.5, iouThresh=0.5):
    # convert coordinates
    print('YOLO output shape: ', featureMap.shape)
    batchXY, batchWH, batchScore, batchProb = extractInfo(featureMap, anchors, numClass)
    # convert boxXY,boxWH to corner coordinates
    batchLoc = getBoxLoc(batchXY, batchWH)
    # scale the coordinates from grid scale to image scale
    batchLoc = scaleBox(batchLoc)
    print('Processed box coordinate shape: ', batchLoc.shape)
    # a for loop is needed because different images have various number of boxes
    # for each image, let's do:
    for boxLoc, objScore, classProb in zip(batchLoc, batchScore, batchProb):
        # filter out low confidence boxes
        boxes, scores, classes = filterBox(boxLoc, objScore, classProb, scoreThresh)
        # filter out overlapped boxes
        boxes, scores, classes = nonMaxSuppress(boxes, scores, classes, maxBox, iouThresh)
        # return a list of boxes for that image
        print('box count: ', len(boxes))
        yield boxes, scores, classes